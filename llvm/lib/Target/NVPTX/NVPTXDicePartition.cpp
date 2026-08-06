//===- NVPTXDicePartition.cpp - split MBBs into DICE p-graphs ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DICE executes straight-line dataflow regions ("p-graphs") mapped spatially
// onto a CGRA: every instruction of a p-graph occupies a functional unit at
// once, loads write back only at the end of the region, and a barrier can
// only stand at a region's head. This pass splits machine basic blocks so
// that EVERY basic block is a legal p-graph, and forces a label on each, so
// the emitted PTX's labels are exactly the p-graph boundaries. The DICE
// toolchain (register renaming, .meta emission) then consumes the PTX
// trusting the labels instead of re-partitioning.
//
// The rules mirror the reference partitioner
// (dice-compiler/src/lib/opt/ptx_non_ilp/DICENonILP.py::_partitionBlocks):
//
//   1. A barrier must be the first instruction of its p-graph.
//   2. An instruction that uses the result of a load or atomic issued in the
//      current p-graph starts a new one (the LDST unit writes results back
//      at the end of the region). Loads from the kernel parameter space are
//      exempt: they cost no LDST port and their value is available.
//   3. A p-graph holds at most pe-count PE ops, sfu-count SFU ops and
//      ldst-count LDST accesses. Costs: ld/st/atom = 1 LDST (param loads
//      free); div/sqrt-family = 1 SFU; cvta and 32<->64 integer cvt = free;
//      mov and special-register reads = free; everything else = 1 PE.
//   4. Branches and returns are terminators, so a block boundary already
//      exists there.
//
// This pass runs immediately before the asm printer: NVPTX keeps virtual
// registers to the end (there is no register allocator), so def-use chains
// are still explicit and instructions map 1:1 to emitted PTX.
//
//===----------------------------------------------------------------------===//

#include "NVPTX.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/NVPTXAddrSpace.h"

using namespace llvm;

static cl::opt<bool> EnableDicePartition(
    "nvptx-dice-partition", cl::init(false), cl::Hidden,
    cl::desc("Split basic blocks into DICE p-graphs and label every block"));

static cl::opt<unsigned> DicePECount(
    "nvptx-dice-pe-count", cl::init(16), cl::Hidden,
    cl::desc("DICE p-graph PE budget"));

static cl::opt<unsigned> DiceSFUCount(
    "nvptx-dice-sfu-count", cl::init(4), cl::Hidden,
    cl::desc("DICE p-graph SFU budget"));

static cl::opt<bool> EnableDiceFuse(
    "nvptx-dice-opt-fuse", cl::init(false), cl::Hidden,
    cl::desc("DICE: fuse adjacent p-graphs when the merged block is legal"));

static cl::opt<unsigned> DiceLDSTCount(
    "nvptx-dice-ldst-count", cl::init(4), cl::Hidden,
    cl::desc("DICE p-graph LDST port budget"));

static cl::opt<unsigned> DiceConstReadCount(
    "nvptx-dice-const-reads", cl::init(0), cl::Hidden,
    cl::desc("DICE p-graph distinct constant-file read budget; 0 (default) "
             "= unenforced. The RF-to-CGRA path is a full 60xN crossbar "
             "(32 banks + 16 const + 12 special), so no limit today; set "
             "alongside the device's const_reads_per_pgraph to re-arm C15"));

namespace {

struct Cost {
  unsigned PE = 0, SFU = 0, LDST = 0;
  bool IsBarrier = false;
  // The def of a load or atomic is not readable inside the p-graph that
  // issued it.
  bool DefsArriveLate = false;
};

class NVPTXDicePartition : public MachineFunctionPass {
public:
  static char ID;
  NVPTXDicePartition() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return "NVPTX DICE p-graph partitioning";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
};

char NVPTXDicePartition::ID = 0;

} // namespace

/// Is MI positioned inside [Entry.begin(), LeadEnd)?
static bool InLeadingRun(const MachineInstr &MI, MachineBasicBlock &Entry,
                         MachineBasicBlock::iterator LeadEnd) {
  for (auto It = Entry.begin(); It != LeadEnd; ++It)
    if (&*It == &MI)
      return true;
  return false;
}

/// True when every memory operand reads the kernel parameter space. A load
/// with no memory operands is treated as a real load: missing MMOs mean
/// missing information, and the conservative direction is to charge a port.
static bool isParamLoad(const MachineInstr &MI) {
  if (MI.memoperands_empty())
    return false;
  for (const MachineMemOperand *MMO : MI.memoperands())
    if (MMO->getAddrSpace() != NVPTXAS::ADDRESS_SPACE_ENTRY_PARAM)
      return false;
  return true;
}

/// The SFU carries the operations the reference compiler routes there:
/// div, rcp, rsqrt, sqrt, sin, cos, exp, ex2, lg2, log. rem stays on a PE
/// for parity with the reference (SFU_OP_PREFIXES in DICENonILP.py).
static bool isSFUName(StringRef Name) {
  return Name.contains("DIV") || Name.contains("SQRT") ||
         Name.contains("SIN_") || Name.contains("COS_") ||
         Name.starts_with("SINF") || Name.starts_with("COSF") ||
         Name.contains("EX2") || Name.contains("LG2") ||
         Name.contains("RCP");
}

/// cvt between 32- and 64-bit integers is a width change on a fabric whose
/// datapath is 32 bits wide -- after address narrowing it disappears, so it
/// must not be charged a PE. Names look like CVT_u64_u32 / CVT_s32_s64.
static bool isFreeIntWidthCvt(StringRef Name) {
  if (!Name.consume_front("CVT_"))
    return false;
  auto ParseTy = [](StringRef T, unsigned &Width) {
    if (!T.consume_front("u") && !T.consume_front("s"))
      return false;
    return !T.getAsInteger(10, Width) && (Width == 32 || Width == 64);
  };
  auto [Dst, Src] = Name.split('_');
  unsigned DstW = 0, SrcW = 0;
  return ParseTy(Dst, DstW) && ParseTy(Src.split('_').first, SrcW) &&
         DstW != SrcW;
}

static Cost classify(const MachineInstr &MI, const TargetInstrInfo &TII,
                     bool InPureParamBlock) {
  Cost C;
  StringRef Name = TII.getName(MI.getOpcode());

  if (Name.starts_with("BARRIER") || Name.starts_with("BAR_")) {
    C.IsBarrier = true;
    return C;
  }

  // Atomics: read-modify-write. One LDST port, and like a load the returned
  // old value only lands at the end of the p-graph.
  if (MI.mayLoad() && MI.mayStore()) {
    C.LDST = 1;
    C.DefsArriveLate = true;
    return C;
  }
  if (MI.mayLoad()) {
    if (!isParamLoad(MI)) {
      C.LDST = 1;
      C.DefsArriveLate = true;
    } else if (!InPureParamBlock) {
      // Hardware loads parameters ONCE per CTA into the constant RF -- but
      // only for IS_PARAMETER_LOAD blocks (dispatch-once). A param load
      // stranded in a normal block (un-hoistable: multi-def destination)
      // executes per thread like any load and must pay for its port; the
      // meta checker (C6) already charges it, and the partitioner
      // disagreeing is how bfs overflowed.
      C.LDST = 1;
      C.DefsArriveLate = true;
    }
    return C;
  }
  if (MI.mayStore()) {
    C.LDST = 1;
    return C;
  }

  if (isSFUName(Name)) {
    C.SFU = 1;
    return C;
  }

  // Free: address-space casts, integer width changes, movs (the fabric
  // routes them), special-register reads (mov.u32 %r, %tid.x).
  if (Name.starts_with("CVTA") || isFreeIntWidthCvt(Name) ||
      Name.starts_with("MOV") || Name.starts_with("IMOV") ||
      Name.starts_with("INT_PTX_SREG"))
    return C;

  // fp64 COSTS AN SFU TILE, and this line was missing until 2026-08-06.
  //
  // Only the four `pe_sfu` tiles carry the width-64 fpnew instance
  // (device-contract.md 5.5e); the twenty ALU tiles have the 32-bit one and
  // nothing else. `dicemap.pgraph.classify_op_class` has said so since the
  // fp64 unit was built -- anything whose PTX text contains `.f64` is
  // op_class `float64`, capable on `sfu` only -- and the PARTITIONER did not
  // agree: it charged an ordinary PE and let a p-graph hold as many fp64 ops
  // as it had PE tiles.
  //
  // It went unnoticed because it needed a p-graph with FIVE fp64 ops to bite,
  // and while `fma.rn.f64` was one instruction backprop's worst block had
  // exactly four. Lowering the FMA (NVPTXDiceFP64Lower) turns one op into two
  // and made `DICE_BB1_8` and `DICE_BB1_15` need five SFU tiles of the four
  // that exist -- the mapper reported `resource_oversubscribed`, which is the
  // right answer to the wrong question: the partitioner should have split
  // them. Measured on backprop and hotspot.
  //
  // The predicate mirrors the mapper's `_WIDE_FLOAT_TY = \.f64\b`: any
  // operation whose PTX names `.f64`. `MOV`/`CVTA` are already free above, so
  // an fp64 register copy stays free exactly as the mapper's fold makes it.
  if (Name.contains("f64") || Name.contains("F64")) {
    C.SFU = 1;
    return C;
  }

  C.PE = 1;
  return C;
}

// Exported: is the DICE pipeline on? Other components (the IR pass setup
// most of all) shape their behavior on it.
bool llvm::nvptxDiceEnabled() { return EnableDicePartition; }

// Exported: the DICE register allocator re-runs partitioning after it
// inserts spill code -- a reload is an LDST access whose consumer must move
// to a later p-graph, and stores can overflow the LDST budget.
bool llvm::nvptxDiceRepartition(MachineFunction &MF) {
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  bool Changed = false;

  // Hoist parameter loads to the entry block. Clang sinks a ld.param next to
  // its use, which can strand it mid-kernel -- even under a predicate. The
  // DICE toolchain expects nvcc's shape, where every parameter load leads the
  // function and is folded into a dedicated free parameter block; a param
  // load left inside a busy p-graph gets its destination counted against the
  // LDST writeback budget by the artifact checker. Hoisting is sound when
  // the load reads only immutable param memory through a static address
  // (no register operands) and its destination has a single definition.
  {
    MachineBasicBlock &Entry = MF.front();
    MachineRegisterInfo &MRI = MF.getRegInfo();
    // The park position: right after the entry block's LEADING run of
    // param loads. Anything already in that run stays put; a param load
    // anywhere else -- including later in the entry block itself, which a
    // C6 failure on backprop and b+tree proved really happens -- moves
    // here.
    auto LeadEnd = Entry.begin();
    for (auto It = Entry.begin(); It != Entry.end(); ++It) {
      if (It->isMetaInstruction() || It->isDebugInstr())
        continue;
      if (It->mayLoad() && !It->mayStore() && isParamLoad(*It)) {
        LeadEnd = std::next(It);
        continue;
      }
      break;
    }
    SmallVector<MachineInstr *, 16> ParamLoads;
    for (MachineBasicBlock &MBB : MF)
      for (MachineInstr &MI : MBB) {
        if (!MI.mayLoad() || MI.mayStore() || !isParamLoad(MI))
          continue;
        if (&MBB == &Entry && InLeadingRun(MI, Entry, LeadEnd))
          continue;
        bool Movable = true;
        Register Def;
        for (const MachineOperand &MO : MI.operands()) {
          if (!MO.isReg())
            continue;
          if (MO.isDef()) {
            Def = MO.getReg();
          } else {
            Movable = false; // register-based param address: leave it alone
            break;
          }
        }
        if (Movable && Def.isVirtual() && MRI.hasOneDef(Def))
          ParamLoads.push_back(&MI);
      }
    for (MachineInstr *MI : ParamLoads) {
      MI->getParent()->remove(MI);
      Entry.insert(LeadEnd, MI);
      Changed = true;
    }
    // Give the leading parameter loads their own block, the reference
    // toolchain's IS_PARAMETER_LOAD block: the runtime preloads it and it
    // costs no fabric resources, so it must not share a p-graph with real
    // work.
    MachineInstr *LastParam = nullptr;
    for (MachineInstr &MI : Entry) {
      if (MI.isMetaInstruction() || MI.isDebugInstr())
        continue;
      if (MI.mayLoad() && !MI.mayStore() && isParamLoad(MI)) {
        LastParam = &MI;
        continue;
      }
      break;
    }
    if (LastParam && &*std::prev(Entry.end()) != LastParam) {
      MachineBasicBlock *Tail =
          Entry.splitAt(*LastParam, /*UpdateLiveIns=*/false);
      Tail->setLabelMustBeEmitted();
      Changed = true;
    }
  }

  // Worklist of blocks still to be scanned. A split appends the tail as a
  // Values the register allocator will place in the constant file: the
  // single-def destinations of parameter loads. A p-graph may read at most
  // DiceConstReadCount distinct ones -- the muxed boundary ports (C15).
  DenseSet<Register> ParamConstDefs;
  {
    const MachineRegisterInfo &MRI = MF.getRegInfo();
    for (const MachineBasicBlock &MBB : MF)
      for (const MachineInstr &MI : MBB)
        if (MI.mayLoad() && !MI.mayStore() && isParamLoad(MI))
          for (const MachineOperand &MO : MI.defs())
            if (MO.isReg() && MO.getReg().isVirtual() &&
                MRI.hasOneDef(MO.getReg()))
              ParamConstDefs.insert(MO.getReg());
  }

  // fresh block whose own budget starts from zero, so it goes back on the
  // list.
  SmallVector<MachineBasicBlock *, 32> Work;
  for (MachineBasicBlock &MBB : MF)
    Work.push_back(&MBB);

  while (!Work.empty()) {
    MachineBasicBlock *MBB = Work.pop_back_val();

    unsigned UsedPE = 0, UsedSFU = 0, UsedLDST = 0;
    DenseSet<Register> LateDefs, UsedConsts;
    MachineInstr *LastReal = nullptr;
    bool PureParam = true;
    for (const MachineInstr &PMI : *MBB) {
      if (PMI.isMetaInstruction() || PMI.isDebugInstr() || PMI.isTerminator())
        continue;
      if (!(PMI.mayLoad() && !PMI.mayStore() && isParamLoad(PMI))) {
        PureParam = false;
        break;
      }
    }

    for (MachineInstr &MI : *MBB) {
      if (MI.isTerminator()) {
        // A RET metadata record is a pure terminator to the runtime: the
        // decoder finishes the CTA without dispatching the block, so any
        // real work sharing a p-graph with `ret` silently never executes
        // (m4/m8/m10/m11's final stores). Give a Return with preceding
        // work its own p-graph, the shape the reference toolchain's
        // artifacts always had for their (luckier) CFGs.
        if (MI.isReturn() && LastReal) {
          MachineBasicBlock *Tail =
              MBB->splitAt(*LastReal, /*UpdateLiveIns=*/false);
          Tail->setLabelMustBeEmitted();
          Changed = true;
        }
        break;
      }
      if (MI.isMetaInstruction() || MI.isDebugInstr())
        continue;

      Cost C = classify(MI, TII, PureParam);

      if (C.PE > DicePECount || C.SFU > DiceSFUCount || C.LDST > DiceLDSTCount)
        report_fatal_error(Twine("DICE: instruction exceeds a p-graph budget "
                                 "alone: ") +
                           TII.getName(MI.getOpcode()));

      bool UsesLateDef = false;
      if (!LateDefs.empty())
        for (const MachineOperand &MO : MI.uses())
          if (MO.isReg() && LateDefs.count(MO.getReg())) {
            UsesLateDef = true;
            break;
          }

      // Distinct constant-file values this instruction adds to the region.
      // Only tracked when a boundary-port budget is armed (0 = the full
      // 60xN RF crossbar, no limit).
      SmallVector<Register, 4> NewConsts;
      if (DiceConstReadCount)
        for (const MachineOperand &MO : MI.uses())
          if (MO.isReg() && ParamConstDefs.count(MO.getReg()) &&
              !UsedConsts.count(MO.getReg()) &&
              !llvm::is_contained(NewConsts, MO.getReg()))
            NewConsts.push_back(MO.getReg());

      bool RegionEmpty = (LastReal == nullptr);
      bool NeedSplit =
          !RegionEmpty &&
          (C.IsBarrier || UsesLateDef ||
           UsedPE + C.PE > DicePECount || UsedSFU + C.SFU > DiceSFUCount ||
           UsedLDST + C.LDST > DiceLDSTCount ||
           (DiceConstReadCount &&
            UsedConsts.size() + NewConsts.size() > DiceConstReadCount));

      if (NeedSplit) {
        MachineBasicBlock *Tail = MBB->splitAt(*LastReal,
                                               /*UpdateLiveIns=*/false);
        // A block only reachable by fallthrough gets a "// %bb.N:" comment
        // instead of a label unless the label is forced -- and the label IS
        // the p-graph boundary.
        Tail->setLabelMustBeEmitted();
        Work.push_back(Tail);
        Changed = true;
        break;
      }

      UsedPE += C.PE;
      UsedSFU += C.SFU;
      UsedLDST += C.LDST;
      UsedConsts.insert(NewConsts.begin(), NewConsts.end());
      if (C.DefsArriveLate)
        for (const MachineOperand &MO : MI.defs())
          if (MO.isReg())
            LateDefs.insert(MO.getReg());
      LastReal = &MI;
    }
  }

  // Lay the blocks out in reverse post-order. Clang's placement can put a
  // join block BEFORE the loop that reaches it, so a branch's reconvergence
  // point precedes its target in emission order -- a shape nvcc never
  // produces and the DICE runtime's reconvergence model mishandles
  // (backprop's layerforward barrier loop). RPO keeps joins after their
  // arms and loop bodies contiguous.
  {
    SmallVector<MachineBasicBlock *, 32> Order;
    for (MachineBasicBlock *MBB : ReversePostOrderTraversal<MachineFunction *>(&MF))
      Order.push_back(MBB);
    // What each block currently falls through to, so terminators can be
    // repaired after the move.
    DenseMap<MachineBasicBlock *, MachineBasicBlock *> OldFallthrough;
    for (MachineBasicBlock &MBB : MF) {
      MachineFunction::iterator Next = std::next(MBB.getIterator());
      OldFallthrough[&MBB] =
          Next != MF.end() ? &*Next : nullptr;
    }
    bool Moved = false;
    MachineBasicBlock *Prev = nullptr;
    for (MachineBasicBlock *MBB : Order) {
      if (Prev && &*std::next(Prev->getIterator()) != MBB) {
        MBB->moveAfter(Prev);
        Moved = true;
      }
      Prev = MBB;
    }
    if (Moved) {
      for (MachineBasicBlock *MBB : Order)
        if (!MBB->succ_empty())
          MBB->updateTerminator(OldFallthrough[MBB]);
      Changed = true;
    }
    // A changed layout materializes lost fallthroughs as explicit branches,
    // leaving blocks with TWO terminators (`@%p bra X; bra.uni Y;`). A
    // p-graph has one branch, as its final instruction (check C1) -- give
    // the trailing unconditional branch its own single-instruction p-graph,
    // exactly the shape the reference partitioner produces.
    for (MachineBasicBlock &MBB : llvm::make_early_inc_range(MF)) {
      SmallVector<MachineInstr *, 2> Terms;
      for (MachineInstr &MI : MBB.terminators())
        Terms.push_back(&MI);
      while (Terms.size() > 1) {
        MachineInstr *Last = Terms.pop_back_val();
        MachineBasicBlock *NewBB =
            MF.CreateMachineBasicBlock(MBB.getBasicBlock());
        MF.insert(std::next(MBB.getIterator()), NewBB);
        NewBB->splice(NewBB->end(), &MBB, Last->getIterator(), MBB.end());
        // The moved branch's target belongs to the new block now.
        const MachineBasicBlock *Target = nullptr;
        for (const MachineOperand &MO : Last->operands())
          if (MO.isMBB())
            Target = MO.getMBB();
        if (Target) {
          NewBB->addSuccessor(const_cast<MachineBasicBlock *>(Target));
          if (MBB.isSuccessor(Target))
            MBB.removeSuccessor(const_cast<MachineBasicBlock *>(Target));
        }
        MBB.addSuccessor(NewBB);
        NewBB->setLabelMustBeEmitted();
        Changed = true;
      }
    }

    // Unconditionally: splitAt hands new blocks numbers at the END of the
    // numbering, so without this the emitted $L__BB labels are not in
    // layout order and every consumer that equates label number with
    // p-graph position (the .meta most of all) misindexes.
    MF.RenumberBlocks();
  }

  // Every block boundary is a p-graph boundary, so every block needs its
  // label in the output, including ones reachable only by fallthrough.
  for (MachineBasicBlock &MBB : MF)
    if (&MBB != &MF.front())
      MBB.setLabelMustBeEmitted();

  return Changed;
}

namespace {
bool runPass(MachineFunction &MF) {
  if (!EnableDicePartition)
    return false;
  // Only kernels become p-graphs. A surviving __device__ .func (nw's
  // `maximum`) is host-linkage plumbing the kernels inlined anyway; giving
  // it p-graph structure or .meta records confuses every consumer (check
  // C12: FUNCTION with no matching .entry).
  if (MF.getFunction().getCallingConv() != CallingConv::PTX_Kernel)
    return false;
  return nvptxDiceRepartition(MF);
}
} // namespace

bool NVPTXDicePartition::runOnMachineFunction(MachineFunction &MF) {
  return runPass(MF);
}

MachineFunctionPass *llvm::createNVPTXDicePartitionPass() {
  return new NVPTXDicePartition();
}


//===----------------------------------------------------------------------===//
// DICE optimization pass `fuse` (-nvptx-dice-opt-fuse, default OFF).
//
// Each p-graph pays ~360 cycles of reconfiguration (bitstream fetch), so
// two half-empty adjacent p-graphs cost more than one merged full one.
// After partitioning, greedily merge consecutive blocks A,B when:
//   - A falls through into B (no terminator), and A is B's only pred
//     (a jump target must stay a block head),
//   - the merged budgets fit (PE/SFU/LDST),
//   - B holds no barrier (must lead its p-graph) and no Return (a RET
//     record is dispatch-free; fusing work into it would never execute),
//   - nothing in B consumes a load/atomic issued in A (its result only
//     writes back at A's end -- the load-to-use rule),
//   - neither block is a pure parameter block (dispatch-once purity).
// Fixpoint: a merge can enable the next one.
//===----------------------------------------------------------------------===//

namespace {
class NVPTXDiceFuse : public MachineFunctionPass {
public:
  static char ID;
  NVPTXDiceFuse() : MachineFunctionPass(ID) {}
  StringRef getPassName() const override { return "NVPTX DICE p-graph fusion"; }
  bool runOnMachineFunction(MachineFunction &MF) override;
};
char NVPTXDiceFuse::ID = 0;

struct BlockCost {
  unsigned PE = 0, SFU = 0, LDST = 0;
  DenseSet<Register> Consts;
  bool HasBarrier = false, HasRet = false, PureParam = true, Real = false;
};
} // namespace

static BlockCost blockCost(const MachineBasicBlock &MBB,
                           const TargetInstrInfo &TII,
                           const DenseSet<Register> &ParamConstDefs) {
  BlockCost B;
  for (const MachineInstr &MI : MBB) {
    if (MI.isMetaInstruction() || MI.isDebugInstr())
      continue;
    if (MI.isReturn())
      B.HasRet = true;
    if (MI.isTerminator())
      continue;
    B.Real = true;
    if (!(MI.mayLoad() && !MI.mayStore() && isParamLoad(MI)))
      B.PureParam = false;
    Cost C = classify(MI, TII, /*InPureParamBlock=*/false);
    if (C.IsBarrier)
      B.HasBarrier = true;
    B.PE += C.PE;
    B.SFU += C.SFU;
    B.LDST += C.LDST;
    for (const MachineOperand &MO : MI.uses())
      if (MO.isReg() && ParamConstDefs.count(MO.getReg()))
        B.Consts.insert(MO.getReg());
  }
  if (!B.Real)
    B.PureParam = false;
  return B;
}

bool NVPTXDiceFuse::runOnMachineFunction(MachineFunction &MF) {
  if (!EnableDiceFuse)
    return false;
  if (MF.getFunction().getCallingConv() != CallingConv::PTX_Kernel)
    return false;
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  DenseSet<Register> ParamConstDefs;
  {
    const MachineRegisterInfo &MRI = MF.getRegInfo();
    for (const MachineBasicBlock &MBB : MF)
      for (const MachineInstr &MI : MBB)
        if (MI.mayLoad() && !MI.mayStore() && isParamLoad(MI))
          for (const MachineOperand &MO : MI.defs())
            if (MO.isReg() && MO.getReg().isVirtual() &&
                MRI.hasOneDef(MO.getReg()))
              ParamConstDefs.insert(MO.getReg());
  }
  bool Changed = false, Local = true;
  while (Local) {
    Local = false;
    for (MachineBasicBlock &A : MF) {
      // A reaches its unique candidate B either by falling through to the
      // layout-next block or by an unconditional GOTO (B anywhere).
      MachineBasicBlock *BP = nullptr;
      MachineInstr *Goto = nullptr;
      MachineBasicBlock::iterator T = A.getFirstTerminator();
      if (T == A.end()) {
        MachineFunction::iterator Next = std::next(A.getIterator());
        if (Next == MF.end())
          continue;
        BP = &*Next;
      } else if (T->isUnconditionalBranch() && std::next(T) == A.end()) {
        Goto = &*T;
        BP = T->getOperand(0).getMBB();
      } else {
        continue;
      }
      MachineBasicBlock &B = *BP;
      if (&B == &A || B.pred_size() != 1 || !A.isSuccessor(&B))
        continue;
      BlockCost CA = blockCost(A, TII, ParamConstDefs),
                CB = blockCost(B, TII, ParamConstDefs);
      if (CB.HasBarrier || CB.HasRet || CA.PureParam || CB.PureParam)
        continue;
      if (CA.PE + CB.PE > DicePECount || CA.SFU + CB.SFU > DiceSFUCount ||
          CA.LDST + CB.LDST > DiceLDSTCount)
        continue;
      CA.Consts.insert(CB.Consts.begin(), CB.Consts.end());
      if (DiceConstReadCount && CA.Consts.size() > DiceConstReadCount)
        continue;
      // Load-to-use: nothing in B may read a value a load/atomic in A
      // defines (it lands only at A's exit).
      DenseSet<Register> LateDefs;
      for (const MachineInstr &MI : A)
        if (MI.mayLoad() && !isParamLoad(MI))
          for (const MachineOperand &MO : MI.defs())
            if (MO.isReg() && MO.getReg().isVirtual())
              LateDefs.insert(MO.getReg());
      bool Hazard = false;
      if (!LateDefs.empty())
        for (const MachineInstr &MI : B) {
          for (const MachineOperand &MO : MI.uses())
            if (MO.isReg() && LateDefs.count(MO.getReg())) {
              Hazard = true;
              break;
            }
          if (Hazard)
            break;
        }
      if (Hazard)
        continue;
      // Merge B into A. B's implicit fallthrough must survive the move to
      // A's layout position as an explicit GOTO.
      MachineBasicBlock *BFall = B.getFallThrough();
      if (Goto)
        Goto->eraseFromParent();
      A.removeSuccessor(&B);
      A.splice(A.end(), &B, B.begin(), B.end());
      A.transferSuccessors(&B);
      B.eraseFromParent();
      if (BFall && A.getFallThrough() != BFall && A.isSuccessor(BFall))
        BuildMI(&A, DebugLoc(), TII.get(NVPTX::GOTO)).addMBB(BFall);
      MF.RenumberBlocks();
      Changed = Local = true;
      break;
    }
  }
  return Changed;
}

MachineFunctionPass *llvm::createNVPTXDiceFusePass() {
  return new NVPTXDiceFuse();
}
