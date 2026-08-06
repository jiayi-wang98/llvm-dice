//===- NVPTXDiceRegAlloc.cpp - DICE architectural register assignment ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Assigns DICE architectural registers to NVPTX virtual registers. Runs
// after NVPTXDicePartition, when every basic block is a legal p-graph, so
// block boundaries ARE p-graph boundaries:
//
//   %p  predicate registers (i1),          pool per device contract
//   %c  parameter-load destinations,       one per value, no pool
//   %w  wires: values whose whole life is  per-block numbering; a wire
//       inside one p-graph                 never crosses a boundary (C7)
//   %r  everything else,                   colored on block-granularity
//                                          live intervals, pool of 32
//
// The assignment is *state*, not printing: it is stored in
// NVPTXMachineFunctionInfo for the asm printer to name registers with, and
// for later passes (unrolling factor selection reads real indices, since
// the register file banks as (tid + reg) % 32).
//
// PHI elimination has already merged control-flow joins into single vregs,
// so "all producers reaching a common use share a register" — the invariant
// the reference compiler had to reconstruct — holds here by construction.
//
//===----------------------------------------------------------------------===//

#include "NVPTX.h"
#include "NVPTXMachineFunctionInfo.h"
#include "NVPTXRegisterInfo.h"
#include "NVPTXSubtarget.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Support/NVPTXAddrSpace.h"
#include "llvm/Support/raw_ostream.h"
#include <set>

using namespace llvm;

static cl::opt<bool> EnableDiceRegAlloc(
    "nvptx-dice-regalloc", cl::init(false), cl::Hidden,
    cl::desc("Assign DICE architectural registers (%r/%c/%p/%w)"));

static cl::opt<unsigned> DiceGPRCount(
    "nvptx-dice-gpr-count", cl::init(32), cl::Hidden,
    cl::desc("DICE general register pool (%r)"));

static cl::opt<unsigned> DicePredCount(
    "nvptx-dice-pred-count", cl::init(24), cl::Hidden,
    cl::desc("DICE predicate register pool (%p)"));

namespace {

class NVPTXDiceRegAlloc : public MachineFunctionPass {
public:
  static char ID;
  NVPTXDiceRegAlloc() : MachineFunctionPass(ID) {}
  StringRef getPassName() const override {
    return "NVPTX DICE register assignment";
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachinePostDominatorTreeWrapperPass>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
  bool runOnMachineFunction(MachineFunction &MF) override;
};

char NVPTXDiceRegAlloc::ID = 0;

struct Interval {
  Register Reg;
  unsigned Start, End;       // block indices, inclusive
  unsigned StartPos, EndPos; // instruction index within Start/End block
  // True when the interval opens with a definition (its first event is a
  // def, not a loop-carried use).
  bool StartIsDef = true;
  // Every block touching the value, with whether the FIRST touch there is
  // a read. Used for backedge extension: the value lives across a loop's
  // iterations iff its EARLIEST touch inside the loop is a read -- then it
  // arrives at the loop's top from outside or from the previous trip. (A
  // read-first block alone is not enough: a reload's consumer block reads
  // first, but the reload one block earlier feeds it within the same
  // iteration -- treating that as loop-carried re-extended every spill
  // artifact around the loop and refilled the pool.)
  SmallVector<std::pair<unsigned, bool>, 4> TouchBlocks;
};

} // namespace

static bool isParamLoadMI(const MachineInstr &MI) {
  if (!MI.mayLoad() || MI.mayStore() || MI.memoperands_empty())
    return false;
  for (const MachineMemOperand *MMO : MI.memoperands())
    if (MMO->getAddrSpace() != NVPTXAS::ADDRESS_SPACE_ENTRY_PARAM)
      return false;
  return true;
}

/// Color intervals over a pool. Same-index reuse is what makes 32 GPRs
/// enough; an interval that finds the pool empty is a hard error until
/// spilling exists (the allocator is where spilling belongs, because a
/// reload is an LDST access that forces a p-graph re-split).
static void colorIntervals(SmallVectorImpl<Interval> &Ivs, unsigned PoolSize,
                           char Cls, MachineFunction &MF,
                           DenseMap<Register, unsigned> &Out,
                           SmallVectorImpl<Register> *Failed = nullptr,
                           const DenseSet<Register> *NoEvict = nullptr) {
  llvm::stable_sort(Ivs, [](const Interval &A, const Interval &B) {
    return std::tie(A.Start, A.StartPos) < std::tie(B.Start, B.StartPos);
  });
  // Two values may share a register when the later one's FIRST DEF comes
  // strictly after the earlier one's LAST USE in program order. That is
  // sound for both execution models at once: the functional simulator runs
  // a p-graph's instructions in program order (so the old value's reads
  // have already happened), and the hardware reads data at p-graph entry
  // and commits writes at exit (old reads see the old value regardless).
  // A branch is always the last instruction, so a predicate consumed by a
  // branch can never legally share its block with a later def -- no
  // special case needed. Loop live-through extension marks the whole tail
  // block busy (EndPos = ~0u).
  SmallVector<std::optional<std::pair<unsigned, unsigned>>, 32> FreeAt(
      PoolSize); // (block, pos) the color frees AFTER; nullopt = never used
  SmallVector<Register, 32> Holder(PoolSize); // who holds each color
  for (const Interval &IV : Ivs) {
    int Chosen = -1;
    for (unsigned C = 0; C < PoolSize; ++C) {
      bool Free = !FreeAt[C];
      if (!Free) {
        auto [FB, FP] = *FreeAt[C];
        Free = IV.Start > FB ||
               (IV.Start == FB && IV.StartIsDef && IV.StartPos > FP &&
                FP != ~0u);
      }
      if (Free) {
        Chosen = C;
        break;
      }
    }
    if (Chosen < 0) {
      if (Failed) {
        // Furthest-end wins the spill: evicting the value that stays live
        // longest frees the most future pressure. Spilling whoever happened
        // to arrive when the pool was full spins without converging --
        // short intervals' reloads live about as long as the original.
        // Never evict a spill artifact: re-spilling a reload just stores
        // the same value back and reloads it again -- churn that burned
        // through every retry round without reducing pressure.
        int Worst = -1;
        for (unsigned C = 0; C < PoolSize; ++C) {
          if (NoEvict && NoEvict->count(Holder[C]))
            continue;
          if (Worst < 0 || *FreeAt[C] > *FreeAt[Worst])
            Worst = C;
        }
        if (Worst >= 0 &&
            *FreeAt[Worst] > std::make_pair(IV.End, IV.EndPos) &&
            !(NoEvict && NoEvict->count(IV.Reg) &&
              false /* incoming artifact may still displace */)) {
          Failed->push_back(Holder[Worst]);
          Out.erase(Holder[Worst]);
          Chosen = Worst;
        } else if (!NoEvict || !NoEvict->count(IV.Reg)) {
          Failed->push_back(IV.Reg);
          continue;
        } else if (Worst >= 0) {
          // The incoming interval is itself a spill artifact; it must have
          // a register. Displace the longest-lived non-artifact even if it
          // ends sooner than we do.
          Failed->push_back(Holder[Worst]);
          Out.erase(Holder[Worst]);
          Chosen = Worst;
        } else {
          std::string Dump;
          raw_string_ostream DOS(Dump);
          DOS << "DICE regalloc: %" << Cls
              << " pool holds only spill artifacts in " << MF.getName()
              << "\nfailing: reg=" << IV.Reg.virtRegIndex() << " ["
              << IV.Start << ":" << IV.StartPos << " .. " << IV.End << ":"
              << (int)IV.EndPos << "] startIsDef=" << IV.StartIsDef << "\n";
          for (unsigned C = 0; C < PoolSize; ++C)
            DOS << "  color " << C << ": reg="
                << Holder[C].virtRegIndex() << " freesAt=("
                << FreeAt[C]->first << "," << (int)FreeAt[C]->second << ")\n";
          report_fatal_error(Twine(Dump));
        }
      } else {
        report_fatal_error(Twine("DICE regalloc: %") + Twine(Cls) +
                           " pool exhausted in " + MF.getName());
      }
    }
    Holder[Chosen] = IV.Reg;
    FreeAt[Chosen] = std::make_pair(IV.End, IV.EndPos);
    Out[IV.Reg] = Chosen;
  }
}

/// Insert spill code for \p Regs: a .local store after every definition and
/// a per-block reload feeding the uses. The reload is a real load, so the
/// caller re-runs partitioning afterwards -- the load-to-use rule moves the
/// consumers into a later p-graph, which is exactly the "a reload restructures
/// the block" cost the handoff warned about. Addressing is manual
/// (%SPL + offset) because NVPTXPrologEpilogPass has already run; the depot
/// declaration is emitted at print time from the stack size set here.
static void insertSpills(MachineFunction &MF, MachineRegisterInfo &MRI,
                         ArrayRef<Register> Regs, unsigned &NextOffset,
                         DenseSet<Register> &Artifacts) {
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  const NVPTXRegisterInfo *NRI =
      MF.getSubtarget<NVPTXSubtarget>().getRegisterInfo();
  const Register SPL = NRI->getFrameLocalRegister(MF);
  const bool Local64 = SPL == NVPTX::VRFrameLocal64;

  auto IsBarrier = [&](const MachineInstr &MI) {
    StringRef N = TII.getName(MI.getOpcode());
    return N.starts_with("BARRIER") || N.starts_with("BAR_");
  };

  // One depot pointer per function. It must sit in an earlier p-graph than
  // any reload that uses it (program order suffices for the functional
  // model), and it may not lead the IS_PARAMETER_LOAD block or displace a
  // barrier from its block's head.
  bool HaveDepot = false;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB)
      if (MI.definesRegister(SPL, /*TRI=*/nullptr)) {
        HaveDepot = true;
        break;
      }
    if (HaveDepot)
      break;
  }
  if (!HaveDepot) {
    for (MachineBasicBlock &MBB : MF) {
      MachineBasicBlock::iterator InsPt = MBB.end();
      bool AllParam = true;
      for (auto It = MBB.begin(); It != MBB.end(); ++It) {
        if (It->isMetaInstruction() || It->isDebugInstr() || IsBarrier(*It))
          continue;
        if (It->mayLoad() && !It->mayStore() && isParamLoadMI(*It))
          continue;
        AllParam = false;
        InsPt = It;
        break;
      }
      if (AllParam)
        continue; // pure parameter block: keep it pure
      BuildMI(MBB, InsPt, DebugLoc(),
              TII.get(Local64 ? NVPTX::MOV_DEPOT_ADDR_64
                              : NVPTX::MOV_DEPOT_ADDR),
              SPL)
          .addImm(MF.getFunctionNumber());
      break;
    }
  }

  for (Register V : Regs) {
    const TargetRegisterClass *RC = MRI.getRegClass(V);
    const bool Is64 = RC == &NVPTX::B64RegClass;
    const bool Is16 = RC == &NVPTX::B16RegClass;
    const unsigned Bytes = Is64 ? 8 : 4;
    NextOffset = alignTo(NextOffset, Bytes);
    const unsigned Off = NextOffset;
    NextOffset += Bytes;
    const unsigned StOpc =
        Is64 ? NVPTX::ST_i64 : (Is16 ? NVPTX::ST_i16 : NVPTX::ST_i32);
    const unsigned LdOpc =
        Is64 ? NVPTX::LD_i64 : (Is16 ? NVPTX::LD_i16 : NVPTX::LD_i32);
    const unsigned Width = Is64 ? 64 : (Is16 ? 16 : 32);

    // Split the live range completely: every definition moves to a fresh
    // register that dies into its spill store, and every use reads either
    // the block's latest fresh definition (in-block flow) or a reload
    // inserted immediately before it. The original register vanishes. A
    // residual multi-def register was the 32-GPR failure: its single
    // interval still spanned the loop even though its value lived in
    // memory, and twenty of those crowded out the pool as unevictable
    // artifacts.
    for (MachineBasicBlock &MBB : MF) {
      Register CurDef; // latest in-block definition's register, if any
      for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
        // An IMPLICIT_DEF emits no PTX: storing "its value" writes a
        // never-defined register (C7 on hotspot at 16 GPRs). The value is
        // undef -- whatever the untouched spill slot holds serves equally.
        if (MI.isMetaInstruction() || MI.isDebugInstr())
          continue;
        if (MI.readsRegister(V, /*TRI=*/nullptr)) {
          Register Repl = CurDef;
          if (!Repl) {
            Repl = MRI.createVirtualRegister(RC);
            Artifacts.insert(Repl);
            BuildMI(MBB, MachineBasicBlock::iterator(&MI), DebugLoc(),
                    TII.get(LdOpc), Repl)
                .addImm(NVPTX::Ordering::NotAtomic)
                .addImm(NVPTX::Scope::Thread)
                .addImm(NVPTX::AddressSpace::Local)
                .addImm(NVPTX::PTXLdStInstCode::Untyped)
                .addImm(Width)
                .addImm(UINT32_MAX)
                .addReg(SPL)
                .addImm(Off);
          }
          for (MachineOperand &MO : MI.operands())
            if (MO.isReg() && MO.isUse() && MO.getReg() == V)
              MO.setReg(Repl);
        }
        if (MI.definesRegister(V, /*TRI=*/nullptr)) {
          Register ND = MRI.createVirtualRegister(RC);
          Artifacts.insert(ND);
          for (MachineOperand &MO : MI.operands())
            if (MO.isReg() && MO.isDef() && MO.getReg() == V)
              MO.setReg(ND);
          BuildMI(MBB, std::next(MachineBasicBlock::iterator(&MI)),
                  DebugLoc(), TII.get(StOpc))
              .addReg(ND)
              .addImm(NVPTX::Ordering::NotAtomic)
              .addImm(NVPTX::Scope::Thread)
              .addImm(NVPTX::AddressSpace::Local)
              .addImm(Width)
              .addReg(SPL)
              .addImm(Off);
          CurDef = ND;
        }
      }
    }
  }
  MF.getFrameInfo().setStackSize(NextOffset);
  MF.getFrameInfo().ensureMaxAlignment(Align(8));
}

bool NVPTXDiceRegAlloc::runOnMachineFunction(MachineFunction &MF) {
  if (!EnableDiceRegAlloc)
    return false;

  // Only kernels become p-graphs. A surviving __device__ .func (nw's
  // `maximum`) is host-linkage plumbing the kernels inlined anyway; giving
  // it p-graph structure or .meta records confuses every consumer (check
  // C12: FUNCTION with no matching .entry).
  if (MF.getFunction().getCallingConv() != CallingConv::PTX_Kernel)
    return false;

  MachineRegisterInfo &MRI = MF.getRegInfo();
  auto *MFI = MF.getInfo<NVPTXMachineFunctionInfo>();

  // Spill-and-retry: when the pool does not suffice, spill the intervals
  // that failed to color, re-partition (the reloads force load-to-use
  // splits and can overflow LDST budgets), and start over. Each round
  // strictly shrinks the failed values' live ranges, so this terminates.
  unsigned SpillOffset = 0;
  DenseSet<Register> SpillArtifacts;
  const unsigned MaxRounds = 24;
  for (unsigned Round = 0;; ++Round) {
  MFI->clearDiceRegNames();

  DenseMap<const MachineBasicBlock *, unsigned> BlockIdx;
  unsigned N = 0;
  for (const MachineBasicBlock &MBB : MF)
    BlockIdx[&MBB] = N++;

  DenseMap<const MachineInstr *, unsigned> InstPos;
  for (const MachineBasicBlock &MBB : MF) {
    unsigned P = 0;
    for (const MachineInstr &MI : MBB)
      InstPos[&MI] = P++;
  }

  SmallVector<Interval, 64> GPRIvs, PredIvs;
  unsigned NextC = 0;
  // Per-block wire numbering: wires are p-graph-local, so the same index
  // is a different wire in a different block.
  DenseMap<const MachineBasicBlock *, unsigned> WireNext;
  unsigned MaxWire = 0, MaxPred = 0;

  for (unsigned I : llvm::seq(MRI.getNumVirtRegs())) {
    Register VR = Register::index2VirtReg(I);
    if (MRI.use_empty(VR) && MRI.def_empty(VR))
      continue;

    unsigned Lo = ~0u, Hi = 0;
    // (block, pos) of the first and last event, and what kind each is.
    std::pair<unsigned, unsigned> First{~0u, ~0u}, Last{0, 0};
    bool FirstIsPureDef = false, LastHasDef = false;
    const MachineBasicBlock *OnlyBB = nullptr;
    bool OneBlock = true, AllDefsParam = true, AnyDef = false;
    SmallDenseMap<unsigned, std::pair<unsigned, bool>, 8>
        FirstTouch; // block -> (pos, is-read)
    for (const MachineInstr &MI : MRI.reg_instructions(VR)) {
      unsigned Idx = BlockIdx.lookup(MI.getParent());
      unsigned Pos = InstPos.lookup(&MI);
      {
        auto It = FirstTouch.find(Idx);
        bool Reads = MI.readsRegister(VR, /*TRI=*/nullptr);
        if (It == FirstTouch.end() || Pos < It->second.first)
          FirstTouch[Idx] = {Pos, Reads};
      }
      Lo = std::min(Lo, Idx);
      Hi = std::max(Hi, Idx);
      bool Defs = MI.definesRegister(VR, /*TRI=*/nullptr);
      bool Uses = MI.readsRegister(VR, /*TRI=*/nullptr);
      std::pair<unsigned, unsigned> Ev{Idx, Pos};
      if (Ev < First) {
        First = Ev;
        FirstIsPureDef = Defs && !Uses;
      }
      if (Ev >= Last) {
        Last = Ev;
        // A def as the last event commits at the block's EXIT, so the
        // register stays busy through the whole block.
        LastHasDef = Defs;
      }
      if (!OnlyBB)
        OnlyBB = MI.getParent();
      else if (OnlyBB != MI.getParent())
        OneBlock = false;
    }
    for (const MachineInstr &Def : MRI.def_instructions(VR)) {
      AnyDef = true;
      if (!isParamLoadMI(Def))
        AllDefsParam = false;
    }
    bool StartIsDef = AnyDef && FirstIsPureDef;
    unsigned StartPos = First.second;
    unsigned EndPos = LastHasDef ? ~0u : Last.second;
    SmallVector<std::pair<unsigned, bool>, 4> Touches;
    for (auto &[Blk, PosRead] : FirstTouch)
      Touches.push_back({Blk, PosRead.second});

    const TargetRegisterClass *RC = MRI.getRegClass(VR);
    if (RC == &NVPTX::B1RegClass) {
      PredIvs.push_back({VR, Lo, Hi, StartPos, EndPos, StartIsDef, Touches});
      continue;
    }
    if (AnyDef && AllDefsParam) {
      MFI->setDiceRegName(VR, ("%c" + Twine(NextC++)).str());
      continue;
    }
    if (OneBlock && AnyDef) {
      // Wire: born and consumed inside one p-graph. Also requires that no
      // load defines it — a load's value arrives only at the p-graph's end,
      // so a load destination must be register-backed (check C9). The
      // partition pass guarantees no in-block consumer exists, so a
      // load-defined vreg is never one-block anyway unless dead-ish; be
      // safe and give loads a %r.
      bool DefIsLoad = false;
      for (const MachineInstr &Def : MRI.def_instructions(VR))
        if (Def.mayLoad())
          DefIsLoad = true;
      if (!DefIsLoad) {
        unsigned &W = WireNext[OnlyBB];
        MFI->setDiceRegName(VR, ("%w" + Twine(W)).str());
        MaxWire = std::max(MaxWire, ++W);
        continue;
      }
    }
    GPRIvs.push_back({VR, Lo, Hi, StartPos, EndPos, StartIsDef, Touches});
  }

  // Loop live-through: a value whose layout interval overlaps a backedge's
  // span [header, tail] is live around the whole iteration, not just up to
  // its last textual use -- the next trip reads it again. Without this, a
  // body-local value shares the color and clobbers it every iteration
  // (pathfinder's result array came back all zeros). Iterate for nested
  // loops.
  {
    SmallVector<std::pair<unsigned, unsigned>, 8> Backedges; // (header, tail)
    for (const MachineBasicBlock &MBB : MF)
      for (const MachineBasicBlock *S : MBB.successors())
        if (BlockIdx.lookup(S) <= BlockIdx.lookup(&MBB))
          Backedges.push_back({BlockIdx.lookup(S), BlockIdx.lookup(&MBB)});
    auto Extend = [&](SmallVectorImpl<Interval> &Ivs) {
      bool Changed = true;
      while (Changed) {
        Changed = false;
        for (Interval &IV : Ivs)
          for (auto [H, T] : Backedges) {
            // Live across the backedge iff some block in [H, T] READS the
            // value before any in-block def: the value arrives at that
            // block's entry from a previous iteration (or from before the
            // loop) and must survive the whole span. Position heuristics
            // (defined-before-header) over-extended: a spilled value's
            // residual defs straddle the loop but every block defs first,
            // so it no longer lives across iterations at all.
            bool AcrossIterations = false;
            unsigned EarliestInLoop = ~0u;
            for (auto [B, ReadFirst] : IV.TouchBlocks)
              if (B >= H && B <= T && B < EarliestInLoop) {
                EarliestInLoop = B;
                AcrossIterations = ReadFirst;
              }
            if (AcrossIterations && IV.Start <= T && IV.End >= H &&
                (IV.End < T || (IV.End == T && IV.EndPos != ~0u))) {
              IV.End = T;
              IV.EndPos = ~0u;
              Changed = true;
            }
          }
      }
    };
    Extend(GPRIvs);
    Extend(PredIvs);
  }

  DenseMap<Register, unsigned> Color;
  SmallVector<Register, 8> Failed;
  colorIntervals(GPRIvs, DiceGPRCount, 'r', MF, Color,
                 Round + 1 < MaxRounds ? &Failed : nullptr, &SpillArtifacts);
  if (!Failed.empty()) {
    insertSpills(MF, MRI, Failed, SpillOffset, SpillArtifacts);
    nvptxDiceRepartition(MF);
    continue;
  }
  for (auto &[Reg, C] : Color)
    MFI->setDiceRegName(Reg, ("%r" + Twine(C)).str());
  unsigned MaxGPR = 0;
  for (auto &[Reg, C] : Color)
    MaxGPR = std::max(MaxGPR, C + 1);

  Color.clear();
  colorIntervals(PredIvs, DicePredCount, 'p', MF, Color);
  for (auto &[Reg, C] : Color) {
    MFI->setDiceRegName(Reg, ("%p" + Twine(C)).str());
    MaxPred = std::max(MaxPred, C + 1);
  }

  MFI->setDiceRegCounts(MaxGPR, NextC, MaxPred, MaxWire);
  break;
  } // spill-retry loop

  // ---- .meta records ------------------------------------------------
  // Rendered here because this pass owns everything the records need: the
  // register names just assigned, the block structure (every block is a
  // p-graph), and the post-dominator tree for RECVPC. The asm printer
  // appends the text as DICE_META comments; dicc splits it into the .meta.
  auto &PDT =
      getAnalysis<MachinePostDominatorTreeWrapperPass>().getPostDomTree();
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();

  auto Name = [&](Register R) -> std::string {
    if (const std::string *S = MFI->getDiceRegName(R))
      return *S;
    return "%r?"; // unassigned: only possible for dead defs
  };
  auto RegList = [](std::set<std::string> &S) {
    std::string Out = "(";
    bool First = true;
    for (const std::string &R : S) {
      if (!First)
        Out += ", ";
      Out += R;
      First = false;
    }
    return Out + ")";
  };
  // Latency model, mirroring config/base_u1.json latency_model.
  auto InstLat = [&](const MachineInstr &MI) -> unsigned {
    StringRef N = TII.getName(MI.getOpcode());
    if (N.contains("DIV") || N.contains("SQRT") || N.contains("EX2") ||
        N.contains("LG2") || N.contains("SIN") || N.contains("COS") ||
        N.contains("RCP"))
      return 8;
    if (N.contains("MUL") || N.contains("MAD") || N.contains("FMA"))
      return 2;
    if (N.starts_with("F") || N.contains("_f32"))
      return 2;
    return 1;
  };

  std::string Meta;
  raw_string_ostream OS(Meta);
  OS << "FUNCTION = " << MF.getName() << ";\n";

  unsigned FnNum = MF.getFunctionNumber();
  // After the partition pass's RenumberBlocks, block number == layout
  // position == the number in the emitted label; key on it directly.
  DenseMap<const MachineBasicBlock *, unsigned> DBB;
  for (const MachineBasicBlock &MBB : MF)
    DBB[&MBB] = MBB.getNumber();

  for (const MachineBasicBlock &MBB : MF) {
    std::set<std::string> InRegs, OutRegs, LdDest, SRegs;
    DenseSet<Register> DefinedHere;
    unsigned NInsts = 0, NStores = 0;
    bool Barrier = false, Ret = false, AllParam = true, AnyReal = false;
    const MachineInstr *Branch = nullptr;
    DenseMap<const MachineInstr *, unsigned> PathLat;
    DenseMap<Register, unsigned> DefLat;
    unsigned BlockLat = 1;

    for (const MachineInstr &MI : MBB) {
      if (MI.isMetaInstruction() || MI.isDebugInstr())
        continue;
      ++NInsts;
      StringRef N = TII.getName(MI.getOpcode());
      bool IsParam = isParamLoadMI(MI);
      bool IsLoadish = (MI.mayLoad() && !IsParam); // includes atomics
      if (!IsParam)
        AllParam = false;
      AnyReal = true;
      if (N.starts_with("BARRIER") || N.starts_with("BAR_"))
        Barrier = true;
      if (N.consume_front("INT_PTX_SREG_")) {
        // INT_PTX_SREG_CTAID_x reads %ctaid.x, and the runtime needs the
        // read declared (check C16).
        auto [Base, Dim] = N.rsplit('_');
        SRegs.insert(("%" + Base.lower() + "." + Dim.lower()));
      }
      if (MI.isReturn())
        Ret = true;
      if (MI.isBranch()) {
        Branch = &MI;
        continue;
      }
      if (MI.mayStore() && !MI.mayLoad())
        ++NStores;

      unsigned Depth = 0;
      for (const MachineOperand &MO : MI.uses()) {
        if (!MO.isReg() || !MO.getReg().isVirtual())
          continue;
        Register R = MO.getReg();
        if (auto It = DefLat.find(R); It != DefLat.end())
          Depth = std::max(Depth, It->second);
        std::string Nm = Name(R);
        if (Nm[1] != 'w' && !DefinedHere.count(R))
          InRegs.insert(Nm);
      }
      unsigned Lat = Depth + InstLat(MI);
      BlockLat = std::max(BlockLat, Lat);
      for (const MachineOperand &MO : MI.defs()) {
        if (!MO.isReg() || !MO.getReg().isVirtual())
          continue;
        Register R = MO.getReg();
        DefinedHere.insert(R);
        DefLat[R] = Lat;
        std::string Nm = Name(R);
        if (Nm[1] == 'w')
          continue;
        if (IsLoadish || IsParam)
          LdDest.insert(Nm);
        else
          OutRegs.insert(Nm);
      }
    }

    unsigned Id = DBB.lookup(&MBB);
    OS << "\n";
    if (!SRegs.empty())
      OS << "// SR_IN = " << RegList(SRegs) << "\n";
    OS << "DBB_ID = " << Id << ",\n";
    OS << "BITSTREAM_ADDR = $DICE_BB" << FnNum << "_" << Id << ",\n";
    if (Ret && !AnyReal) {
      OS << "RET;\n";
      continue;
    }
    // A ret block matches the reference's short form when it is only a ret.
    if (Ret && NInsts == 1) {
      OS << "RET;\n";
      continue;
    }
    OS << "BITSTREAM_LENGTH = " << NInsts << ",\n";
    OS << "UNROLLING_FACTOR = 1,\n";
    OS << "UNROLLING_STRATEGY = 0,\n";
    OS << "LAT = " << BlockLat << ",\n";
    SmallVector<std::string, 8> Fields;
    if (!InRegs.empty())
      Fields.push_back("IN_REGS = " + RegList(InRegs));
    if (!OutRegs.empty())
      Fields.push_back("OUT_REGS = " + RegList(OutRegs));
    if (!LdDest.empty())
      Fields.push_back("LD_DEST_REGS = " + RegList(LdDest));
    if (NStores)
      Fields.push_back("STORE = " + std::to_string(NStores));
    if (Branch) {
      Fields.push_back("BRANCH = 1");
      if (Barrier)
        Fields.push_back("BARRIER");
      const MachineBasicBlock *Target = nullptr;
      for (const MachineOperand &MO : Branch->operands())
        if (MO.isMBB())
          Target = MO.getMBB();
      unsigned TargetId = Target ? DBB.lookup(Target) : Id + 1;
      bool Uniform = Branch->getOpcode() == NVPTX::GOTO;
      unsigned Recv = TargetId;
      if (Uniform) {
        Fields.push_back("BRANCH_UNI = 1");
      } else {
        // operands: (pred, target, negate-flag)
        Register P = Branch->getOperand(0).getReg();
        bool Neg = Branch->getOperand(2).getImm() != 0;
        Fields.push_back(std::string("BRANCH_PRED = (") + (Neg ? "!" : "") +
                         Name(P) + ")");
        if (MachineBasicBlock *IPDom =
                PDT.getNode(const_cast<MachineBasicBlock *>(&MBB))
                    ? PDT.getNode(const_cast<MachineBasicBlock *>(&MBB))
                              ->getIDom()
                          ? PDT.getNode(const_cast<MachineBasicBlock *>(&MBB))
                                ->getIDom()
                                ->getBlock()
                          : nullptr
                    : nullptr)
          Recv = DBB.lookup(IPDom);
        else if (TargetId < Id)
          Recv = Id + 1;
        if (Recv >= DBB.size())
          Recv = TargetId;
      }
      Fields.push_back("BRANCH_TARGET = " + std::to_string(TargetId));
      Fields.push_back("BRANCH_RECVPC = " + std::to_string(Recv));
    } else if (Barrier) {
      Fields.push_back("BARRIER");
    }
    if (Ret)
      Fields.push_back("RET");
    if (AllParam && AnyReal)
      Fields.push_back("IS_PARAMETER_LOAD");
    if (Fields.empty()) {
      // Punctuation: the record must end in ';', so the last always-on
      // field carries it.
      Meta.pop_back(); // drop '\n' after LAT line
      Meta.pop_back(); // drop ','
      Meta += ";\n";
    } else {
      for (unsigned I = 0; I < Fields.size(); ++I)
        OS << Fields[I] << (I + 1 == Fields.size() ? ";\n" : ",\n");
    }
  }
  MFI->setDiceMetaText(Meta);

  return false; // no IR change; the assignment is side state
}

MachineFunctionPass *llvm::createNVPTXDiceRegAllocPass() {
  return new NVPTXDiceRegAlloc();
}
