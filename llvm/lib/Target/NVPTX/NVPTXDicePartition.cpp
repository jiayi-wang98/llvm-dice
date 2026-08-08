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
#include "llvm/IR/Constants.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
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

static cl::opt<bool> DiceCostDump(
    "nvptx-dice-cost-dump", cl::init(false), cl::Hidden,
    cl::desc("Print each p-graph's tile accounting to stderr. The three cost "
             "models (this pass, dicemap.pgraph, dicevfy C4) must agree, and "
             "this is how a disagreement is localised to an instruction"));

static cl::opt<unsigned> DiceImmBits(
    "nvptx-dice-imm-bits", cl::init(16), cl::Hidden,
    cl::desc("Width of the SIGNED immediate field a PE tile's configuration "
             "holds (device fabric.pe.immediate_bits; 0 = none). A literal "
             "that fits folds into the tile for free; one that does not has to "
             "be MATERIALISED on tiles of its own, and that is a PE cost"));

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

  // ---- TILES THE MAPPER SYNTHESISES, which are PE cost this pass owes -----
  //
  // AN INSTRUCTION COUNT IS NOT A TILE COUNT, and until 2026-08-07 this pass
  // assumed it was. The fabric has no way to produce a literal or a memory
  // displacement other than a tile, so `dicemap.pgraph` synthesises one --
  // and the partitioner, charging one PE per arithmetic instruction and
  // nothing else, emitted p-graphs the mapper then refused as
  // `resource_oversubscribed`. Measured over all 538 p-graphs of the bundled
  // corpus on 2026-08-07:
  //
  //   | uncharged ALU tile                                   | occurrences |
  //   |------------------------------------------------------|-------------|
  //   | ld/st with a nonzero bracket displacement            |         202 |
  //   | mov of a .shared SYMBOL (the address is a literal)   |         145 |
  //   | a selp literal operand (the phi mux has no imm path) |          28 |
  //   | a literal too wide for the tile immediate            |          15 |
  //   | a SECOND literal on one node (one imm field/tile)    |           6 |
  //
  // FOUR p-graphs of the corpus exceed the 16 ALU tiles, and every one of them
  // does so BECAUSE OF THESE and for no other reason: hotspot and hotspot_fp32
  // `DICE_BB0_7` (17 int) and `DICE_BB0_8` (19 int), and kmeans `DICE_BB1_3`
  // (21 int). That is the whole of hotspot_fp32's `resource_oversubscribed`
  // and the whole of kmeans fn1's.
  //
  // MATERIALISED CONSTANTS ARE SHARED WITHIN A P-GRAPH, keyed by the 32-bit
  // pattern (`dicemap.pgraph`'s `const_nodes` dict), so the cost cannot be a
  // per-instruction addend: two `selp`s selecting 15 pay for one tile between
  // them. These vectors are therefore CANDIDATES that the region-level scan
  // deduplicates, not a charge.
  SmallVector<uint32_t, 4> MatConsts;
  SmallVector<const GlobalValue *, 2> MatSymbols;
  // rule 1: THE FABRIC IS THE ONLY ADDER. `pgraph_meta_t` carries no
  // displacement field and the LDST port takes the address straight off
  // `xbar_mem_addr`, so `[%r1+4]` is lowered to an in-fabric `add.s32`. This
  // one is NOT deduplicated -- each access needs its own adder.
  unsigned DispAdds = 0;
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

/// Does `Value` fit a signed `Bits`-wide tile immediate field?
/// The mirror of `dicevfy.ir.imm_fits` / `dicemap.fabric.ImmediatePolicy.fits`.
static bool immFits(int64_t Value, unsigned Bits) {
  if (Bits == 0 || Bits > 63)
    return Bits > 63;
  return Value >= -(int64_t(1) << (Bits - 1)) &&
         Value <= (int64_t(1) << (Bits - 1)) - 1;
}

/// ALU tiles spent MATERIALISING the 32-bit pattern `K`, or 0 if it cannot be.
///
/// The mirror of `dicemap.pgraph`'s `const_node` and of
/// `dicevfy.ir.materialisation_tiles`: one tile when the SIGNED value fits the
/// field (`and.b32 K, K` -- both operand muxes select the tile's own immediate
/// and `K AND K == K`); otherwise the halves, `and`+`shl` for a nonzero high
/// half, `and` for a nonzero low half plus `shl`+`shr` when its bit 15 is set
/// (the field is SIGN-extended by `imm_const_32b`, so a low half of 0x999A
/// arrives with every top bit on), plus an `or` to join two halves. Six tiles
/// at the worst, which is what a float literal like 0f3E99999A costs -- so one
/// `mul.f32 %r, %r, 0f3E99999A` is SEVEN tiles, not one.
static unsigned matTiles(uint32_t K, unsigned Bits) {
  int64_t Signed = int64_t(int32_t(K));
  if (immFits(Signed, Bits))
    return 1;
  if (Bits < 16)
    return 0; // the mapper refuses such a device by name rather than splitting
  uint32_t Hi = (K >> 16) & 0xFFFF, Lo = K & 0xFFFF;
  unsigned Tiles = 0, Parts = 0;
  if (Hi) {
    Tiles += 2; // and.b32 hi, hi + shl.b32 16
    ++Parts;
  }
  if (Lo) {
    Tiles += 1; // and.b32 lo, lo
    if (Lo & 0x8000)
      Tiles += 2; // shl.b32 16 + shr.u32 16 to zero-extend
    ++Parts;
  }
  if (Parts == 2)
    Tiles += 1; // or.b32
  return Tiles;
}

/// The 32-bit pattern a literal MachineOperand carries, if it is one.
///
/// A float literal is charged BY ITS BIT PATTERN because that is what the tile
/// has to produce: `dicemap.pgraph`'s `_any_literal` reads `0f3E99999A` as the
/// integer 0x3E99999A and materialises exactly that.
static std::optional<uint32_t> literalPattern(const MachineOperand &MO) {
  if (MO.isImm())
    return uint32_t(uint64_t(MO.getImm()) & 0xFFFFFFFFu);
  if (MO.isCImm())
    return uint32_t(MO.getCImm()->getValue().getZExtValue() & 0xFFFFFFFFu);
  if (MO.isFPImm()) {
    APFloat F = MO.getFPImm()->getValueAPF();
    // fp64 IS DELIBERATELY NOT CHARGED, and it mirrors a mapper LIMITATION
    // rather than a saving: `dicemap.pgraph.resolve` gates a literal on
    // `is_immediate`, whose regex accepts int and `0f` but not `0d`, so an fp64
    // literal operand is DROPPED -- no node and no tile. Those p-graphs are the
    // ones dice-fasm already refuses for the fp64 register pair
    // (docs/device-contract.md 5.5e item 5), so charging here would make this
    // pass disagree with the mapper on a path that cannot run either way.
    if (&F.getSemantics() != &APFloat::IEEEsingle())
      return std::nullopt;
    return F.bitcastToAPInt().getZExtValue() & 0xFFFFFFFFu;
  }
  return std::nullopt;
}

/// `selp` EXECUTES ON THE PHI MUX, whose data_true/data_false inputs have no
/// immediate path at all (unlike the ALU/cmp/bitwise operand muxes), so EVERY
/// literal operand of a selp has to be materialised -- including BOTH of
/// `selp.b32 %r8, 1, 0, %p0`. `dicemap.pgraph` says so in as many words when it
/// passes `allow_fold = policy.enabled and inst.base_opcode != "selp"`.
static bool isSelpName(StringRef Name) { return Name.starts_with("SELP_"); }

/// Tiles a SUB-WORD `cvt` occupies: 2 when it sign-extends, 1 when it masks,
/// 0 when it is not one.
///
/// There is no `cvt` in the fabric's op table at all, so `dicemap.pgraph`
/// lowers it into shifts: `cvt.s32.s8 d, s` becomes `shl.b32 t, s, 24 ;
/// shr.s32 d, t, 24` -- TWO tiles for one instruction -- and `cvt.u32.u8`
/// becomes a single `and.b32 d, s, 0xFF`. `isFreeIntWidthCvt` above already
/// excludes these (8 and 16 are not in its {32, 64}), so without this they were
/// charged one PE like any other op. m5_bytes's `DICE_BB0_4` is the case.
///
/// Names look like `CVT_s32_s8`; the SOURCE type decides sign extension, so
/// `cvt.s32.u8` masks and `cvt.u32.s8` sign-extends.
static unsigned subwordCvtTiles(StringRef Name) {
  if (!Name.consume_front("CVT_"))
    return 0;
  auto [Dst, Src] = Name.split('_');
  if (Dst != "s32" && Dst != "u32")
    return 0;
  Src = Src.split('_').first;
  if (Src.size() < 2 || (Src[0] != 's' && Src[0] != 'u'))
    return 0;
  unsigned Width = 0;
  if (Src.substr(1).getAsInteger(10, Width) || (Width != 8 && Width != 16))
    return 0;
  return Src[0] == 's' ? 2 : 1;
}

/// Tiles a memory access spends on its bracket DISPLACEMENT (0 or 1).
///
/// THE FABRIC IS THE ONLY ADDER. mini_dice's LDST port takes the address
/// straight off `xbar_mem_addr` (`dice_cgra_rf`: `cgra_mem_addr_lo[p] =
/// fab_mem_addr_o[p]`, no offset) and `pgraph_meta_t` carries no displacement
/// field, so `dicemap.pgraph` lowers `[%r1+4]` to a real in-fabric `add.s32`
/// with the 4 in the tile immediate. 202 accesses in the corpus have one, and
/// the partitioner charged them an LDST port and no PE at all.
///
/// NVPTX addresses a load/store as (base, offset) with the OFFSET LAST -- the
/// `ADDRri` complex pattern, printed as `LD_i32 0, 0, 1, 3, 32, -1, %62:b64, 0`
/// where the trailing 0 is the offset. Reading the last use operand is how the
/// asm printer itself finds it, so it does not depend on the opcode name.
/// A memory access's literal ADDRESS operands, which are tiles like any other.
///
/// `st.shared.u32 [_ZZ..temp], %r6` addresses a `.shared` symbol directly: the
/// address is a literal, so the mapper materialises it (needle's `DICE_BB1_3`
/// and `DICE_BB2_20` are one tile each and nothing else). A literal DATA
/// operand that fits the field folds into the memory tile's own configuration
/// on a device whose memory units ARE array tiles, which every device in
/// `device/` with LDST tiles is; one that does not fit materialises.
static void collectMemLiterals(const MachineInstr &MI, Cost &C) {
  for (const MachineOperand &MO : MI.uses()) {
    if (MO.isGlobal()) {
      C.MatSymbols.push_back(MO.getGlobal());
      continue;
    }
    if (std::optional<uint32_t> K = literalPattern(MO))
      if (!immFits(int64_t(int32_t(*K)), DiceImmBits))
        C.MatConsts.push_back(*K);
  }
}

static unsigned memDisplacementAdds(const MachineInstr &MI) {
  const MachineOperand *Last = nullptr;
  for (const MachineOperand &MO : MI.uses())
    Last = &MO;
  if (!Last)
    return 0;
  if (Last->isImm())
    return Last->getImm() != 0 ? 1 : 0;
  // A global-address offset (`[sym+8]`) is an address the mapper materialises
  // outright rather than an add on a routed base, so it is charged as a
  // MatSymbol by the caller and costs no adder here.
  return 0;
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
      C.DispAdds = memDisplacementAdds(MI);
      collectMemLiterals(MI, C);
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
    C.DispAdds = memDisplacementAdds(MI);
    collectMemLiterals(MI, C);
    return C;
  }

  if (isSFUName(Name)) {
    C.SFU = 1;
    return C;
  }

  // Free: address-space casts, integer width changes, movs (the fabric
  // routes them), special-register reads (mov.u32 %r, %tid.x).
  //
  // `cvta` IS MATCHED CASE-INSENSITIVELY, and that is a BUG FIX, not a
  // tidy-up. NVPTX names the instruction `cvta_to_global_64` -- LOWERCASE --
  // so `starts_with("CVTA")` never matched and EVERY `cvta` has been charged a
  // PE since this pass was written, contradicting both this file's own header
  // comment ("cvta and 32<->64 integer cvt = free") and
  // dicevfy.ir.is_non_pe, which has always returned True for it. Measured
  // 2026-08-07 with -nvptx-dice-cost-dump: needle's `needle_cuda_shared_2`
  // block 11 opens with two `cvta_to_global_64` charged 1 PE each, which took
  // a p-graph the mapper costs at 16 tiles to 17 and split it. Every other
  // free-list name really is upper-case (`CVT_u64_u32`, `MOV_B64_sym`,
  // `INT_PTX_SREG_CTAID_x`), so this is the only one that was wrong -- and it
  // was wrong in the SAFE direction, which is why it survived: over-charging
  // only ever produces a smaller p-graph, never an illegal one.
  if (Name.starts_with_insensitive("cvta") || isFreeIntWidthCvt(Name) ||
      Name.starts_with("MOV") || Name.starts_with("IMOV") ||
      Name.starts_with("INT_PTX_SREG")) {
    // A `mov` is free ROUTING, unless what it names is NOT A REGISTER.
    //
    //  * `MOV_B64_sym @_ZZ..temp_on_cuda` asks for the ADDRESS of a `.shared`
    //    array. An address is a literal, and there is no crossbar source for a
    //    literal, so the mapper materialises it on a tile. 145 of these in the
    //    corpus, and two of them are why hotspot's `DICE_BB0_8` needs 19 tiles.
    //  * a literal too WIDE for the tile immediate cannot wait for a consumer
    //    to fold it, so it materialises here and now. `mov.b32 %w0, 0f41800000`
    //    is two tiles.
    //
    // A `mov` of a literal that FITS stays free: the consumer folds it into its
    // own immediate field, and if some consumer cannot, the consumer pays.
    for (const MachineOperand &MO : MI.uses()) {
      if (MO.isGlobal()) {
        C.MatSymbols.push_back(MO.getGlobal());
      } else if (std::optional<uint32_t> K = literalPattern(MO)) {
        if (!immFits(int64_t(int32_t(*K)), DiceImmBits))
          C.MatConsts.push_back(*K);
      }
    }
    return C;
  }

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

  if (unsigned CvtTiles = subwordCvtTiles(Name)) {
    // The lowering resolves only its SOURCE operand and folds both shift
    // amounts into the tiles' own immediates, so no literal is materialised.
    C.PE = CvtTiles;
    return C;
  }

  // ONE IMMEDIATE FIELD PER TILE. `dicemap.pgraph.split_operands` folds the
  // FIRST literal operand that fits and MATERIALISES every other literal on
  // the same node, "because a tile's configuration holds a single immediate
  // field". `selp` folds none at all (`isSelpName`, above).
  bool Folded = false;
  bool AllowFold = DiceImmBits > 0 && !isSelpName(Name);
  for (const MachineOperand &MO : MI.uses()) {
    if (MO.isGlobal()) {
      C.MatSymbols.push_back(MO.getGlobal());
      continue;
    }
    std::optional<uint32_t> K = literalPattern(MO);
    if (!K)
      continue;
    int64_t Signed = int64_t(int32_t(*K));
    if (AllowFold && !Folded && immFits(Signed, DiceImmBits)) {
      Folded = true;
      continue;
    }
    C.MatConsts.push_back(*K);
  }
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
    // MATERIALISED CONSTANTS ARE SHARED WITHIN A P-GRAPH, keyed by the 32-bit
    // pattern for a literal and by the symbol for an address -- so this is a
    // region-level SET, exactly as `dicemap.pgraph`'s `const_nodes` dict is,
    // and two `selp`s selecting 15 pay for one tile between them.
    DenseSet<uint32_t> MatValues;
    DenseSet<const GlobalValue *> MatSyms;
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

      // What this instruction ADDS to the region's materialisation set. A value
      // already materialised here is free; the tiles for a new one are
      // `matTiles`, which is 1 for a literal that fits the field and up to SIX
      // for one that does not.
      SmallVector<uint32_t, 4> NewMat;
      SmallVector<const GlobalValue *, 2> NewMatSyms;
      unsigned MatPE = 0;
      for (uint32_t K : C.MatConsts)
        if (!MatValues.count(K) && !llvm::is_contained(NewMat, K)) {
          NewMat.push_back(K);
          MatPE += matTiles(K, DiceImmBits);
        }
      for (const GlobalValue *GV : C.MatSymbols)
        if (!MatSyms.count(GV) && !llvm::is_contained(NewMatSyms, GV)) {
          NewMatSyms.push_back(GV);
          // A `.shared` SYMBOL ADDRESS is charged ONE tile, and that is exact
          // for every symbol in the bundled corpus (measured: all 145 cost 1)
          // but not a guarantee. This pass cannot know the address: the layout
          // is `dicemap.pgraph.shared_symbol_layout` over the module's `.shared`
          // sizes offset by `dice-map --smem-base`, which is a MAPPER input and
          // not a compiler one. A shared block that pushed a symbol above the
          // tile immediate would cost up to six tiles and this would undercharge
          // by five. dice-verify's C4 computes the exact number from the artifact
          // and the device file, so the case is CAUGHT rather than silent -- it
          // is named here so it is not mistaken for exactness.
          MatPE += 1;
        }
      unsigned InstPE = C.PE + C.DispAdds + MatPE;

      if (InstPE > DicePECount || C.SFU > DiceSFUCount ||
          C.LDST > DiceLDSTCount)
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
           UsedPE + InstPE > DicePECount || UsedSFU + C.SFU > DiceSFUCount ||
           UsedLDST + C.LDST > DiceLDSTCount ||
           (DiceConstReadCount &&
            UsedConsts.size() + NewConsts.size() > DiceConstReadCount));

      if (NeedSplit && DiceCostDump)
        errs() << "DICE-COST " << MF.getName() << " " << MBB->getNumber()
               << " SPLIT before " << TII.getName(MI.getOpcode())
               << " (would be PE=" << (UsedPE + InstPE) << " of "
               << DicePECount << ", SFU=" << (UsedSFU + C.SFU) << ", LDST="
               << (UsedLDST + C.LDST) << ", lateDef=" << UsesLateDef
               << ", barrier=" << C.IsBarrier << ")\n";
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

      if (DiceCostDump)
        errs() << "DICE-COST " << MF.getName() << " " << MBB->getNumber()
               << " +PE=" << InstPE << " (op=" << C.PE << " disp=" << C.DispAdds
               << " mat=" << MatPE << ") +SFU=" << C.SFU
               << " +LDST=" << C.LDST << " running(PE=" << (UsedPE + InstPE)
               << ") " << TII.getName(MI.getOpcode()) << "\n";
      UsedPE += InstPE;
      UsedSFU += C.SFU;
      UsedLDST += C.LDST;
      UsedConsts.insert(NewConsts.begin(), NewConsts.end());
      MatValues.insert(NewMat.begin(), NewMat.end());
      MatSyms.insert(NewMatSyms.begin(), NewMatSyms.end());
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
  // The SAME region-level materialisation sets the splitter keeps. Fusing two
  // blocks whose PE counts sum to 16 is illegal if either spends tiles on a
  // literal, so the fuser has to account for them or it would undo the splits
  // the partitioner just made.
  DenseSet<uint32_t> MatValues;
  DenseSet<const GlobalValue *> MatSyms;
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
    B.PE += C.PE + C.DispAdds;
    for (uint32_t K : C.MatConsts)
      if (B.MatValues.insert(K).second)
        B.PE += matTiles(K, DiceImmBits);
    for (const GlobalValue *GV : C.MatSymbols)
      if (B.MatSyms.insert(GV).second)
        B.PE += 1;
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
      // `CA.PE + CB.PE` OVER-counts a literal the two blocks share, because
      // each already paid for it inside its own set. That is the SAFE direction
      // for a fuse decision -- it refuses a legal fusion rather than allowing an
      // illegal one -- and `-nvptx-dice-opt-fuse` is off by default anyway.
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
