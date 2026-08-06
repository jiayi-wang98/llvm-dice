//===- NVPTXDiceIfConvert.cpp - if-conversion to predication for DICE ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DICE optimization pass `ifcvt` (-nvptx-dice-opt-ifcvt, default OFF).
//
// Every p-graph carries a fixed reconfiguration cost (~360 cycles measured:
// bitstream fetch dominates), so branch-terminated p-graphs and their join
// blocks are the single largest cycle tax on control-heavy kernels. This
// pass converts small triangles and diamonds into @%p-predicated straight
// line code BEFORE partitioning, removing the branch (one p-graph) and
// letting the side block(s) merge into their predecessor.
//
//   triangle:  A: @p bra C            diamond:  A: @p bra C
//              B: work; (fall C)                B: work; bra D
//              C: join                          C: work; (fall/bra D)
//                                               D: join
//
// The branch is taken when (negFlag ? !p : p); a side block executes on the
// path that does NOT reach it via the branch, which fixes each block's
// guard polarity. Post-PHI-elimination both arms of a former phi write the
// SAME vreg (copies materialized per arm), so predicating each arm with
// complementary guards leaves the register holding whichever arm executed
// -- exactly the original merge semantics.
//
// Mechanics: the guard is recorded in NVPTXMachineFunctionInfo and the
// predicate register is attached to each converted instruction as an
// IMPLICIT USE, so the partitioner's load-to-use scan and the register
// allocator's liveness see it with no special cases. The asm printer emits
// a trailing `// DICE_PRED @[!]%pN` marker that dicc rewrites into the
// PTX prefix.
//
// Conservative v1 limits: side blocks must be straight-line, single-pred /
// single-succ, contain no barrier / call / inline-asm / atomic / already-
// predicated instruction / definition of the guard predicate, and hold at
// most `-nvptx-dice-ifcvt-max-ops` real instructions (predicated ops still
// occupy PEs, so oversized conversions would just split again downstream).
//
//===----------------------------------------------------------------------===//

#include "NVPTX.h"
#include "NVPTXMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/NVPTXAddrSpace.h"

using namespace llvm;

static cl::opt<bool> EnableDiceIfCvt(
    "nvptx-dice-opt-ifcvt", cl::init(false), cl::Hidden,
    cl::desc("DICE: if-convert small triangles/diamonds into predication"));

static cl::opt<unsigned> DiceIfCvtMaxOps(
    "nvptx-dice-ifcvt-max-ops", cl::init(12), cl::Hidden,
    cl::desc("DICE ifcvt: max real instructions per converted side block"));

namespace {
class NVPTXDiceIfConvert : public MachineFunctionPass {
public:
  static char ID;
  NVPTXDiceIfConvert() : MachineFunctionPass(ID) {}
  StringRef getPassName() const override {
    return "NVPTX DICE if-conversion";
  }
  bool runOnMachineFunction(MachineFunction &MF) override;
};
char NVPTXDiceIfConvert::ID = 0;
} // namespace

static bool isBarrierName(const TargetInstrInfo &TII, const MachineInstr &MI) {
  StringRef N = TII.getName(MI.getOpcode());
  return N.starts_with("BARRIER") || N.starts_with("BAR_");
}

/// A block predicable in v1: straight-line real work only.
static bool isPredicableBlock(const MachineBasicBlock &MBB, Register Pred,
                              const TargetInstrInfo &TII,
                              const NVPTXMachineFunctionInfo &MFI,
                              unsigned MaxOps, bool AllowTerminator) {
  unsigned Real = 0;
  for (const MachineInstr &MI : MBB) {
    if (MI.isMetaInstruction() || MI.isDebugInstr())
      continue;
    if (MI.isTerminator()) {
      if (!AllowTerminator || !MI.isUnconditionalBranch())
        return false;
      continue; // the diamond's `bra D` is deleted, not predicated
    }
    if (MI.isCall() || MI.isInlineAsm() || isBarrierName(TII, MI))
      return false;
    if (MI.mayLoad() && MI.mayStore())
      return false; // atomics: predicated RMW untested in the runtime
    if (MFI.getDicePredGuard(&MI))
      return false; // no guard composition in v1
    if (MI.definesRegister(Pred, /*TRI=*/nullptr))
      return false;
    ++Real;
  }
  return Real > 0 && Real <= MaxOps;
}

static bool isParamOnlyLoad(const MachineInstr &MI) {
  if (!MI.mayLoad() || MI.mayStore() || MI.memoperands_empty())
    return false;
  for (const MachineMemOperand *MMO : MI.memoperands())
    if (MMO->getAddrSpace() != NVPTXAS::ADDRESS_SPACE_ENTRY_PARAM)
      return false;
  return true;
}

static void predicateBlock(MachineBasicBlock &MBB, Register Pred, bool Neg,
                           NVPTXMachineFunctionInfo &MFI) {
  for (MachineInstr &MI : MBB) {
    if (MI.isMetaInstruction() || MI.isDebugInstr() || MI.isTerminator())
      continue;
    // Param loads stay UNguarded: reading immutable parameter memory is
    // side-effect free, an unconditional entry-block write is overwritten
    // by any later arm's write (program order), and — the practical
    // reason — a guarded param load cannot be hoisted into the free
    // IS_PARAMETER_LOAD block, so its destination would burn LDST budget
    // mid-block (C6 on bfs).
    if (isParamOnlyLoad(MI))
      continue;
    MFI.setDicePredGuard(&MI, Pred, Neg);
    MachineInstrBuilder(*MBB.getParent(), &MI)
        .addReg(Pred, RegState::Implicit);
  }
}

bool NVPTXDiceIfConvert::runOnMachineFunction(MachineFunction &MF) {
  if (!EnableDiceIfCvt)
    return false;
  if (MF.getFunction().getCallingConv() != CallingConv::PTX_Kernel)
    return false;

  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  auto *MFI = MF.getInfo<NVPTXMachineFunctionInfo>();
  bool Changed = false;

  // Iterate to a fixpoint: converting an inner triangle can expose an outer
  // one. Blocks are only ever merged, so this terminates.
  bool LocalChange = true;
  while (LocalChange) {
    LocalChange = false;
    for (MachineBasicBlock &A : MF) {
      // Match: A ends with exactly one conditional branch and falls through.
      MachineBasicBlock::iterator T = A.getFirstTerminator();
      if (T == A.end() || T->getOpcode() != NVPTX::CBranch)
        continue;
      if (std::next(T) != A.end())
        continue; // CBranch + GOTO pair: layout-fixed branches, skip in v1
      Register Pred = T->getOperand(0).getReg();
      bool NegFlag = T->getOperand(2).getImm() != 0;
      MachineBasicBlock *Target = T->getOperand(1).getMBB();
      MachineFunction::iterator AIt = A.getIterator();
      if (std::next(AIt) == MF.end())
        continue;
      MachineBasicBlock *B = &*std::next(AIt); // fallthrough arm
      if (B == Target)
        continue;

      // The fallthrough arm executes when the branch is NOT taken.
      const bool BNeg = !NegFlag;

      // ---- triangle: B falls through (or brs) to Target, Target is join.
      bool BSingle = B->pred_size() == 1 && B->succ_size() == 1 &&
                     *B->succ_begin() == Target;
      if (BSingle &&
          isPredicableBlock(*B, Pred, TII, *MFI, DiceIfCvtMaxOps,
                            /*AllowTerminator=*/true)) {
        bool BFallsThrough = std::next(B->getIterator()) != MF.end() &&
                             &*std::next(B->getIterator()) == Target;
        // Diamond check first: B branches to D, and Target (single-pred
        // from A besides...) also reaches D. v1 keeps it simple: only the
        // triangle (join == Target) here; the diamond case below.
        predicateBlock(*B, Pred, BNeg, *MFI);
        // Remove A's branch: A now falls into B, B into Target.
        T->eraseFromParent();
        A.removeSuccessor(Target);
        if (!BFallsThrough) {
          // B ended with `bra Target`; keep it only if Target is not the
          // next block after B in layout.
          MachineBasicBlock::iterator BT = B->getFirstTerminator();
          if (BT != B->end() && std::next(B->getIterator()) != MF.end() &&
              &*std::next(B->getIterator()) == Target)
            BT->eraseFromParent();
        }
        // Merge B into A when possible (single pred, layout-adjacent).
        A.removeSuccessor(B); // the fallthrough edge; B is about to die
        A.splice(A.end(), B, B->begin(), B->end());
        A.transferSuccessors(B);
        B->eraseFromParent();
        Changed = LocalChange = true;
        break; // CFG changed; restart scan
      }

      // ---- diamond: B ends `bra D`; Target single-pred, ends in D too.
      if (B->pred_size() == 1 && B->succ_size() == 1 &&
          Target->pred_size() == 1 && Target->succ_size() == 1 &&
          *B->succ_begin() == *Target->succ_begin() &&
          *B->succ_begin() != Target && *B->succ_begin() != B &&
          isPredicableBlock(*B, Pred, TII, *MFI, DiceIfCvtMaxOps, true) &&
          isPredicableBlock(*Target, Pred, TII, *MFI, DiceIfCvtMaxOps,
                            true)) {
        MachineBasicBlock *D = *B->succ_begin();
        predicateBlock(*B, Pred, BNeg, *MFI);
        predicateBlock(*Target, Pred, NegFlag, *MFI);
        // Drop all terminators in A, B, Target; chain A->B->Target->D.
        T->eraseFromParent();
        for (MachineBasicBlock *X : {B, Target})
          while (X->getFirstTerminator() != X->end())
            X->getFirstTerminator()->eraseFromParent();
        A.removeSuccessor(Target);
        A.removeSuccessor(B);
        A.splice(A.end(), B, B->begin(), B->end());
        A.splice(A.end(), Target, Target->begin(), Target->end());
        A.transferSuccessors(B); // A -> (B's succ = D)
        Target->removeSuccessor(D);
        B->eraseFromParent();
        Target->eraseFromParent();
        // Re-establish fallthrough or explicit branch to D.
        if (std::next(A.getIterator()) == MF.end() ||
            &*std::next(A.getIterator()) != D)
          BuildMI(&A, DebugLoc(), TII.get(NVPTX::GOTO)).addMBB(D);
        Changed = LocalChange = true;
        break;
      }
    }
  }
  return Changed;
}

MachineFunctionPass *llvm::createNVPTXDiceIfConvertPass() {
  return new NVPTXDiceIfConvert();
}
