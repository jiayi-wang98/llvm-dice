//===- NVPTXDiceFP64Lower.cpp - lower fp64 FMA to mul + add ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// HARDWARE-OWNER DECISION, 2026-08-06: DICE's fp64 unit is TWO-OPERAND by
// construction, and `fma.rn.f64` is therefore lowered here to a multiply
// followed by an add.
//
// WHY THE HARDWARE CANNOT DO IT
// -----------------------------
// A PE tile has four 32-bit diagonal inputs. Two 64-bit operands are exactly
// four 32-bit words, so a two-input fp64 operation fits the tile exactly and a
// three-input one has nowhere for its third operand to arrive. That is an
// operand-DELIVERY limit, not an fpnew option: `fpnew_pkg::get_opgroup` maps
// FMADD, ADD and MUL onto one 3-operand ADDMUL block and there is no knob for
// "ADDMUL without FMA", so the fused datapath is physically present, would
// accept the op word, and would multiply-add a GARBAGE third operand with
// nothing downstream saying so. See docs/device-contract.md 5.5e.
//
// WHY IT IS LOWERED IN THE COMPILER AND NOT IN THE MAPPER
// -------------------------------------------------------
// Two reasons, both structural:
//
//  1. RESOURCE ACCOUNTING. `fma.rn.f64` costs one SFU tile; `mul` + `add`
//     costs two, and the fabric has four. NVPTXDicePartition closes a p-graph
//     when adding an instruction would exceed the SFU budget, so the split has
//     to be visible to it. Lowering later would let the partitioner size a
//     p-graph for four fp64 ops that then needs eight tiles, and the failure
//     would surface as an unmappable p-graph with no obvious cause.
//
//  2. THE REFERENCE MODELS. `dicemap.golden` and `rtl/tb/dice_host.py` both
//     evaluate the `.pptx`. Lowering here means they model the LOWERED form
//     for free -- so a pass means "the RTL computes what we asked the hardware
//     to compute", not "the RTL is within 1 ULP of something else we compared
//     it against". Both models REFUSE `fma.*.f64` by name and a test asserts
//     the absence, which is what makes that guarantee real rather than
//     intended.
//
// THE SEMANTIC COST, WHICH IS REAL AND IS NOT HIDDEN
// ---------------------------------------------------
// A fused multiply-add rounds ONCE; `mul` then `add` rounds TWICE. The results
// differ by up to about 1 ULP from PTX's `.rn` semantics, and for a
// catastrophic cancellation (`a*b + c` where `a*b ~= -c`) they can differ by
// very much more, because the fused form keeps the exact product's low bits
// and the split form has already thrown them away.
//
// So every function this fires in records a QUALIFICATION. The asm printer
// emits it as a `// DICE_QUAL` comment (a plain PTX comment: nothing in the
// simulator's parser or in dicemeta.y has to change), `dice-verify` reports it
// as a warning, and `dice-rtlsweep` carries it into the per-kernel verdict.
// An artifact that computes double-rounded where the program asked for fused
// must SAY SO; a 1-ULP gap discovered later at the answer-checking stage looks
// exactly like a fabric bug, and this project has already spent a round on one
// of those.
//
// The pass is a no-op unless -nvptx-dice-partition is on, so the non-DICE
// NVPTX pipeline is byte-for-byte unchanged.
//
//===----------------------------------------------------------------------===//

#include "NVPTX.h"
#include "NVPTXMachineFunctionInfo.h"
#include "NVPTXRegisterInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "nvptx-dice-fp64-lower"

namespace llvm {
bool nvptxDiceEnabled();
}

namespace {

class NVPTXDiceFP64Lower : public MachineFunctionPass {
public:
  static char ID;
  NVPTXDiceFP64Lower() : MachineFunctionPass(ID) {}
  StringRef getPassName() const override {
    return "NVPTX DICE fp64 FMA lowering";
  }
  bool runOnMachineFunction(MachineFunction &MF) override;
};

char NVPTXDiceFP64Lower::ID = 0;

/// The five `FMA<F64RT>` forms TableGen generates, and what each half becomes.
/// `r` is a register operand and `i` an fp64 immediate, in (a, b, c) order:
/// the product is a*b and the addend is c.
struct FmaForm {
  unsigned Fma;      // the FMA opcode to match
  unsigned Mul;      // opcode for a*b
  unsigned Add;      // opcode for t+c
};

} // namespace

bool NVPTXDiceFP64Lower::runOnMachineFunction(MachineFunction &MF) {
  if (!nvptxDiceEnabled())
    return false;

  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  // NOTE the operand-kind suffixes: FMA_F64rri has an IMMEDIATE ADDEND, so its
  // add is the `ri` form and its multiply the plain `rr` one -- and FMA_F64rir
  // is the mirror image. Getting the pair the wrong way round builds an
  // instruction whose operand kinds do not match its opcode, which asserts in
  // a +Asserts build and mis-prints in a Release one.
  const FmaForm Forms[] = {
      {NVPTX::FMA_F64rrr, NVPTX::FMUL_rnf64rr, NVPTX::FADD_rnf64rr},
      {NVPTX::FMA_F64rri, NVPTX::FMUL_rnf64rr, NVPTX::FADD_rnf64ri},
      {NVPTX::FMA_F64rir, NVPTX::FMUL_rnf64ri, NVPTX::FADD_rnf64rr},
      {NVPTX::FMA_F64rii, NVPTX::FMUL_rnf64ri, NVPTX::FADD_rnf64ri},
      // The NVVM intrinsic form of __fma_rn(double, double, double). Three
      // register operands, so it lowers exactly like `rrr`.
      {NVPTX::INT_NVVM_FMA_rn_f64, NVPTX::FMUL_rnf64rr, NVPTX::FADD_rnf64rr},
  };

  unsigned Lowered = 0;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
      const FmaForm *Form = nullptr;
      for (const FmaForm &F : Forms)
        if (MI.getOpcode() == F.Fma) {
          Form = &F;
          break;
        }
      if (!Form) {
        // `iir` is a*b with BOTH factors immediate -- a constant expression
        // the folder should already have evaluated. It has no register to
        // start a multiply from, so rather than invent a materialization that
        // has never been exercised, refuse by name. If a kernel ever produces
        // one, this message says exactly what to add.
        if (MI.getOpcode() == NVPTX::FMA_F64iir)
          report_fatal_error(
              "NVPTXDiceFP64Lower: fma.rn.f64 with two immediate factors "
              "(FMA_F64iir) has no register to seed the multiply with. DICE's "
              "fp64 unit is two-operand, so this must be lowered, and the "
              "constant-folded form has never been seen in the corpus. Add a "
              "materialization here rather than letting a 3-operand fp64 op "
              "reach the fabric.");
        continue;
      }

      // dst = fma(a, b, c)  ->  t = a * b ; dst = t + c
      const MachineOperand &Dst = MI.getOperand(0);
      const MachineOperand &A = MI.getOperand(1);
      const MachineOperand &B = MI.getOperand(2);
      const MachineOperand &C = MI.getOperand(3);
      Register Tmp = MRI.createVirtualRegister(&NVPTX::B64RegClass);

      MachineInstrBuilder Mul =
          BuildMI(MBB, MI, MI.getDebugLoc(), TII.get(Form->Mul), Tmp);
      Mul.add(A).add(B);
      MachineInstrBuilder Add =
          BuildMI(MBB, MI, MI.getDebugLoc(), TII.get(Form->Add),
                  Dst.getReg());
      Add.addReg(Tmp).add(C);

      // The predication guard, if if-conversion put one on the FMA, has to
      // move to BOTH halves: the multiply writes a fresh vreg (harmless when
      // the guard is false) but the add writes the FMA's destination, and an
      // unguarded add would commit a value the program said not to.
      auto *NMFI = MF.getInfo<NVPTXMachineFunctionInfo>();
      if (NMFI)
        if (const auto *G = NMFI->getDicePredGuard(&MI)) {
          NMFI->setDicePredGuard(Mul.getInstr(), G->first, G->second);
          NMFI->setDicePredGuard(Add.getInstr(), G->first, G->second);
          Mul.addReg(G->first, RegState::Implicit);
          Add.addReg(G->first, RegState::Implicit);
        }

      MI.eraseFromParent();
      ++Lowered;
    }
  }

  if (Lowered) {
    auto *NMFI = MF.getInfo<NVPTXMachineFunctionInfo>();
    if (NMFI)
      NMFI->addDiceQualification(
          ("fma.rn.f64 x " + Twine(Lowered) +
           " lowered to mul.rn.f64 + add.rn.f64: DICE's fp64 unit is "
           "two-operand (device-contract.md 5.5e), so this function rounds "
           "TWICE where PTX asked for ONE fused rounding. Results may differ "
           "from `.rn` FMA semantics by ~1 ULP, and by more where a*b "
           "cancels against c.")
              .str());
    errs() << "warning: " << MF.getName() << ": lowered " << Lowered
           << " fma.rn.f64 to mul.rn.f64 + add.rn.f64 (DICE's fp64 unit is "
              "two-operand); this rounds twice where PTX asked for once\n";
  }
  return Lowered != 0;
}

MachineFunctionPass *llvm::createNVPTXDiceFP64LowerPass() {
  return new NVPTXDiceFP64Lower();
}
