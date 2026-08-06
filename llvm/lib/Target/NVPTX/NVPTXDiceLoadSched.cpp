//===- NVPTXDiceLoadSched.cpp - load clustering for DICE partitioning ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DICE optimization pass `loadsched` (-nvptx-dice-opt-loadsched, default
// OFF).
//
// The partitioner must split a p-graph between a load and the first use of
// its result (the value only writes back at p-graph exit). When loads are
// interleaved with their consumers, every load-use chain buys its own
// split. Reordering is free in a dataflow fabric -- program order inside a
// block only constrains register RAW and memory aliasing -- so this pass
// bubbles each load upward past instructions it does not depend on. Loads
// end up clustered right below their address computations, and one
// load-to-use split then serves the whole cluster (up to the 4-port LDST
// budget per p-graph).
//
// A load stops hoisting at the first instruction that (a) defines a
// register it reads -- including an ifcvt guard, which is an implicit use
// -- or (b) may write memory / is a barrier, call, or inline asm (memory
// order is kept conservatively: loads never cross potential stores).
// Volatile accesses and atomics do not move at all. Param loads are left
// alone: the partitioner already hoists them into the dispatch-once
// parameter block.
//
//===----------------------------------------------------------------------===//

#include "NVPTX.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/NVPTXAddrSpace.h"

using namespace llvm;

static cl::opt<bool> EnableDiceLoadSched(
    "nvptx-dice-opt-loadsched", cl::init(false), cl::Hidden,
    cl::desc("DICE: hoist loads to cluster them ahead of their uses"));

namespace {
class NVPTXDiceLoadSched : public MachineFunctionPass {
public:
  static char ID;
  NVPTXDiceLoadSched() : MachineFunctionPass(ID) {}
  StringRef getPassName() const override {
    return "NVPTX DICE load clustering";
  }
  bool runOnMachineFunction(MachineFunction &MF) override;
};
char NVPTXDiceLoadSched::ID = 0;
} // namespace

static bool isBarrierLike(const TargetInstrInfo &TII, const MachineInstr &MI) {
  StringRef N = TII.getName(MI.getOpcode());
  return N.starts_with("BARRIER") || N.starts_with("BAR_");
}

static bool isEntryParamLoad(const MachineInstr &MI) {
  if (MI.memoperands_empty())
    return false;
  for (const MachineMemOperand *MMO : MI.memoperands())
    if (MMO->getAddrSpace() != NVPTXAS::ADDRESS_SPACE_ENTRY_PARAM)
      return false;
  return true;
}

/// A plain load this pass is allowed to move.
static bool isHoistableLoad(const MachineInstr &MI,
                            const TargetInstrInfo &TII) {
  if (!MI.mayLoad() || MI.mayStore() || MI.isCall() || MI.isInlineAsm() ||
      MI.isTerminator() || MI.hasOrderedMemoryRef() ||
      isBarrierLike(TII, MI) || isEntryParamLoad(MI))
    return false;
  for (const MachineMemOperand *MMO : MI.memoperands())
    if (MMO->isVolatile() || MMO->isAtomic())
      return false;
  return true;
}

/// May L move above X? (register and memory dependences)
static bool canHoistAbove(const MachineInstr &L, const MachineInstr &X,
                          const TargetInstrInfo &TII) {
  if (X.isCall() || X.isInlineAsm() || isBarrierLike(TII, X) ||
      X.hasOrderedMemoryRef() || X.mayStore())
    return false;
  for (const MachineOperand &MO : L.operands()) {
    if (!MO.isReg() || !MO.isUse() || !MO.getReg())
      continue;
    if (X.definesRegister(MO.getReg(), /*TRI=*/nullptr))
      return false;
  }
  for (const MachineOperand &MO : L.defs())
    if (MO.isReg() && MO.getReg() &&
        (X.readsRegister(MO.getReg(), nullptr) ||
         X.definesRegister(MO.getReg(), nullptr)))
      return false;
  return true;
}

bool NVPTXDiceLoadSched::runOnMachineFunction(MachineFunction &MF) {
  if (!EnableDiceLoadSched)
    return false;
  if (MF.getFunction().getCallingConv() != CallingConv::PTX_Kernel)
    return false;
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    // Walk top-down; bubble each hoistable load up to its dependence
    // frontier. Loads may cross each other -- safe, since no store is ever
    // crossed, so any aliasing pair still reads the same memory state.
    MachineBasicBlock::iterator I = MBB.begin();
    while (I != MBB.end()) {
      MachineBasicBlock::iterator Next = std::next(I);
      MachineInstr &MI = *I;
      if (!MI.isMetaInstruction() && !MI.isDebugInstr() &&
          isHoistableLoad(MI, TII)) {
        MachineBasicBlock::iterator Ins = I;
        while (Ins != MBB.begin()) {
          MachineBasicBlock::iterator Prev = std::prev(Ins);
          if (Prev->isMetaInstruction() || Prev->isDebugInstr())
            break; // keep prologue labels/meta ahead of hoisted code
          if (!canHoistAbove(MI, *Prev, TII))
            break;
          Ins = Prev;
        }
        if (Ins != I) {
          MBB.splice(Ins, &MBB, I);
          Changed = true;
        }
      }
      I = Next;
    }
  }
  return Changed;
}

MachineFunctionPass *llvm::createNVPTXDiceLoadSchedPass() {
  return new NVPTXDiceLoadSched();
}
