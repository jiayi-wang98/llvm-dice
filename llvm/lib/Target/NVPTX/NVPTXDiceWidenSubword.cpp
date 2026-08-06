//===- NVPTXDiceWidenSubword.cpp - widen sub-word loads for DICE ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A sub-word memory operation must never reach the DICE fabric, because the
// fabric has no way to express one.
//
// WHAT THE HARDWARE ACTUALLY DOES. A DICE memory port is 32 bits wide and there
// is no access-size field anywhere: not in `pgraph_meta_t`, not on the
// CGRA<->LDST bus, and no byte enable in the LDST path. A load returns the FOUR
// BYTES STARTING AT ITS BYTE ADDRESS -- unaligned, little-endian. That is not an
// inference; it is what the retire path computes:
//
//     DE_pkg.sv, assemble_ldst_wr():
//       data = core_rsp_data[outcmd_address_map[i]*8 +: DICE_REG_DATA_WIDTH]
//
// where `address_map[i]` is thread i's byte offset within the 32-byte line. So
// `ld.global.u8 [p]` already loads *(i32*)p and writes the whole word to the
// destination register. Measured on the m5_bytes microbenchmark: `o[0]` came back
// 0x03020164 where 0x00000064 was wanted, i.e. `c[0]+100` with c[1..3] still
// sitting in the upper three bytes.
//
// WHAT THIS PASS DOES. Exactly what the hardware does, but explicitly, so the
// mask exists in the program:
//
//     %v = load i8,  ptr %p        ==>   %w = load i32, ptr %p, align 4
//                                       %m = and i32 %w, 255
//                                       %v = trunc i32 %m to i8
//
// and the pass rewrites the EXTENSIONS of that value itself, rather than leaving
// them to InstCombine:
//
//     zext  i8 %v to i32          ==>   and  i32 %w, 255
//     sext  i8 %v to i32          ==>   ashr (shl i32 %w, 24), 24
//
// No address arithmetic is needed, precisely BECAUSE the port is unaligned: the
// widened load reads from the same pointer.
//
// WHY THE EXTENSIONS ARE REWRITTEN HERE and not left to InstCombine: InstCombine
// is happy to keep `sext i8`, which ISel emits as **`cvt.s32.s8`** -- and `cvt`
// is not in the fabric's op table at all (tools/dice-fasm has no entry for it),
// so dice-fasm refuses the p-graph by name and the kernel is blocked instead of
// wrong. `shl`/`ashr` map onto the PE's `lsl`/`asr` and `and` onto `bitwise`, so
// this form is executable. This was measured on m5_bytes: relying on InstCombine
// produced `ld.global.u32` + `cvt.s32.s8`, which is progress and still unbuildable.
//
// The mask lands after the load, so the partitioner puts it in the NEXT p-graph
// on its own (a load is `DefsArriveLate`, and a use of a late def forces a
// split). That is also why this cannot be done in the mapper: the mapper works
// one p-graph at a time and a load's destination is not readable inside the
// p-graph that issued it, so a mapper-side rewrite would have to synthesise
// nodes in a *different* p-graph than the load -- which it has no mechanism for.
//
// WHAT THIS PASS DELIBERATELY DOES NOT DO.
//
//  * STORES. `st.global.u8` cannot be widened: a 32-bit store would clobber the
//    three neighbouring bytes, and a read-modify-write is not atomic against the
//    other threads of the CTA. There is no byte enable to do it properly. So a
//    sub-word store is left alone and `dice-verify` refuses it by name (check
//    C17) rather than emitting an image that corrupts neighbouring data.
//  * volatile / atomic accesses, which must keep their exact width.
//  * anything wider than 16 bits, and anything already 32 bits.
//
// THE DEVIATION THIS ACCEPTS, stated plainly: a widened load reads up to three
// bytes past the object. On DICE that is not a new access -- the hardware
// already fetched those bytes and the whole 32-byte line they live in -- so this
// pass does not make any previously-safe program unsafe. It does mean the DICE
// target cannot be used with a byte-granular memory protection model. Recorded
// in docs/device-contract.md 5.5c.
//
//===----------------------------------------------------------------------===//

#include "NVPTX.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "nvptx-dice-widen-subword"

STATISTIC(NumWidened, "Number of sub-word loads widened to 32 bits");
STATISTIC(NumSubwordStores, "Number of sub-word stores left for dice-verify");

static cl::opt<bool> DiceWidenSubword(
    "nvptx-dice-widen-subword", cl::init(true), cl::Hidden,
    cl::desc("DICE: widen 8/16-bit loads to 32-bit loads plus a mask. On by "
             "default whenever the DICE pipeline is enabled; the fabric has no "
             "access-size field, so turning this off emits loads whose upper "
             "bytes are the neighbouring data."));

namespace {

class NVPTXDiceWidenSubword : public FunctionPass {
public:
  static char ID;
  NVPTXDiceWidenSubword() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override;

  StringRef getPassName() const override {
    return "NVPTX DICE widen sub-word loads";
  }
};

} // end anonymous namespace

char NVPTXDiceWidenSubword::ID = 0;

/// Sub-word integer load this pass can widen? Only plain, non-volatile,
/// non-atomic integer loads narrower than 32 bits.
static bool isWidenableSubwordLoad(const LoadInst *LI) {
  if (LI->isVolatile() || !LI->isSimple())
    return false;
  auto *IT = dyn_cast<IntegerType>(LI->getType());
  if (!IT)
    return false;
  unsigned Bits = IT->getBitWidth();
  // i1 is a bool in a byte; i8/i16 are the real cases. Anything else either
  // already fills a word or is not a legal memory width.
  return Bits == 1 || Bits == 8 || Bits == 16;
}

bool NVPTXDiceWidenSubword::runOnFunction(Function &F) {
  if (!nvptxDiceEnabled() || !DiceWidenSubword)
    return false;

  SmallVector<LoadInst *, 8> Work;
  for (BasicBlock &BB : F)
    for (Instruction &I : BB) {
      if (auto *LI = dyn_cast<LoadInst>(&I)) {
        if (isWidenableSubwordLoad(LI))
          Work.push_back(LI);
        continue;
      }
      // Not rewritten -- counted, so `-stats` shows what dice-verify will
      // refuse instead of it being discovered on the fabric.
      if (auto *SI = dyn_cast<StoreInst>(&I)) {
        Type *VT = SI->getValueOperand()->getType();
        if (VT->isIntegerTy() && VT->getIntegerBitWidth() < 32 &&
            SI->isSimple())
          ++NumSubwordStores;
      }
    }

  if (Work.empty())
    return false;

  Type *I32 = Type::getInt32Ty(F.getContext());
  for (LoadInst *LI : Work) {
    IRBuilder<> B(LI);
    // align 4, NOT align 1, and this is load-bearing rather than cosmetic.
    //
    // A DICE memory port performs a 4-byte read at an arbitrary BYTE address, so
    // there is no such thing as a misaligned 32-bit load on this target and no
    // alignment fault to avoid. Declaring `align 1` describes the pointer
    // honestly but describes the TARGET wrongly, and the consequence is not
    // subtle: NVPTX's TargetLowering does not allow misaligned memory accesses,
    // so the legalizer EXPANDS an `align 1` i32 load into four byte loads plus
    // shifts and ORs, and DAGCombine then deletes the three whose bytes the trunc
    // discards -- leaving exactly one `ld.global.u8`, i.e. this pass undone and
    // the sub-word load back on the fabric. Measured on m5_bytes: with align 1
    // the emitted PTX still read `ld.global.u8`; with align 4 it reads
    // `ld.global.u32`.
    LoadInst *Wide = B.CreateAlignedLoad(I32, LI->getPointerOperand(), Align(4));
    Wide->setName(LI->getName() + ".w32");
    // Carry the metadata that survives a width change. !invariant.load and
    // !nontemporal describe the ADDRESS, not the width; !range does not survive
    // and is deliberately dropped.
    Wide->copyMetadata(*LI, {LLVMContext::MD_invariant_load,
                             LLVMContext::MD_nontemporal});
    unsigned Bits = LI->getType()->getIntegerBitWidth();
    const uint64_t Mask = (Bits >= 64) ? ~0ULL : ((1ULL << Bits) - 1);

    // Rewrite the extension users first: they are the ones that would otherwise
    // become `cvt`, and they are the common case (a `char` or `bool` element
    // widened to `int` by the source language's promotion rules).
    SmallVector<Instruction *, 4> Dead;
    for (User *U : LI->users()) {
      auto *Ext = dyn_cast<CastInst>(U);
      if (!Ext || !(isa<ZExtInst>(Ext) || isa<SExtInst>(Ext)))
        continue;
      Type *DstTy = Ext->getType();
      if (!DstTy->isIntegerTy(32))
        continue;      // i64 extensions narrow elsewhere; leave them alone
      IRBuilder<> EB(Ext);
      Value *Rep;
      if (isa<ZExtInst>(Ext)) {
        Rep = EB.CreateAnd(Wide, ConstantInt::get(I32, Mask),
                           Ext->getName() + ".zx");
      } else {
        unsigned Sh = 32 - Bits;
        Value *Up = EB.CreateShl(Wide, ConstantInt::get(I32, Sh),
                                 Ext->getName() + ".up");
        Rep = EB.CreateAShr(Up, ConstantInt::get(I32, Sh),
                            Ext->getName() + ".sx");
      }
      Ext->replaceAllUsesWith(Rep);
      Dead.push_back(Ext);
    }
    for (Instruction *I : Dead)
      I->eraseFromParent();

    // Whatever still reads the narrow value gets `trunc(and w, mask)`. The `and`
    // is what makes the truncation explicit in the program rather than implicit
    // in a byte lane the hardware does not have.
    if (!LI->use_empty()) {
      Value *Masked = B.CreateAnd(Wide, ConstantInt::get(I32, Mask),
                                  LI->getName() + ".msk");
      Value *Narrow = B.CreateTrunc(Masked, LI->getType(),
                                    LI->getName() + ".sub");
      LI->replaceAllUsesWith(Narrow);
    }
    LI->eraseFromParent();
    ++NumWidened;
    LLVM_DEBUG(dbgs() << "DICE: widened a sub-word load in " << F.getName()
                      << "\n");
  }
  return true;
}

FunctionPass *llvm::createNVPTXDiceWidenSubwordPass() {
  return new NVPTXDiceWidenSubword();
}
