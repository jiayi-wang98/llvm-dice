//===-- NVPTXMachineFunctionInfo.h - NVPTX-specific Function Info  --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class is attached to a MachineFunction instance and tracks target-
// dependent information
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NVPTX_NVPTXMACHINEFUNCTIONINFO_H
#define LLVM_LIB_TARGET_NVPTX_NVPTXMACHINEFUNCTIONINFO_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/MachineFunction.h"
#include <map>

namespace llvm {
class CallBase;

class NVPTXMachineFunctionInfo : public MachineFunctionInfo {
private:
  /// DICE architectural register assignment (set by NVPTXDiceRegAlloc).
  /// Queryable state: the asm printer names registers with it, and later
  /// passes (unrolling-factor selection) read the real indices.
  DenseMap<Register, std::string> DiceRegNames;
  unsigned DiceCounts[4] = {0, 0, 0, 0}; // r, c, p, w
  /// Rendered .meta records for this function (NVPTXDiceRegAlloc). The asm
  /// printer emits them as trailing DICE_META comments; dicc splits them out.
  std::string DiceMetaText;
  /// DICE predication guards from if-conversion: instruction -> (predicate
  /// vreg, negated). The instruction also carries the predicate as an
  /// implicit use so liveness and allocation see it; this table is what the
  /// asm printer reads to emit the @%p prefix marker.
  DenseMap<const MachineInstr *, std::pair<Register, bool>> DicePredGuards;

  /// Stores a mapping from index to symbol name for image handles that are
  /// replaced with image references
  SmallVector<std::string, 8> ImageHandleList;

  /// Stores a mapping from a unique call-site id to the call instruction that
  /// needs an indirect-call prototype emitted.
  std::map<unsigned, const CallBase *> CallPrototypes;

public:
  NVPTXMachineFunctionInfo(const Function &F, const TargetSubtargetInfo *STI) {}

  void setDiceRegName(Register R, std::string Name) {
    DiceRegNames[R] = std::move(Name);
  }
  bool hasDiceRegNames() const { return !DiceRegNames.empty(); }
  const std::string *getDiceRegName(Register R) const {
    auto It = DiceRegNames.find(R);
    return It == DiceRegNames.end() ? nullptr : &It->second;
  }
  void clearDiceRegNames() { DiceRegNames.clear(); }
  void setDicePredGuard(const MachineInstr *MI, Register P, bool Neg) {
    DicePredGuards[MI] = {P, Neg};
  }
  const std::pair<Register, bool> *getDicePredGuard(
      const MachineInstr *MI) const {
    auto It = DicePredGuards.find(MI);
    return It == DicePredGuards.end() ? nullptr : &It->second;
  }
  bool hasDicePredGuards() const { return !DicePredGuards.empty(); }
  void setDiceMetaText(std::string T) { DiceMetaText = std::move(T); }
  const std::string &getDiceMetaText() const { return DiceMetaText; }

  void setDiceRegCounts(unsigned R, unsigned C, unsigned P, unsigned W) {
    DiceCounts[0] = R; DiceCounts[1] = C; DiceCounts[2] = P; DiceCounts[3] = W;
  }
  ArrayRef<unsigned> getDiceRegCounts() const { return DiceCounts; }

  MachineFunctionInfo *
  clone(BumpPtrAllocator &Allocator, MachineFunction &DestMF,
        const DenseMap<MachineBasicBlock *, MachineBasicBlock *> &Src2DstMBB)
      const override {
    return DestMF.cloneInfo<NVPTXMachineFunctionInfo>(*this);
  }

  /// Returns the index for the symbol \p Symbol. If the symbol was previously,
  /// added, the same index is returned. Otherwise, the symbol is added and the
  /// new index is returned.
  unsigned getImageHandleSymbolIndex(StringRef Symbol) {
    // Is the symbol already present?
    for (unsigned i = 0, e = ImageHandleList.size(); i != e; ++i)
      if (ImageHandleList[i] == Symbol)
        return i;
    // Nope, insert it
    ImageHandleList.push_back(Symbol.str());
    return ImageHandleList.size()-1;
  }

  /// Check if the symbol has a mapping. Having a mapping means the handle is
  /// replaced with a reference
  bool checkImageHandleSymbol(StringRef Symbol) const {
    return llvm::is_contained(ImageHandleList, Symbol);
  }

  void addCallPrototype(unsigned Id, const CallBase *CB) {
    CallPrototypes.try_emplace(Id, CB);
  }

  const std::map<unsigned, const CallBase *> &getCallPrototypes() const {
    return CallPrototypes;
  }
};
}

#endif
