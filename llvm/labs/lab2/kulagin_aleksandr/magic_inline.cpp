#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Value.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/Casting.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <map>
#include <vector>

namespace {
llvm::BasicBlock *splitBlockBefore(llvm::BasicBlock *Old,
                                   llvm::Instruction *SplitPt,
                                   llvm::DomTreeUpdater *DTU = nullptr,
                                   llvm::LoopInfo *LI = nullptr,
                                   llvm::MemorySSAUpdater *MSSAU = nullptr,
                                   const llvm::Twine &BBName = "",
                                   bool Before = true) {
  return llvm::SplitBlock(Old, SplitPt, DTU, LI, MSSAU, BBName, Before);
}

class KulaginMagicInlinePass
    : public llvm::PassInfoMixin<KulaginMagicInlinePass> {
public:
  llvm::PreservedAnalyses run(llvm::Function &Func,
                              llvm::FunctionAnalysisManager &) {
    std::vector<llvm::Instruction *> InstrDeleteList;
    llvm::LLVMContext &CTX = Func.getContext();
    if (Func.getReturnType()->isVoidTy() &&
        Func.getFunctionType()->getNumParams() == 0) {
      EmptyFuncs.push_back(&Func);
    }
    for (llvm::BasicBlock &Block : Func) {
      for (llvm::Instruction &Instr : Block) {
        if (llvm::CallInst *CallInstruction =
                llvm::dyn_cast<llvm::CallInst>(&Instr)) {
          llvm::Function *Called = CallInstruction->getCalledFunction();
          Called = findEmptyFunction(Called);
          if (!Called || Called == &Func) {
            continue;
          }
          std::map<llvm::BasicBlock *, llvm::BasicBlock *> BlockMap;
          llvm::ValueToValueMapTy VMap;
          llvm::BasicBlock *SplittedBlock = splitBlockBefore(&Block, &Instr);
          llvm::BasicBlock *NextBlockPlace = SplittedBlock->getNextNode();
          for (llvm::BasicBlock &EmptyBasicBlock : *Called) {
            llvm::BasicBlock *TmpBlock =
                llvm::BasicBlock::Create(CTX, "", &Func, NextBlockPlace);
            BlockMap[&EmptyBasicBlock] = TmpBlock;
            NextBlockPlace = TmpBlock->getNextNode();
          }
          for (llvm::BasicBlock &EmptyBasicBlock : *Called) {
            for (llvm::Instruction &EmptyInstr : EmptyBasicBlock) {
              llvm::BasicBlock *CurrentBlock = BlockMap[&EmptyBasicBlock];
              if (llvm::isa<llvm::ReturnInst>(&EmptyInstr)) {
                llvm::BranchInst::Create(&Block)->insertInto(
                    CurrentBlock, CurrentBlock->end());
              } else {
                llvm::Instruction *NewInstr = EmptyInstr.clone();
                if (llvm::isa<llvm::BranchInst>(*NewInstr)) {
                  unsigned OpNum = NewInstr->getNumOperands();
                  for (unsigned Ind = 0; Ind < OpNum; Ind++) {
                    llvm::Value *CurOperand = NewInstr->getOperand(Ind);
                    if (llvm::isa<llvm::BasicBlock>(*CurOperand)) {
                      NewInstr->setOperand(
                          Ind, llvm::dyn_cast<llvm::Value>(
                                   BlockMap[llvm::dyn_cast<llvm::BasicBlock>(
                                       CurOperand)]));
                    }
                  }
                }
                NewInstr->insertInto(CurrentBlock, CurrentBlock->end());
                llvm::RemapInstruction(NewInstr, VMap,
                                       llvm::RF_NoModuleLevelChanges |
                                           llvm::RF_IgnoreMissingLocals);
                VMap[&EmptyInstr] = NewInstr;
              }
            }
          }
          SplittedBlock->getTerminator()->setOperand(
              0,
              llvm::dyn_cast<llvm::Value>(BlockMap[&Called->getEntryBlock()]));
          InstrDeleteList.push_back(&Instr);
        }
      }
    }
    for (llvm::Instruction *Instr : InstrDeleteList) {
      Instr->eraseFromParent();
    }
    return llvm::PreservedAnalyses::all();
  }

private:
  std::vector<llvm::Function *> EmptyFuncs;
  llvm::Function *findEmptyFunction(const llvm::Function *Func) {
    for (llvm::Function *F : EmptyFuncs) {
      if (F == Func) {
        return F;
      }
    }
    return nullptr;
  }
};
} // namespace

llvm::PassPluginLibraryInfo
getPutInlineFunctionsKulaginAleksandrFI3PluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "KulaginMagicInlinePass", "0.1",
          [](llvm::PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](llvm::StringRef Name, llvm::FunctionPassManager &PM,
                   llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
                  if (Name == "kulagin-magic-inline") {
                    PM.addPass(KulaginMagicInlinePass());
                    return true;
                  }
                  return false;
                });
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getPutInlineFunctionsKulaginAleksandrFI3PluginInfo();
}
