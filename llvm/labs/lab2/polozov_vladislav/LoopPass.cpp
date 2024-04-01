#include "llvm/Passes/PassBuilder.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/IR/IRBuilder.h"

struct LoopPass : public llvm::PassInfoMixin<LoopPass> {
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM) {
    llvm::Module *ParentModule = F.getParent();
    llvm::FunctionType *myFuncType =
        llvm::FunctionType::get(llvm::Type::getVoidTy(F.getContext()), false);
    llvm::FunctionCallee loopStartFunc =
        ParentModule->getOrInsertFunction("loop_start", myFuncType);
    llvm::FunctionCallee loopEndFunc =
        ParentModule->getOrInsertFunction("loop_end", myFuncType);
    llvm::LoopAnalysis::Result &LI = FAM.getResult<llvm::LoopAnalysis>(F);
    for (auto *L : LI) {
      insertIntoLoopFuncStartEnd(L, loopStartFunc, loopEndFunc);
    }
    auto PA = llvm::PreservedAnalyses::all();
    PA.abandon<llvm::LoopAnalysis>();
    return PA;
  }

  void insertIntoLoopFuncStartEnd(llvm::Loop *L, llvm::FunctionCallee loopStart,
                          llvm::FunctionCallee loopEnd) {
    llvm::IRBuilder<> Builder(L->getHeader()->getContext());
    llvm::SmallVector<llvm::BasicBlock *, 1> ExitBlocks;
    L->getExitBlocks(ExitBlocks);
    for (auto *const BB : ExitBlocks) {
      if (isCalled(BB, loopEnd) == false) {
        Builder.SetInsertPoint(BB->getFirstNonPHI());
        Builder.CreateCall(loopEnd);
      }
    }
    llvm::BasicBlock *Header = L->getHeader();
    for (auto it = llvm::pred_begin(Header), et = llvm::pred_end(Header); it != et; ++it) {
      llvm::BasicBlock* Pred = *it;
      if (L->contains(Pred) == false && isCalled(Pred, loopStart) == false) {
        Builder.SetInsertPoint(Pred->getTerminator());
        Builder.CreateCall(loopStart);
      }
    }
  }

  bool isCalled(llvm::BasicBlock *const BB, llvm::FunctionCallee &callee) {
    bool called = false;
    for (auto &inst : *BB) {
      if (auto *instCall = llvm::dyn_cast<llvm::CallInst>(&inst)) {
        if (instCall->getCalledFunction() == callee.getCallee()) {
          called = true;
          break;
        }
      }
    }
    return called;
  }

};

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "LoopPass", LLVM_VERSION_STRING,
          [](llvm::PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](llvm::StringRef Name, llvm::FunctionPassManager &PM,
                   llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
                  if (Name == "polozov-loop-plugin") {
                    PM.addPass(LoopPass());
                    return true;
                  }
                  return false;
                });
          }};
}