#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

struct LoopFramer : public llvm::PassInfoMixin<LoopFramer> {
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM) {
    llvm::LLVMContext &Context = F.getContext();
    llvm::Module *ParentModule = F.getParent();

    llvm::FunctionType *funcType =
        llvm::FunctionType::get(llvm::Type::getVoidTy(Context), false);
    llvm::FunctionCallee loopStart =
        ParentModule->getOrInsertFunction("loop_start", funcType);
    llvm::FunctionCallee loopEnd =
        ParentModule->getOrInsertFunction("loop_end", funcType);

    llvm::LoopAnalysis::Result &LI = FAM.getResult<llvm::LoopAnalysis>(F);
    for (auto *L : LI) {
      insertLoopStartEnd(L, loopStart, loopEnd);
    }

    auto PA = llvm::PreservedAnalyses::all();
    PA.abandon<llvm::LoopAnalysis>();
    PA.abandon<llvm::DominatorTreeAnalysis>();
    return PA;
  }

  void insertLoopStartEnd(llvm::Loop *L, llvm::FunctionCallee loopStart,
                          llvm::FunctionCallee loopEnd) {
    llvm::LLVMContext &Context = L->getHeader()->getContext();
    llvm::IRBuilder<> Builder(Context);

    llvm::BasicBlock *Header = L->getHeader();
    for (auto *const Pre :
         llvm::children<llvm::Inverse<llvm::BasicBlock *>>(Header)) {
      if (!(L->contains(Pre))) {
        Builder.SetInsertPoint(Pre->getTerminator());
        Builder.CreateCall(loopStart);
      }
    }

    llvm::SmallVector<llvm::BasicBlock *, 4> ExitBlocks;
    L->getExitBlocks(ExitBlocks);
    for (auto *const BB : ExitBlocks) {
      Builder.SetInsertPoint(BB->getFirstNonPHI());
      Builder.CreateCall(loopEnd);
    }
  }
};

bool registerPipeline(llvm::StringRef Name, llvm::FunctionPassManager &FPM,
                      llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
  if (Name == "LoopFramer") {
    FPM.addPass(LoopFramer());
    return true;
  }
  return false;
}

llvm::PassPluginLibraryInfo getLoopFramerPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "LoopFramer", LLVM_VERSION_STRING,
          [](llvm::PassBuilder &PB) {
            PB.registerPipelineParsingCallback(registerPipeline);
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getLoopFramerPluginInfo();
}
