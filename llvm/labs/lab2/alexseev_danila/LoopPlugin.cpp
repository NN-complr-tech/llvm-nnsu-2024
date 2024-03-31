#include "llvm/IR/IRBuilder.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

struct LoopPlugin : public llvm::PassInfoMixin<LoopPlugin> {
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM) {
    llvm::LLVMContext &Context = F.getContext();
    llvm::Module *ParentModule = F.getParent();

    llvm::FunctionType *funcType =
        llvm::FunctionType::get(llvm::Type::getVoidTy(Context), false);

    llvm::LoopAnalysis::Result &LI = FAM.getResult<llvm::LoopAnalysis>(F);
    for (auto *Loop : LI) {
      llvm::IRBuilder<> Builder(Loop->getHeader()->getContext());
      llvm::BasicBlock *Preheader = Loop->getLoopPreheader();
      llvm::BasicBlock *ExitBlock = Loop->getExitBlock();
	  bool IsStartLoopNotFound = llvm::findFunction(*ParentModule, "loop_start") == nullptr;
      bool IsEndLoopNotFound = llvm::findFunction(*ParentModule, "loop_end") == nullptr;
      if (Preheader && ExitBlock && IsStartLoopNotFound) {
        Builder.SetInsertPoint(Preheader->getTerminator());
        Builder.CreateCall(
            ParentModule->getOrInsertFunction("loop_start", funcType));
      }
      if (Preheader && ExitBlock && IsEndLoopNotFound) {
        Builder.SetInsertPoint(&*ExitBlock->getFirstInsertionPt());
        Builder.CreateCall(
            ParentModule->getOrInsertFunction("loop_end", funcType));
      }
    }
    return llvm::PreservedAnalyses::all();
  }
};

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "LoopPlugin", LLVM_VERSION_STRING,
          [](llvm::PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](llvm::StringRef Name, llvm::FunctionPassManager &PM,
                   llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
                  if (Name == "alexseev-loop-plugin") {
                    PM.addPass(LoopPlugin());
                    return true;
                  }
                  return false;
                });
          }};
}