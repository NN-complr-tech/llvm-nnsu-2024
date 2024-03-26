#include "llvm/ADT/ArrayRef.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/Compiler.h"

namespace {

struct ForWrapper : llvm::PassInfoMixin<ForWrapper> {
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM) {
    auto *Ty =
        llvm::FunctionType::get(llvm::Type::getVoidTy(F.getContext()), false);
    auto *M = F.getParent();
    auto LoopStartFunc = M->getOrInsertFunction("loop_start", Ty);
    auto LoopEndFunc = M->getOrInsertFunction("loop_end", Ty);

    auto &LI = FAM.getResult<llvm::LoopAnalysis>(F);
    for (auto &L : LI) {
      auto *Preheader = L->getLoopPreheader();
      if (!Preheader)
        continue;

      llvm::IRBuilder<> Builder(Preheader->getTerminator());
      Builder.CreateCall(LoopStartFunc);

      auto *ExitBlock = L->getExitBlock();
      if (!ExitBlock)
        continue;

      Builder.SetInsertPoint(ExitBlock->getFirstNonPHI());
      Builder.CreateCall(LoopEndFunc);
    }

    auto PA = llvm::PreservedAnalyses::all();
    PA.abandon<llvm::LoopAnalysis>();
    return PA;
  }
};

} // namespace

#define PPCAT_NX2(A, B, C) A##B##C
#define PPCAT2(A, B, C) PPCAT_NX2(A, B, C)
#define getNAMEPluginInfo PPCAT2(get, NAME, PluginInfo)

/* New PM Registration */
llvm::PassPluginLibraryInfo getNAMEPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "ForWrapperPlugin", LLVM_VERSION_STRING,
          [](llvm::PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](llvm::StringRef Name, llvm::FunctionPassManager &PM,
                   llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
                  if (Name == "kulikov-wrap-plugin") {
                    PM.addPass(ForWrapper());
                    return true;
                  }
                  return false;
                });
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getNAMEPluginInfo();
}
