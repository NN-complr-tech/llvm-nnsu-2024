#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

using namespace llvm;

class MyLoopPass : public PassInfoMixin<MyLoopPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) {
    auto *Ty = FunctionType::get(llvm::Type::getVoidTy(F.getContext()),
                                 false); // get func type
    auto *M = F.getParent();

    auto LoopStartFunc =
        M->getOrInsertFunction("loop_start", Ty); // create func start
    auto LoopEndFunc =
        M->getOrInsertFunction("loop_end", Ty); // create func end

    auto &LI = FAM.getResult<LoopAnalysis>(F);

    for (auto &L : LI) { // for by loop
      auto *Preheader =
          L->getLoopPreheader();           // get preheader block of the loop
      auto *ExitBlock = L->getExitBlock(); // end exit block of the loop
      IRBuilder<> Builder(Preheader->getTerminator()); // api for basic block

      if (!Preheader)
        continue;

      Builder.CreateCall(LoopStartFunc); // paste loop_start

      if (!ExitBlock)
        continue;

      Builder.SetInsertPoint(
          ExitBlock->getFirstNonPHI()); // set pointer to exit block
      Builder.CreateCall(LoopEndFunc);  // paste loop_start
    }
    return PreservedAnalyses::all();
  }
};

/* New PM Registration */
llvm::PassPluginLibraryInfo getAtikinLoopPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "AtikinLoopPass", LLVM_VERSION_STRING,
          [](llvm::PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](llvm::StringRef Name, llvm::FunctionPassManager &PM,
                   llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
                  if (Name == "ivanov-loop-pass") {
                    PM.addPass(MyLoopPass());
                    return true;
                  }
                  return false;
                });
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getAtikinLoopPassPluginInfo();
}
