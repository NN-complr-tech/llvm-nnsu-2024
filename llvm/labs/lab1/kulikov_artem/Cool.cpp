#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

struct ForWrapper : PassInfoMixin<ForWrapper> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) {
    auto &LI = FAM.getResult<LoopAnalysis>(F);
    for (auto &L : LI) {
      auto preheader = L->getLoopPreheader();
      if (!preheader) {
        continue;
      }
      IRBuilder<> Builder(preheader->getTerminator());
      Function *loopStartFunc = F.getParent()->getFunction("loop_start");

      if (!loopStartFunc) {
        loopStartFunc = Function::Create(
            FunctionType::get(Type::getVoidTy(F.getContext()), false),
            Function::ExternalLinkage, "loop_start", F.getParent());
      }
      Builder.CreateCall(loopStartFunc);

      auto exitBlock = L->getExitBlock();
      if (!exitBlock) {
        continue;
      }
      Builder.SetInsertPoint(exitBlock->getTerminator());
      Function *loopEndFunc = F.getParent()->getFunction("loop_end");
      if (!loopEndFunc) {
        loopEndFunc = Function::Create(
            FunctionType::get(Type::getVoidTy(F.getContext()), false),
            Function::ExternalLinkage, "loop_end", F.getParent());
      }
      Builder.CreateCall(loopEndFunc);
    }

    auto PA = PreservedAnalyses::all();
    PA.abandon<LoopAnalysis>();
    return PA;
  }
};

} // namespace

#define PPCAT_NX2(A, B, C) A##B##C
#define PPCAT2(A, B, C) PPCAT_NX2(A, B, C)
#define getNAMEPluginInfo PPCAT2(get, NAME, PluginInfo())

/* New PM Registration */
llvm::PassPluginLibraryInfo getNAMEPluginInfo {
  return {LLVM_PLUGIN_API_VERSION, "ForWrapperPlugin", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, llvm::FunctionPassManager &PM,
                   ArrayRef<llvm::PassBuilder::PipelineElement>) {
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
  return getNAMEPluginInfo;
}
