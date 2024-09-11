#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

using namespace llvm;

struct ZawadaLoopPass : public PassInfoMixin<ZawadaLoopPass> {
  PreservedAnalyses run(Function &function,
                        FunctionAnalysisManager &functionAnalysisManager) {
    LoopInfo &loopInfo =
        functionAnalysisManager.getResult<LoopAnalysis>(function);
    auto &context = function.getContext();
    FunctionType *functionType =
        FunctionType::get(Type::getVoidTy(context), false);
    Module *parent = function.getParent();

    for (auto &loop : loopInfo) {
      IRBuilder<> builder(loop->getHeader()->getContext());
      if (auto *entryBlock = loop->getLoopPreheader()) {
        builder.SetInsertPoint(entryBlock->getTerminator());
        builder.CreateCall(parent->getOrInsertFunction("loop_start", functionType));
      }

      SmallVector<BasicBlock *, 8> exitBlock;
      loop->getExitBlocks(exitBlock);
      for (auto *exit : exitBlock) {
        builder.SetInsertPoint(&*exit->getFirstInsertionPt());
        builder.CreateCall(parent->getOrInsertFunction("loop_end", functionType));
      }
    }

    return PreservedAnalyses::all();
  }
};

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "AddNewPlugin", LLVM_VERSION_STRING,
          [](llvm::PassBuilder &passBuilder) {
            passBuilder.registerPipelineParsingCallback(
                [](llvm::StringRef name, llvm::FunctionPassManager &passManager,
                   llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
                  if (name != "loop_func")
                    return false;
                  passManager.addPass(ZawadaLoopPass());
                  return true;
                });
          }};
}
