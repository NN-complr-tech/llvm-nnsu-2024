#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

using namespace llvm;

namespace {
struct AddPass : public PassInfoMixin<AddPass> {
  PreservedAnalyses run(Function &Func,
                        FunctionAnalysisManager &FuncAnalysisMgr) {

    LoopInfo &LoopInf = FuncAnalysisMgr.getResult<LoopAnalysis>(Func);
    auto &Context = Func.getContext();
    Module *ParentModule = Func.getParent();
    FunctionType *FuncType = FunctionType::get(Type::getVoidTy(Context), false);
    FunctionCallee loopStartFunc =
        ParentModule->getOrInsertFunction("loop_start", FuncType);
    FunctionCallee loopEndFunc =
        ParentModule->getOrInsertFunction("loop_end", FuncType);

    for (auto &Loop : LoopInf) { // Iterate over all loops in the function
      BasicBlock *LoopHeader = Loop->getHeader();
      // Insert loop_start at the terminators of the predecessors of the loop
      // header
      for (auto *Pred : predecessors(LoopHeader)) {
        if (!Loop->contains(Pred)) {
          bool hasLoopStart = false;
          for (auto &I : *Pred) {
            if (auto *CI = dyn_cast<CallInst>(&I)) {
              if (CI->getCalledFunction() == loopStartFunc.getCallee()) {
                hasLoopStart = true;
                break;
              }
            }
          }
          if (!hasLoopStart) {
            IRBuilder<> builder(&*Pred->getTerminator());
            builder.CreateCall(loopStartFunc);
          }
        }
      }

      // Insert loop_end at the first insertion points of the successors of the
      // exiting blocks
      SmallVector<BasicBlock *, 8> ExitingBlocks;
      Loop->getExitingBlocks(ExitingBlocks);
      for (auto *ExitingBlock : ExitingBlocks) {
        for (auto it = succ_begin(ExitingBlock), e = succ_end(ExitingBlock);
             it != e; ++it) {
          BasicBlock *Successor = *it;
          if (!Loop->contains(Successor)) {
            bool hasLoopEnd = false;
            for (auto &I : *Successor) {
              if (auto *CI = dyn_cast<CallInst>(&I)) {
                if (CI->getCalledFunction() == loopEndFunc.getCallee()) {
                  hasLoopEnd = true;
                  break;
                }
              }
            }
            if (!hasLoopEnd) {
              IRBuilder<> builder(&*Successor->getFirstInsertionPt());
              builder.CreateCall(loopEndFunc);
            }
            break;
          }
        }
      }
    }
    return PreservedAnalyses::all();
  }
};
} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "AddNewPlugin", LLVM_VERSION_STRING,
          [](llvm::PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](llvm::StringRef Name, llvm::FunctionPassManager &PM,
                   llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
                  if (Name == "loop_func_kalinin") {
                    PM.addPass(AddPass());
                    return true;
                  }
                  return false;
                });
          }};
}
