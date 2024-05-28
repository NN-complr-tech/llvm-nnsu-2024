#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

using namespace llvm;

struct AddPass : public PassInfoMixin<AddPass> {
PreservedAnalyses run(Function &Func, FunctionAnalysisManager &FuncAnalysisMgr) {

    LoopInfo &LoopInf = FuncAnalysisMgr.getResult<LoopAnalysis>(Func);
    auto &Context = Func.getContext();
    Module *ParentModule = Func.getParent();
    FunctionType *FuncType = FunctionType::get(Type::getVoidTy(Context), false);

    for (auto &Loop : LoopInf) {
      BasicBlock *LoopHeader = Loop->getHeader();
      BasicBlock *PreheaderBlock = Loop->getLoopPreheader();
      IRBuilder<> IRBuild(LoopHeader->getContext());

      if (PreheaderBlock != nullptr) {
        bool IsLoopStartFunc = false;
        for (auto &Inst : *PreheaderBlock) {
          if (auto *CallInst = dyn_cast<CallInst>(&Inst)) {
            if (CallInst->getCalledFunction() &&
                CallInst->getCalledFunction()->getName() == "loop_start") {
              IsLoopStartFunc = true;
              break;
            }
          }
        }
        if (!IsLoopStartFunc) {
          IRBuild.SetInsertPoint(PreheaderBlock->getTerminator());
          IRBuild.CreateCall(ParentModule->getOrInsertFunction("loop_start", FuncType));
        }
      }
    }
    return PreservedAnalyses::all();
  }
};

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "AddNewPlugin", LLVM_VERSION_STRING,
          [](llvm::PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](llvm::StringRef Name, llvm::FunctionPassManager &PM,
                   llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
                  if (Name == "loop_func") {
                    PM.addPass(AddPass());
                    return true;
                  }
                  return false;
                });
          }};
}
