#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

namespace {
struct InsFunc : llvm::PassInfoMixin<InsFunc> {
  llvm::PreservedAnalyses run(llvm::Function &func,
                              llvm::FunctionAnalysisManager &) {
    llvm::LLVMContext &context = func.getContext();
    llvm::IRBuilder<> bld(context);
    llvm::Module *mdl = func.getParent();
    bool start = false;
    bool end = false;

    llvm::FunctionType *typeFunc =
        llvm::FunctionType::get(llvm::Type::getVoidTy(context), false);
    llvm::FunctionCallee startIns =
        (*mdl).getOrInsertFunction("instrument_start", typeFunc);
    llvm::FunctionCallee endIns =
        (*mdl).getOrInsertFunction("instrument_end", typeFunc);

    for (auto &block : func) {
      for (auto &instruction : block) {
        if (llvm::isa<llvm::CallInst>(&instruction)) {
          llvm::CallInst *callInst = llvm::cast<llvm::CallInst>(&instruction);
          if (callInst->getCalledFunction() == startIns.getCallee()) {
            start = true;
          } else if (callInst->getCalledFunction() == endIns.getCallee()) {
            end = true;
          }
        }
      }
    }

    if (!start) {
      bld.SetInsertPoint(&func.getEntryBlock().front());
      bld.CreateCall(startIns);
    }
    if (!end) {
      for (llvm::BasicBlock &BB : func) {
        if (llvm::dyn_cast<llvm::ReturnInst>(BB.getTerminator())) {
          bld.SetInsertPoint(BB.getTerminator());
          bld.CreateCall(endIns);
        }
      }
    }

    return llvm::PreservedAnalyses::all();
  }

  static bool isRequired() { return true; }
};
} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "instrument_func", "0.1",
          [](llvm::PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](llvm::StringRef name, llvm::FunctionPassManager &FPM,
                   llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) -> bool {
                  if (name == "instrument-func") {
                    FPM.addPass(InsFunc{});
                    return true;
                  }
                  return false;
                });
          }};
}