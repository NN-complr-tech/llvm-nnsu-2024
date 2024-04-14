#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

namespace {
struct instrFunct : llvm::PassInfoMixin<instrFunct> {
  llvm::PreservedAnalyses run(llvm::Function &func,
                              llvm::FunctionAnalysisManager &) {
    llvm::LLVMContext &context = func.getContext();
    llvm::IRBuilder<> builder(context);
    auto module = func.getParent();

    llvm::FunctionType *funcType =
        llvm::FunctionType::get(llvm::Type::getVoidTy(context), false);
    llvm::FunctionCallee instrStart =
        module->getOrInsertFunction("instrument_start", funcType);
    llvm::FunctionCallee instrEnd =
        module->getOrInsertFunction("instrument_end", funcType);

    builder.SetInsertPoint(&func.getEntryBlock().front());
    builder.CreateCall(instrStart);

    for (auto &block : func) {
      if (llvm::isa<llvm::ReturnInst>(block.getTerminator())) {
        builder.SetInsertPoint(block.getTerminator());
        builder.CreateCall(instrEnd);
      }
    }

    return llvm::PreservedAnalyses::all();
  }

  static bool isRequired() { return true; }
};
} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "instrumentation_functions", "0.1",
          [](llvm::PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](llvm::StringRef name, llvm::FunctionPassManager &FPM,
                   llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) -> bool {
                  if (name == "instrumentation-functions") {
                    FPM.addPass(instrFunct{});
                    return true;
                  }
                  return false;
                });
          }};
}
