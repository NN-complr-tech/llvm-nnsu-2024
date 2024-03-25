#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

namespace {
struct InstrumentFunctions : llvm::PassInfoMixin<InstrumentFunctions> {
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &) {
    llvm::errs() << "Function name: " << F.getName() << '\n';
    llvm::errs() << "Arg size: " << F.arg_size() << '\n';
    return llvm::PreservedAnalyses::all();
  }

  static bool isRequired() { return true; }
};
}  // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "InstrumentFunctions", "0.1",
          [](llvm::PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](llvm::StringRef name, llvm::FunctionPassManager &FPM,
                   llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) -> bool {
                  if (name == "instr_func") {
                    FPM.addPass(InstrumentFunctions{});
                    return true;
                  }
                  return false;
                });
          }};
}