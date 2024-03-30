#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

namespace {
struct InstrumentationPass : llvm::PassInfoMixin<InstrumentationPass> {
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &) {
    llvm::LLVMContext &context = F.getContext();
    llvm::IRBuilder<> builder(context);
    llvm::Module *module = F.getParent();
    llvm::FunctionType *type =
        llvm::FunctionType::get(llvm::Type::getVoidTy(context), false);
    llvm::FunctionCallee func_f =
        (*module).getOrInsertFunction("instrument_start", type);
    llvm::FunctionCallee func_l =
        (*module).getOrInsertFunction("instrument_end", type);

    bool foundInstrument_start = false;
    bool foundInstrument_end = false;

    for (llvm::BasicBlock &BB : F) {
      for (llvm::Instruction &I : BB) {
        if (auto *callInst = llvm::dyn_cast<llvm::CallInst>(&I)) {
          if (callInst->getCalledFunction()->getName() ==
              func_f.getCallee()->getName()) {
            foundInstrument_start = true;
          } else if (callInst->getCalledFunction()->getName() ==
                     func_l.getCallee()->getName()) {
            foundInstrument_end = true;
          }
        }
      }
      if (foundInstrument_start && foundInstrument_end)
        break;
    }

    if (!foundInstrument_start) {
      builder.SetInsertPoint(&F.getEntryBlock().front());
      builder.CreateCall(func_f);
    }
    llvm::ReturnInst *re;
    if (!foundInstrument_end) {
      for (llvm::BasicBlock &BB : F) {
        if ((re = llvm::dyn_cast<llvm::ReturnInst>(BB.getTerminator())) !=
            NULL) {
          builder.SetInsertPoint(BB.getTerminator());
          builder.CreateCall(func_l);
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
  return {LLVM_PLUGIN_API_VERSION, "instrumentation", "0.1",
          [](llvm::PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](llvm::StringRef name, llvm::FunctionPassManager &FPM,
                   llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) -> bool {
                  if (name == "instrumentation") {
                    FPM.addPass(InstrumentationPass{});
                    return true;
                  }
                  return false;
                });
          }};
}