#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

namespace {
struct InstrumentationStrc : llvm::PassInfoMixin<InstrumentationStrc> {
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

    llvm::Function *func_f_f =
        llvm::dyn_cast<llvm::Function>(func_f.getCallee());
    llvm::Function *func_l_f =
        llvm::dyn_cast<llvm::Function>(func_l.getCallee());

    for (auto *U : func_f_f->users()) {
      if (auto *callInst = llvm::dyn_cast<llvm::CallInst>(U)) {
        if (callInst->getFunction() == &F) {
          foundInstrument_start = true;
        }
      }
    }
    for (auto *U : func_l_f->users()) {
      if (auto *callInst = llvm::dyn_cast<llvm::CallInst>(U)) {
        if (callInst->getFunction() == &F) {
          foundInstrument_end = true;
        }
      }
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
  return {LLVM_PLUGIN_API_VERSION, "instrumentation_function", "0.1",
          [](llvm::PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](llvm::StringRef name, llvm::FunctionPassManager &FPM,
                   llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) -> bool {
                  if (name == "instrumentation_function") {
                    FPM.addPass(InstrumentationStrc{});
                    return true;
                  }
                  return false;
                });
          }};
}