#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

namespace {
struct InstrumentFunctionsPass : llvm::PassInfoMixin<InstrumentFunctionsPass> {
  // Function to check if function call is instrumented
  bool isInstrumentedCall(llvm::Instruction &inst, llvm::FunctionCallee &func) {
    if (llvm::CallInst *callInst = llvm::dyn_cast<llvm::CallInst>(&inst)) {
      if (callInst->getCalledFunction() == func.getCallee()) {
        return true;
      }
    }
    return false;
  }

  // Function to insert instrument function call if not already inserted
  void insertInstrumentCall(llvm::Function &F, llvm::FunctionCallee &func,
                            llvm::IRBuilder<> &builder, bool &inserted) {
    if (!inserted) {
      builder.SetInsertPoint(&F.getEntryBlock().front());
      builder.CreateCall(func);
      inserted = true;
    }
  }

  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &) {
    llvm::LLVMContext &context = F.getContext();
    llvm::IRBuilder<> builder(context);
    llvm::Module *module = F.getParent();

    // Get the instrument_start() and instrument_end() functions
    llvm::FunctionType *funcType =
        llvm::FunctionType::get(llvm::Type::getVoidTy(context), false);
    llvm::FunctionCallee startFunc =
        module->getOrInsertFunction("instrument_start", funcType);
    llvm::FunctionCallee endFunc =
        module->getOrInsertFunction("instrument_end", funcType);

    // Check if instrument_start() or instrument_end() has been already inserted
    bool startInserted = false;
    bool endInserted = false;

    // Check for existing instrument calls in the function
    for (auto &block : F) {
      for (auto &instruction : block) {
        if (isInstrumentedCall(instruction, startFunc)) {
          startInserted = true;
        }
        if (isInstrumentedCall(instruction, endFunc)) {
          endInserted = true;
        }
      }
    }

    // Insert instrument_start() if not already inserted
    insertInstrumentCall(F, startFunc, builder, startInserted);

    if (!endInserted) {
      // Insert instrument_end() if not already inserted
      llvm::Instruction *terminator = nullptr;
      for (auto &BB : F) {
        if (llvm::isa<llvm::ReturnInst>(BB.getTerminator())) {
          terminator = BB.getTerminator();
          break;
        }
      }
      if (terminator) {
        builder.SetInsertPoint(terminator);
        builder.CreateCall(endFunc);
        endInserted = true;
      }
    }

    return llvm::PreservedAnalyses::all();
  }

  static bool isRequired() { return true; }
};
} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "InstrumentFunctionsPass", "0.1",
          [](llvm::PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](llvm::StringRef name, llvm::FunctionPassManager &FPM,
                   llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) -> bool {
                  if (name == "instrument_functions_pass") {
                    FPM.addPass(InstrumentFunctionsPass{});
                    return true;
                  }
                  return false;
                });
          }};
}
