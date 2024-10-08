#include "llvm/IR/Function.h" 
#include "llvm/IR/Instructions.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

namespace {
struct FuncInstrumentationPass : llvm::PassInfoMixin<FuncInstrumentationPass> {

  bool hasInstrumentCalls(llvm::Function &func, llvm::FunctionCallee &calleeFunc) {
    for (auto *user : calleeFunc.getCallee()->users()) {
      if (auto *callInstruction = llvm::dyn_cast<llvm::CallInst>(user)) {
        if (callInstruction->getParent()->getParent() == &func) {
          return true;
        }
      }
    }
    return false;
  }

  void addInstrumentCall(llvm::Function &func, llvm::FunctionCallee &calleeFunc,
                         llvm::IRBuilder<> &irBuilder, bool &alreadyInserted) {
    if (!alreadyInserted) {
      irBuilder.SetInsertPoint(&func.getEntryBlock().front());
      irBuilder.CreateCall(calleeFunc);
      alreadyInserted = true;
    }
  }

  llvm::ReturnInst *findFinalReturnInst(llvm::Function &func) {
    llvm::ReturnInst *finalReturnInst = nullptr;
    for (llvm::BasicBlock &block : func) {
      llvm::Instruction *terminatorInst = block.getTerminator();
      if (llvm::isa<llvm::ReturnInst>(terminatorInst)) {
        finalReturnInst = llvm::cast<llvm::ReturnInst>(terminatorInst);
      }
    }
    return finalReturnInst;
  }

  llvm::PreservedAnalyses run(llvm::Function &func, llvm::FunctionAnalysisManager &) {
    llvm::LLVMContext &context = func.getContext();
    llvm::IRBuilder<> irBuilder(context);
    llvm::Module *module = func.getParent();

    llvm::FunctionType *instrumentFuncType = llvm::FunctionType::get(llvm::Type::getVoidTy(context), false);
    llvm::FunctionCallee beginFunc = module->getOrInsertFunction("instrument_start", instrumentFuncType);
    llvm::FunctionCallee finishFunc = module->getOrInsertFunction("instrument_end", instrumentFuncType);

    bool beginInserted = hasInstrumentCalls(func, beginFunc);
    bool finishInserted = hasInstrumentCalls(func, finishFunc);

    addInstrumentCall(func, beginFunc, irBuilder, beginInserted);

    if (!finishInserted) {
      llvm::ReturnInst *finalReturnInst = findFinalReturnInst(func);

      if (finalReturnInst) {
        irBuilder.SetInsertPoint(finalReturnInst);
        irBuilder.CreateCall(finishFunc);
        finishInserted = true;
      }
    }

    return llvm::PreservedAnalyses::all();
  }

  static bool isRequired() { return true; }
};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "instrument_functions", "0.1",
          [](llvm::PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](llvm::StringRef name, llvm::FunctionPassManager &FPM,
                   llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) -> bool {
                  if (name == "instrument-functions") {
                    FPM.addPass(FuncInstrumentationPass{});
                    return true;
                  }
                  return false;
                });
          }};
}
