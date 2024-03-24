#include "llvm/IR/PassManager.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
struct InstrumentPass : public PassInfoMixin<InstrumentPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    for (Function &F : M) {
      instrumentFunction(F);
    }
    return PreservedAnalyses::all();
  }

private:
  void instrumentFunction(Function &F) {
    LLVMContext &Ctx = F.getContext();
    IRBuilder<> Builder(Ctx);

    // Create the function prototype for instrument_start and instrument_end
    FunctionType *InstrumentFnType = FunctionType::get(Type::getVoidTy(Ctx), {}, false);
    FunctionCallee InstrumentStartFn = F.getParent()->getOrInsertFunction("instrument_start", InstrumentFnType);
    FunctionCallee InstrumentEndFn = F.getParent()->getOrInsertFunction("instrument_end", InstrumentFnType);

    // Insert instrument_start() at the beginning of the function
    BasicBlock &EntryBlock = F.getEntryBlock();
    Instruction *FirstInstruction = &(*EntryBlock.getFirstInsertionPt());
    Builder.SetInsertPoint(FirstInstruction);
    Builder.CreateCall(InstrumentStartFn);

    // Insert instrument_end() at the end of the function
    BasicBlock &ExitBlock = F.back();
    Builder.SetInsertPoint(&(*ExitBlock.getTerminator()));
    Builder.CreateCall(InstrumentEndFn);
  }
};
}  // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "InstrumentPass", "1.0",
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "instrument-functions") {
                    FPM.addPass(InstrumentPass());
                    return true;
                  }
                  return false;
                });
          }};
}
