#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

struct Bend : PassInfoMixin<Bend> {

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) {
    FunctionCallee instrument_start = F.getParent()->getOrInsertFunction(
        "instrument_start", Type::getVoidTy(F.getContext()));
    FunctionCallee instrument_end = F.getParent()->getOrInsertFunction(
        "instrument_end", Type::getVoidTy(F.getContext()));

    bool startInserted = false;
    bool endInserted = false;

    for (auto &Block : F) {
      for (auto &instruction : Block) {
        if (llvm::isa<llvm::CallInst>(&instruction)) {
          llvm::CallInst *callInst = llvm::cast<llvm::CallInst>(&instruction);
          if (callInst->getCalledFunction() == instrument_start.getCallee()) {
            startInserted = true;
          } else if (callInst->getCalledFunction() ==
                     instrument_end.getCallee()) {
            endInserted = true;
          }
        }
      }
    }

    llvm::LLVMContext &Context = F.getContext();
    IRBuilder<> Builder(Context);
    if (!startInserted) {
      Builder.SetInsertPoint(&F.getEntryBlock().front());
      Builder.CreateCall(instrument_start);
    }

    if (!endInserted) {
      for (auto &Block : F) {
        if (llvm::isa<llvm::ReturnInst>(Block.getTerminator())) {
          Builder.SetInsertPoint(Block.getTerminator());
          Builder.CreateCall(instrument_end);
        }
      }
    }
    return PreservedAnalyses::all();
  }
};

} // namespace

/* New PM Registration */
llvm::PassPluginLibraryInfo getBendPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "Bend", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, llvm::FunctionPassManager &PM,
                   ArrayRef<llvm::PassBuilder::PipelineElement>) {
                  if (Name == "MrrrBend") {
                    PM.addPass(Bend());
                    return true;
                  }
                  return false;
                });
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getBendPluginInfo();
}