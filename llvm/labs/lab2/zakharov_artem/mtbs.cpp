#include "mtbs.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

PreservedAnalyses MTBSPass::run(llvm::Function &F,
                                llvm::FunctionAnalysisManager &AM) {
  for (auto &BB : F) {
    for (auto InstIt = BB.begin(); InstIt != BB.end(); InstIt++) {
      if (InstIt->getOpcode() != Instruction::Mul) {
        continue;
      }

      auto *Op = dyn_cast<BinaryOperator>(InstIt);
      Value *LVal = Op->getOperand(0);
      Value *RVal = Op->getOperand(1);

      int LLog = getLog2(LVal);
      int RLog = getLog2(RVal);

      if (RLog < LLog) {
        std::swap(LLog, RLog);
        std::swap(LVal, RVal);
      }
      if (RLog < 0) {
        continue;
      }

      IRBuilder<> Builder(Op);
      Value *NewVal =
          Builder.CreateShl(LVal, ConstantInt::get(Op->getType(), RLog));
      ReplaceInstWithValue(InstIt, NewVal);
    }
  }
  return PreservedAnalyses::all();
}

int MTBSPass::getLog2(llvm::Value *Op) {
  if (auto *CI = dyn_cast<ConstantInt>(Op)) {
    return CI->getValue().exactLogBase2();
  }
  return -1;
}

bool registerPipeLine(llvm::StringRef Name, llvm::FunctionPassManager &FPM,
                      llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
  if (Name == "ZakharovMTBSPlugin") {
    FPM.addPass(MTBSPass());
    return true;
  }
  return false;
}

PassPluginLibraryInfo getMTBSPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "ZakharovMTBSPlugin", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(registerPipeLine);
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return getMTBSPluginInfo();
}
