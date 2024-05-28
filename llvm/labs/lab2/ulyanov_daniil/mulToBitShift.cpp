#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <optional>

using namespace llvm;

namespace {
class MTBSPass : public PassInfoMixin<MTBSPass> {
public:
  PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &AM) {
    for (auto &BB : F) {
      for (auto InstIt = BB.begin(); InstIt != BB.end(); InstIt++) {
        if (InstIt->getOpcode() != Instruction::Mul) {
          continue;
        }

        auto *Op = dyn_cast<BinaryOperator>(InstIt);
        Value *LVal = Op->getOperand(0);
        Value *RVal = Op->getOperand(1);

        std::optional<int> LLog = getLog2(LVal);
        std::optional<int> RLog = getLog2(RVal);

        if (!LLog.has_value() || !RLog.has_value()) {
          continue;
        }
        if (LLog.has_value() && !RLog.has_value()) {
          std::swap(LLog, RLog);
          std::swap(LVal, RVal);
        } else if (LLog.has_value() && RLog.has_value()) {
          if (RLog.value() < LLog.value()) {
            std::swap(LLog, RLog);
            std::swap(LVal, RVal);
          }
        }

        IRBuilder<> Builder(Op);
        Value *NewVal;
        if (RVal->getType()->isIntegerTy(8) && RLog.value() > 8) {
          NewVal = Builder.CreateShl(LVal, ConstantInt::get(Op->getType(), 8));
        } else if(RVal->getType()->isIntegerTy(16) && RLog.value() > 16) {
          NewVal = Builder.CreateShl(LVal, ConstantInt::get(Op->getType(), 16));
        } else if(RVal->getType()->isIntegerTy(32) && RLog.value() > 32) {
          NewVal = Builder.CreateShl(LVal, ConstantInt::get(Op->getType(), 32));
        } else {
          NewVal =
            Builder.CreateShl(LVal, ConstantInt::get(Op->getType(), RLog.value()));
        }
        ReplaceInstWithValue(InstIt, NewVal);
      }
    }
    return PreservedAnalyses::all();
  }

  std::optional<int> getLog2(llvm::Value *Op) {
    if (auto *CI = dyn_cast<ConstantInt>(Op)) {
      return CI->getValue().exactLogBase2();
    }
    return std::nullopt;
  }
};
} // namespace

bool registerPipeLine(llvm::StringRef Name, llvm::FunctionPassManager &FPM,
                      llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
  if (Name == "ulyanovMulToBitShiftPlugin") {
    FPM.addPass(MTBSPass());
    return true;
  }
  return false;
}

PassPluginLibraryInfo getUlyanovMulToBitShiftPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "ulyanovMulToBitShiftPlugin",
          LLVM_VERSION_STRING, [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(registerPipeLine);
          }};
}

#ifndef LLVM_ULYANOVMULTOBITSHIFTPLUGIN_LINK_INTO_TOOLS
extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return getUlyanovMulToBitShiftPluginInfo();
}
#endif