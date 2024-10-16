#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <optional>

class BitwiseShift : public llvm::PassInfoMixin<BitwiseShift> {
public:
  llvm::PreservedAnalyses run(llvm::Function &func,
                              llvm::FunctionAnalysisManager &FAM) {
    bool changed = false;
    for (auto &basicBlock : func) {
      for (auto InstIt = basicBlock.begin(), end = basicBlock.end();
           InstIt != end; ++InstIt) {
        if (InstIt->getOpcode() != llvm::Instruction::Mul)
          continue;
        auto *Op = llvm::dyn_cast<llvm::BinaryOperator>(InstIt);
        if (!Op)
          continue;

        llvm::Value *lhs = Op->getOperand(0);
        llvm::Value *rhs = Op->getOperand(1);
        auto lLog = getLog2(lhs);
        auto rLog = getLog2(rhs);

        if (rLog < lLog) {
          std::swap(lLog, rLog);
          std::swap(lhs, rhs);
        }

        if (rLog >= 0) {
          llvm::IRBuilder<> Builder(Op);
          llvm::Value *NewVal = Builder.CreateShl(
              lhs, llvm::ConstantInt::get(Op->getType(), *rLog));
          InstIt->replaceAllUsesWith(NewVal);
          InstIt = InstIt->eraseFromParent();
          changed = true;
        }
      }
    }
    return changed ? llvm::PreservedAnalyses::none()
                   : llvm::PreservedAnalyses::all();
  }

private:
  std::optional<int> getLog2(llvm::Value *Op) {
    if (auto *CI = llvm::dyn_cast<llvm::ConstantInt>(Op)) {
      return CI->getValue().exactLogBase2();
    }
    return std::nullopt;
  }
};

bool registerPipeLine(llvm::StringRef Name, llvm::FunctionPassManager &FPM,
                      llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
  if (Name == "kanakov-bitwise-shift") {
    FPM.addPass(BitwiseShift());
    return true;
  }
  return false;
}

llvm::PassPluginLibraryInfo getBitwiseShiftPluginPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "kanakov-bitwise-shift", LLVM_VERSION_STRING,
          [](llvm::PassBuilder &PB) {
            PB.registerPipelineParsingCallback(registerPipeLine);
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getBitwiseShiftPluginPluginInfo();
}