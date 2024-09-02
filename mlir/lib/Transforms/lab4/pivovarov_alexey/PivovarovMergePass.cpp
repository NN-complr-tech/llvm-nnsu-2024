#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

namespace {
class PivovarovMergePass
    : public PassWrapper<PivovarovMergePass, OperationPass<LLVM::LLVMFuncOp>> {
private:
  void replaceAddWithFMA(LLVM::FAddOp &addOp, LLVM::FMulOp &mulOp,
                         Value &otherOperand) {
    OpBuilder builder(addOp);
    Value fmaValue = builder.create<LLVM::FMAOp>(
        addOp.getLoc(), mulOp.getOperand(0), mulOp.getOperand(1), otherOperand);
    addOp.replaceAllUsesWith(fmaValue);
    addOp.erase();
  }

  void replaceDoubleMulAdd(LLVM::FAddOp &addOp, LLVM::FMulOp &mulOp) {
    OpBuilder builder(addOp);
    Value fmaValue = builder.create<LLVM::FMAOp>(
        addOp.getLoc(), mulOp.getOperand(0), mulOp.getOperand(1), mulOp);
    addOp.replaceAllUsesWith(fmaValue);
    addOp.erase();
  }

public:
  void runOnOperation() override {
    LLVM::LLVMFuncOp function = getOperation();
    function.walk([this](LLVM::FAddOp addOp) {
      Value addLeftOperand = addOp.getOperand(0);
      Value addRightOperand = addOp.getOperand(1);
      if (auto mulLeftOperand = addLeftOperand.getDefiningOp<LLVM::FMulOp>()) {
        if (addRightOperand == addLeftOperand) {
          replaceDoubleMulAdd(addOp, mulLeftOperand);
        } else {
          replaceAddWithFMA(addOp, mulLeftOperand, addRightOperand);
        }
      } else if (auto mulRightOperand =
                     addRightOperand.getDefiningOp<LLVM::FMulOp>()) {
        if (addLeftOperand == addRightOperand) {
          replaceDoubleMulAdd(addOp, mulRightOperand);
        } else {
          replaceAddWithFMA(addOp, mulRightOperand, addLeftOperand);
        }
      }
    });
    function.walk([](LLVM::FMulOp mulOp) {
      if (mulOp.use_empty()) {
        mulOp.erase();
      }
    });
  }

  StringRef getArgument() const final { return "pivovarov-combine-mul-add"; }
  StringRef getDescription() const final {
    return "Replaces add and multiply operations with a single instruction.";
  }
};
} // namespace

MLIR_DECLARE_EXPLICIT_TYPE_ID(PivovarovMergePass)
MLIR_DEFINE_EXPLICIT_TYPE_ID(PivovarovMergePass)

PassPluginLibraryInfo getPivovarovMergePassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "pivovarov-combine-mul-add",
          LLVM_VERSION_STRING,
          []() { PassRegistration<PivovarovMergePass>(); }};
}

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return getPivovarovMergePassPluginInfo();
}
