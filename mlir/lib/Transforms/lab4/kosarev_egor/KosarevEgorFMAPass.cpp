#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
class KosarevEgorFMAPass
    : public PassWrapper<KosarevEgorFMAPass, OperationPass<FuncOp>> {
public:
  StringRef getArgument() const final { return "KosarevEgorFMAPass"; }
  StringRef getDescription() const final { return "fma pass"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::math::MathDialect>();
  }

  void runOnOperation() override {
    FuncOp function = getOperation();
    OpBuilder builder(function);

    auto replaceAndEraseOp = [&](LLVM::FMulOp &mulOp, LLVM::FAddOp &addOp,
                                 Value &thirdOperand) {
      builder.setInsertionPoint(addOp);
      Value fmaResult =
          builder.create<math::FmaOp>(addOp.getLoc(), mulOp.getOperand(0),
                                      mulOp.getOperand(1), thirdOperand);
      addOp.replaceAllUsesWith(fmaResult);
      addOp.erase();
      mulOp.erase();
    };

    function.walk([&](LLVM::FAddOp addOp) {
      Value addLhs = addOp.getOperand(0);
      Value addRhs = addOp.getOperand(1);

      auto isSingleUse = [&](Value value, Operation *userOp) {
        for (auto &use : value.getUses()) {
          if (use.getOwner() != userOp) {
            return false;
          }
        }
        return true;
      };

      if (auto mulOp = addLhs.getDefiningOp<LLVM::FMulOp>()) {
        if (isSingleUse(mulOp->getResult(0), addOp)) {
          replaceAndEraseOp(mulOp, addOp, addRhs);
        }
      } else if (auto mulOp = addRhs.getDefiningOp<LLVM::FMulOp>()) {
        if (isSingleUse(mulOp->getResult(0), addOp)) {
          replaceAndEraseOp(mulOp, addOp, addLhs);
        }
      }
    });
  }
};
} // namespace

MLIR_DECLARE_EXPLICIT_TYPE_ID(KosarevEgorFMAPass)
MLIR_DEFINE_EXPLICIT_TYPE_ID(KosarevEgorFMAPass)

mlir::PassPluginLibraryInfo getFunctionCallCounterPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "KosarevEgorFMAPass", "0.1",
          []() { mlir::PassRegistration<KosarevEgorFMAPass>(); }};
}

extern "C" LLVM_ATTRIBUTE_WEAK mlir::PassPluginLibraryInfo
mlirGetPassPluginInfo() {
  return getFunctionCallCounterPassPluginInfo();FunctionCallCounterPassPluginInfo();
}
