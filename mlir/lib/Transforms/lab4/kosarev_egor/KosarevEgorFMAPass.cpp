#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
class KosarevEgorFMAPass
    : public PassWrapper<KosarevEgorFMAPass, OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const final { return "KosarevEgorFMAPass"; }
  StringRef getDescription() const final { return "fma pass"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::math::MathDialect>();
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::OpBuilder builder(module);

    auto replaceAndEraseOp = [&](mlir::LLVM::FMulOp &mulOp,
                                 mlir::LLVM::FAddOp &addOp,
                                 mlir::Value &thirdOperand) -> void {
      builder.setInsertionPoint(addOp);
      mlir::Value fmaResult =
          builder.create<mlir::math::FmaOp>(addOp.getLoc(), mulOp.getOperand(0),
                                            mulOp.getOperand(1), thirdOperand);
      addOp.replaceAllUsesWith(fmaResult);
      addOp.erase();
      mulOp.erase();
    };

    module.walk([](LLVM::FAddOp addOp) {
      Value addLHS = addOp.getOperand(0);
      Value addRHS = addOp.getOperand(1);

      auto tryFuse = [&](Value mulOperand, Value otherOperand) {
        if (auto mulOp = mulOperand.getDefiningOp<LLVM::FMulOp>()) {
          OpBuilder builder(addOp);
          Value fma =
              builder.create<LLVM::FMAOp>(addOp.getLoc(), mulOp.getOperand(0),
                                          mulOp.getOperand(1), otherOperand);
          addOp.replaceAllUsesWith(fma);
          return true;
        }
        return false;
      };

      if (tryFuse(addLHS, addRHS) || tryFuse(addRHS, addLHS)) {
        addOp.erase();
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
  return getFunctionCallCounterPassPluginInfo();
}
