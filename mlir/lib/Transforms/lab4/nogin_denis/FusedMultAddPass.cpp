#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

using namespace mlir;

namespace {
class FusedMultAddPass
    : public PassWrapper<FusedMultAddPass, OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const final { return "fused-mult-add"; }
  StringRef getDescription() const final {
    return "Fuses multiply and add operations into a single fma operation";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](Operation *op) {
      if (auto addOp = dyn_cast<LLVM::FAddOp>(op)) {
        Value addLHS = addOp.getOperand(0);
        Value addRHS = addOp.getOperand(1);

        if (auto mulOpLHS = addLHS.getDefiningOp<LLVM::FMulOp>()) {
          OpBuilder builder(addOp);
          Value fma = builder.create<LLVM::FMAOp>(
              addOp.getLoc(), mulOpLHS.getOperand(0), mulOpLHS.getOperand(1),
              addRHS);
          addOp.replaceAllUsesWith(fma);
          addOp.erase();
          mulOpLHS.erase();
        } else if (auto mulOpRHS = addRHS.getDefiningOp<LLVM::FMulOp>()) {
          OpBuilder builder(addOp);
          Value fma = builder.create<LLVM::FMAOp>(
              addOp.getLoc(), mulOpRHS.getOperand(0), mulOpRHS.getOperand(1),
              addLHS);
          addOp.replaceAllUsesWith(fma);
          addOp.erase();
          mulOpRHS.erase();
        }
      }
    });
  }
};
} // namespace

MLIR_DECLARE_EXPLICIT_TYPE_ID(FusedMultAddPass)
MLIR_DEFINE_EXPLICIT_TYPE_ID(FusedMultAddPass)

PassPluginLibraryInfo getFusedMultAddPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "fused-mult-add", LLVM_VERSION_STRING,
          []() { PassRegistration<FusedMultAddPass>(); }};
}

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return getFusedMultAddPassPluginInfo();
}