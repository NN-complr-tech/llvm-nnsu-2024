#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

using namespace mlir;

namespace {
class sCeilDivPass : public PassWrapper<sCeilDivPass, OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const final { return "sharapov_ceildiv"; }
  StringRef getDescription() const final {
    return "breaks arith.ceildivui and arith.ceildivsi operations into arith. "
           "calculations";
  }

  void runOnOperation() override {
    getOperation()->walk([&](Operation *op) {
      if (auto ceilDivUI = dyn_cast<arith::CeilDivUIOp>(op)) {
        replaceCeilDivUI(ceilDivUI);
      } else if (auto ceilDivSI = dyn_cast<arith::CeilDivSIOp>(op)) {
        replaceCeilDivSI(ceilDivSI);
      }
    });
  }

private:
  void replaceCeilDivUI(arith::CeilDivUIOp op) {
    OpBuilder builder(op);
    Location loc = op.getLoc();
    Value a = op.getLhs();
    Value b = op.getRhs();

    Value one =
        builder.create<arith::ConstantIntOp>(loc, 1, builder.getI32Type());
    Value add = builder.create<arith::AddIOp>(loc, a, b);
    Value sub = builder.create<arith::SubIOp>(loc, add, one);
    Value div = builder.create<arith::DivUIOp>(loc, sub, b);

    op.replaceAllUsesWith(div);
    op.erase();
  }

  void replaceCeilDivSI(arith::CeilDivSIOp op) {
    OpBuilder builder(op);
    Location loc = op.getLoc();
    Value a = op.getLhs();
    Value b = op.getRhs();

    Value one =
        builder.create<arith::ConstantIntOp>(loc, 1, builder.getI32Type());
    Value add = builder.create<arith::AddIOp>(loc, a, b);
    Value sub = builder.create<arith::SubIOp>(loc, add, one);
    Value div = builder.create<arith::DivSIOp>(loc, sub, b);

    op.replaceAllUsesWith(div);
    op.erase();
  }
};
} // anonymous namespace

MLIR_DECLARE_EXPLICIT_TYPE_ID(sCeilDivPass)
MLIR_DEFINE_EXPLICIT_TYPE_ID(sCeilDivPass)

PassPluginLibraryInfo getsCeilDivPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "sharapov_ceildiv", LLVM_VERSION_STRING,
          []() { PassRegistration<sCeilDivPass>(); }};
}

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return getsCeilDivPassPluginInfo();
}
