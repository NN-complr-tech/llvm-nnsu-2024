#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

using namespace mlir;

namespace {
class DivPass : public PassWrapper<DivPass, OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const final { return "akopyan_divpass"; }
  StringRef getDescription() const final {
    return "splits the arith.ceildivui and arith.ceildivsi into arith "
		   "operations";
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

MLIR_DECLARE_EXPLICIT_TYPE_ID(DivPass)
MLIR_DEFINE_EXPLICIT_TYPE_ID(DivPass)

PassPluginLibraryInfo getsCeilDivPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "akopyan_divpass", LLVM_VERSION_STRING,
          []() { PassRegistration<DivPass>(); }};
}

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return getsCeilDivPassPluginInfo();
}