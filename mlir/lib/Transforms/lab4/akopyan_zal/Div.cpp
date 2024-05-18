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
    replaceCeilDiv(op, arith::DivUIOp::getOperationName());
  }

  void replaceCeilDivSI(arith::CeilDivSIOp op) {
    replaceCeilDiv(op, arith::DivSIOp::getOperationName());
  }

  template <typename DivOp>
  void replaceCeilDiv(Operation *op, StringRef divOpName) {
    OpBuilder builder(op);
    Location loc = op->getLoc();
    Value a = op->getOperand(0);
    Value b = op->getOperand(1);
    Type type = a.getType();

    Value one =
        builder.create<arith::ConstantOp>(loc, builder.getIntegerAttr(type, 1));
    Value add = builder.create<arith::AddIOp>(loc, a, b);
    Value sub = builder.create<arith::SubIOp>(loc, add, one);
    Value div = builder.create<arith::DivOp>(loc, type, sub, b);

    op->replaceAllUsesWith(div);
    op->erase();
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