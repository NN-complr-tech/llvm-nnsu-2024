#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

using namespace mlir;

namespace {
class ZawadaMaxDepth
    : public PassWrapper<ZawadaMaxDepth, OperationPass<func::FuncOp>> {
public:
  StringRef getArgument() const final { return "ZawadaMaxDepth"; }
  StringRef getDescription() const final {
    return "Calculates the maximum nesting depth of regions in functions.";
  }

  void runOnOperation() override {
    getOperation()->walk([&](Operation *operation) {
      operation->setAttr(
          "maxDepth",
          IntegerAttr::get(IntegerType::get(operation->getContext(), 32),
                           getMaxDepth(operation)));
    });
  }

private:
  int getMaxDepth(Operation *operation) {
    int maxDepth = 1;

    operation->walk([&](Operation *op) {
      int depth = -1;

      while (op) {
        if (op->getParentOp())
          depth++;
        op = op->getParentOp();
      }

      if (depth > maxDepth)
        maxDepth = depth;
    });

    return maxDepth;
  }
};
} // anonymous namespace

MLIR_DECLARE_EXPLICIT_TYPE_ID(ZawadaMaxDepth)
MLIR_DEFINE_EXPLICIT_TYPE_ID(ZawadaMaxDepth)

PassPluginLibraryInfo getMaxDepthPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "ZawadaMaxDepth", LLVM_VERSION_STRING,
          []() { PassRegistration<ZawadaMaxDepth>(); }};
}

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return getMaxDepthPassPluginInfo();
}