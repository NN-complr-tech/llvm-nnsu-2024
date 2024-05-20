#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

using namespace mlir;

class LebedevaCallFuncCounter
    : public PassWrapper<LebedevaCallFuncCounter, OperationPass<ModuleOp>> {
private:
  std::map<StringRef, int> сounter;

public:
  void runOnOperation() override {
    getOperation()->walk([&](Operation *op) {
      if (auto callOp = dyn_cast<LLVM::CallOp>(op)) {
        StringRef functionName = callOp.getCallee().value();
        сounter[functionName]++;
      }
    });
    getOperation()->walk([&](Operation *op) {
      if (auto functionOp = dyn_cast<LLVM::LLVMFuncOp>(op)) {
        StringRef functionName = functionOp.getName();
        int nummerCalls = сounter[functionName];
        auto attrValue = IntegerAttr::get(
            IntegerType::get(functionOp.getContext(), 32), nummerCalls);
        functionOp->setAttr("call-count", attrValue);
      }
    });
  }
  StringRef getArgument() const final { return "lebedeva-call-func-counter"; }
  StringRef getDescription() const final {
    return "Simple pass that counts the number of calls to each function";
  }
};

MLIR_DECLARE_EXPLICIT_TYPE_ID(LebedevaCallFuncCounter)
MLIR_DEFINE_EXPLICIT_TYPE_ID(LebedevaCallFuncCounter)

PassPluginLibraryInfo getLebedevaCallFuncCounterPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "lebedeva-call-func-counter",
          LLVM_VERSION_STRING,
          []() { PassRegistration<LebedevaCallFuncCounter>(); }};
}

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return getLebedevaCallFuncCounterPluginInfo();
}
