#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

using namespace mlir;

class KanakovCountFuncCalls
    : public PassWrapper<KanakovCountFuncCalls, OperationPass<ModuleOp>> {
private:
  std::map<StringRef, int> counter;

public:
  void runOnOperation() override {
    getOperation()->walk([&](LLVM::CallOp callOp) {
      StringRef functionName = callOp.getCallee().value();
      counter[functionName]++;
    });

    getOperation()->walk([&](LLVM::LLVMFuncOp functionOp) {
      StringRef functionName = functionOp.getName();
      int numberCalls = counter[functionName];
      auto attrValue = IntegerAttr::get(
          IntegerType::get(functionOp.getContext(), 32), numberCalls);
      functionOp->setAttr("call-count", attrValue);
    });
  }

  StringRef getArgument() const final { return "kanakov-call-func-counter"; }
};

MLIR_DECLARE_EXPLICIT_TYPE_ID(KanakovCountFuncCalls)
MLIR_DEFINE_EXPLICIT_TYPE_ID(KanakovCountFuncCalls)

PassPluginLibraryInfo getKanakovCountFuncCallsPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "kanakov-call-func-counter",
          LLVM_VERSION_STRING,
          []() { PassRegistration<KanakovCountFuncCalls>(); }};
}

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return getKanakovCountFuncCallsPluginInfo();
}