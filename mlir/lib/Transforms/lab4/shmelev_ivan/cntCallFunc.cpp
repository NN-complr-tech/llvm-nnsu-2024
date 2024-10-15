#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

using namespace mlir;

namespace {
class CntCallFuncShmelevPass
    : public PassWrapper<CntCallFuncShmelevPass, OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const final { return "cntCallFuncShmelev"; }
  void runOnOperation() override {
    std::vector<LLVM::LLVMFuncOp> functions;
    std::map<StringRef, int> funcCallTally;

    getOperation()->walk([&](Operation *object_operation) {
      auto funcCheck = dyn_cast<LLVM::LLVMFuncOp>(object_operation);
      if (funcCheck) {
        functions.push_back(funcCheck);
      }
      auto callCheck = dyn_cast<LLVM::CallOp>(object_operation);
      if (callCheck) {
        funcCallTally[callCheck.getCallee().value()]++;
      }
    });

    for (auto &func : functions) {
      auto count = funcCallTally[func.getName()];
      auto attrValue =
          IntegerAttr::get(IntegerType::get(func.getContext(), 32), count);
      func->setAttr("cnt_call", attrValue);
    }
  }
};
} // namespace

MLIR_DECLARE_EXPLICIT_TYPE_ID(CntCallFuncShmelevPass)
MLIR_DEFINE_EXPLICIT_TYPE_ID(CntCallFuncShmelevPass)

PassPluginLibraryInfo getFuncCallCounterPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "cntCallFuncShmelev", LLVM_VERSION_STRING,
          []() { PassRegistration<CntCallFuncShmelevPass>(); }};
}

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return getFuncCallCounterPassPluginInfo();
}
