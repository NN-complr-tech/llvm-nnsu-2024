#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

using namespace mlir;

namespace {
class MirzakhmedovPassCountFCalls
    : public PassWrapper<MirzakhmedovPassCountFCalls, OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const final { return "MirzakhmedovCountFCalls"; }
  StringRef getDescription() const final {
    return "A pass that tallies the number of calls to each function.";
  }

  void runOnOperation() override {
    llvm::DenseMap<StringRef, int> cnt;

    getOperation()->walk([&](LLVM::CallOp callOper) {
      if (auto callee = callOper.getCallee()) {
        cnt[callee.value()]++;
      }
    });

    getOperation()->walk([&](LLVM::LLVMFuncOp funcOp) {
      StringRef funcName = funcOp.getName();
      int callsNum = cnt.lookup(funcName);
      funcOp->setAttr("call-count",
                      IntegerAttr::get(
                          IntegerType::get(funcOp.getContext(), 32), callsNum));
    });
  }
};
} // namespace

MLIR_DECLARE_EXPLICIT_TYPE_ID(MirzakhmedovPassCountFCalls)
MLIR_DEFINE_EXPLICIT_TYPE_ID(MirzakhmedovPassCountFCalls)

PassPluginLibraryInfo getMirzakhmedovCountFCallsPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "MirzakhmedovCountFCalls",
          LLVM_VERSION_STRING,
          []() { PassRegistration<MirzakhmedovPassCountFCalls>(); }};
}

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return getMirzakhmedovCountFCallsPluginInfo();
}