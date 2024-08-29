#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

namespace {
    class PivovarovMergePass
        : public PassWrapper<PivovarovMergePass,
        OperationPass<LLVM::LLVMFuncOp>> {
    private:
        void combineMulAdd(LLVM::FAddOp addOp, LLVM::FMulOp mulOp, Value otherOperand) {
            OpBuilder builder(addOp);
            Value fmaValue = builder.create<LLVM::FMAOp>(
                addOp.getLoc(), mulOp.getOperand(0), mulOp.getOperand(1), otherOperand);
            addOp.replaceAllUsesWith(fmaValue);
            addOp.erase();
        }

    public:
        void runOnOperation() override {
            LLVM::LLVMFuncOp function = getOperation();

            function.walk([this](LLVM::FAddOp addOp) {
                Value lhs = addOp.getOperand(0);
                Value rhs = addOp.getOperand(1);

                if (auto mulOp = lhs.getDefiningOp<LLVM::FMulOp>()) {
                    combineMulAdd(addOp, mulOp, rhs);
                }
                else if (auto mulOp = rhs.getDefiningOp<LLVM::FMulOp>()) {
                    combineMulAdd(addOp, mulOp, lhs);
                }
                });

            function.walk([](LLVM::FMulOp mulOp) {
                if (mulOp.use_empty()) {
                    mulOp.erase();
                }
                });
        }

        StringRef getArgument() const final { return "pivovarov-combine-mul-add"; }
        StringRef getDescription() const final {
            return "Combines multiplication and addition into a single FMA operation.";
        }
    };

MLIR_DECLARE_EXPLICIT_TYPE_ID(PivovarovMergePass)
MLIR_DEFINE_EXPLICIT_TYPE_ID(PivovarovMergePass)

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo
mlirGetPassPluginInfo() {
    return { MLIR_PLUGIN_API_VERSION, "pivovarov-combine-mul-add",
            LLVM_VERSION_STRING,
            []() { PassRegistration<PivovarovMergePass>(); } };
}

