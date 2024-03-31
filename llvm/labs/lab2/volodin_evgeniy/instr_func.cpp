#include "llvm/IR/PassManager.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

namespace {
class InstrFuncPass : public llvm::PassInfoMixin<InstrFuncPass> {
public:
    llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &AM) {
        llvm::LLVMContext &context = F.getContext();

        auto module = F.getParent();

        llvm::FunctionType *instrFuncType = llvm::FunctionType::get(llvm::Type::getVoidTy(context), false);
        llvm::FunctionCallee instrStartFunction = module->getOrInsertFunction("instrument_start", instrFuncType);
        llvm::FunctionCallee instrEndFunction = module->getOrInsertFunction("instrument_end", instrFuncType);

        llvm::Instruction *firstInstruction = &F.front().front();
        llvm::IRBuilder<> builder(firstInstruction);

        builder.CreateCall(instrStartFunction);

        for (auto& block: F) {
            if (llvm::isa<llvm::ReturnInst>(block.getTerminator())) {
                builder.SetInsertPoint(block.getTerminator());
                builder.CreateCall(instrEndFunction);
            }
        }

        return llvm::PreservedAnalyses::all();
    }

    static bool isRequired() {return true; }
};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
    return { LLVM_PLUGIN_API_VERSION, "InstrFuncVolodin", "0.1",
            [](llvm::PassBuilder &PB) {
                PB.registerPipelineParsingCallback(
                    [](llvm::StringRef name, llvm::FunctionPassManager &FPM,
                        llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) -> bool {
                            if (name == "instr-func-volodin") {
                                FPM.addPass(InstrFuncPass{});
                                return true;
                            }
                            return false;
                        });
            }};
}