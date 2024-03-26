#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

namespace {
struct InstrumentFunctions : llvm::PassInfoMixin<InstrumentFunctions> {
  llvm::PreservedAnalyses run(llvm::Function &func,
                              llvm::FunctionAnalysisManager &) {
    llvm::LLVMContext &context = func.getContext();
    llvm::IRBuilder<> builder(context);
    auto module = func.getParent();

    llvm::FunctionType *funcType =
        llvm::FunctionType::get(llvm::Type::getVoidTy(context), false);
    llvm::FunctionCallee instrStartFunc =
        module->getOrInsertFunction("instrument_start", funcType);
    llvm::FunctionCallee instEndFunc =
        module->getOrInsertFunction("instrument_end", funcType);

    builder.SetInsertPoint(&func.getEntryBlock().front());
    builder.CreateCall(instrStartFunc);

    for (auto &block : func) {
      if (llvm::isa<llvm::ReturnInst>(block.getTerminator())) {
        builder.SetInsertPoint(block.getTerminator());
        builder.CreateCall(instEndFunc);
      }
    }

    return llvm::PreservedAnalyses::all();
  }

  static bool isRequired() { return true; }
};
} // namespace

#define CONCATENATE_IMPL(A, B, C) A##B##C
#define CONCATENATE(A, B, C) CONCATENATE_IMPL(A, B, C)
#define GET_PLUGIN_INFO CONCATENATE(get, NAME, PluginInfo)

llvm::PassPluginLibraryInfo GET_PLUGIN_INFO() {
  return {LLVM_PLUGIN_API_VERSION, "InstrumentFunctions", "0.1",
          [](llvm::PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](llvm::StringRef Name, llvm::FunctionPassManager &PM,
                   llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
                  if (Name == "instr_func") {
                    PM.addPass(InstrumentFunctions());
                    return true;
                  }
                  return false;
                });
          }};
}

#define LLVM_PLUGIN_LINK_INTO_TOOLS_ON                                         \
  CONCATENATE(LLVM_, NAME, _LINK_INTO_TOOLS)

#ifdef LLVM_PLUGIN_LINK_INTO_TOOLS_ON
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return GET_PLUGIN_INFO();
}
#endif // LLVM_PLUGIN_LINK_INTO_TOOLS_ON
