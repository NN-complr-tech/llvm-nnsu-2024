#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

namespace {
struct InstrumentFunction : llvm::PassInfoMixin<InstrumentFunction> {
  llvm::PreservedAnalyses run(llvm::Function &function,
                              llvm::FunctionAnalysisManager &fam) {
    llvm::LLVMContext &con = function.getContext();
    llvm::IRBuilder<> build(con);
    llvm::Module *mod = function.getParent();
    llvm::FunctionType *type =
        llvm::FunctionType::get(llvm::Type::getVoidTy(con), false);
    llvm::FunctionCallee f_start =
        (*mod).getOrInsertFunction("instrument_start", type);
    llvm::FunctionCallee f_end =
        (*mod).getOrInsertFunction("instrument_end", type);
    bool is_start = false;
    bool is_end = false;
    llvm::Function *f_start_f =
        llvm::dyn_cast<llvm::Function>(f_start.getCallee());
    llvm::Function *f_end_f = llvm::dyn_cast<llvm::Function>(f_end.getCallee());
    for (auto *user : f_start_f->users()) {
      if (auto *ci = llvm::dyn_cast<llvm::CallInst>(user)) {
        if (ci->getFunction() == &function) {
          is_start = true;
        }
      }
    }
    for (auto *user : f_end_f->users()) {
      if (auto *ci = llvm::dyn_cast<llvm::CallInst>(user)) {
        if (ci->getFunction() == &function) {
          is_end = true;
        }
      }
    }
    if (!is_start) {
      build.SetInsertPoint(&function.getEntryBlock().front());
      build.CreateCall(f_start);
    }
    llvm::ReturnInst *retI;
    if (!is_end) {
      for (llvm::BasicBlock &bb : function) {
        if ((retI = llvm::dyn_cast<llvm::ReturnInst>(bb.getTerminator())) !=
            NULL) {
          build.SetInsertPoint(bb.getTerminator());
          build.CreateCall(f_end);
        }
      }
    }
    return llvm::PreservedAnalyses::all();
  }

  static bool require() { return true; }
};
} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPuginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "InstrumentFunction", "0.1",
          [](llvm::PassBuilder &pb) {
            pb.registerPipelineParsingCallback(
                [](llvm::StringRef name, llvm::FunctionPassManager &fpm,
                   llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) -> bool {
                  if (name == "instrument_function") {
                    fpm.addPass(InstrumentFunction{});
                    return true;
                  }
                  return false;
                });
          }};
}