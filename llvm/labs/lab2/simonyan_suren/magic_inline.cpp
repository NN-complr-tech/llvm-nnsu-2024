#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

namespace {

struct SimonyanInliningPass : public llvm::PassInfoMixin<SimonyanInliningPass> {
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &) {
    llvm::outs() << "Start plagin\n";
    llvm::outs() << "Start plagin\n";
    llvm::outs() << "Start plagin\n";
    llvm::outs() << "Start plagin\n";
    llvm::outs() << "Start plagin\n";

    llvm::SmallPtrSet<llvm::CallInst *, 8> CallsToRemove;

    for (llvm::BasicBlock &BB : F) {
      for (llvm::Instruction &Instr : BB) {
        if (llvm::CallInst *CI = llvm::dyn_cast<llvm::CallInst>(&Instr)) {
          llvm::Function *Callee = CI->getCalledFunction();
          if (Callee && Callee->arg_empty() &&
              Callee->getReturnType()->isVoidTy()) {
            std::map<llvm::Instruction *, llvm::Instruction *> InstructionMap;
            llvm::BasicBlock *InsertBlock = CI->getParent();

            for (llvm::BasicBlock &CalleeBB : *Callee) {
              for (llvm::Instruction &Inst : CalleeBB) {
                llvm::Instruction *NewInst = Inst.clone();
                if (!NewInst) {
                  llvm::errs() << "Error: Failed to clone instruction.\n";
                  return llvm::PreservedAnalyses::none();
                }
                NewInst->insertBefore(CI);
                InstructionMap[&Inst] = NewInst;
              }
            }

            for (auto &InstMapping : InstructionMap) {
              for (llvm::Use &Op : InstMapping.second->operands()) {
                if (llvm::Instruction *OpInst =
                        llvm::dyn_cast<llvm::Instruction>(Op)) {
                  if (InstructionMap.count(OpInst)) {
                    Op.set(InstructionMap[OpInst]);
                  }
                }
              }
            }

            CallsToRemove.insert(CI);
          }
        }
      }
    }

    for (llvm::CallInst *CI : CallsToRemove) {
      CI->eraseFromParent();
    }

    return llvm::PreservedAnalyses::all();
  }
};

} // end anonymous namespace

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "SimonyanInliningPass", "v0.1",
          [](llvm::PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](llvm::StringRef Name, llvm::FunctionPassManager &FPM,
                   llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
                  if (Name == "simonyan-inlining") {
                    FPM.addPass(SimonyanInliningPass());
                    return true;
                  }
                  return false;
                });
          }};
}