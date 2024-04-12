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
    llvm::SmallPtrSet<llvm::CallInst *, 8> CallsToRemove;
    llvm::ValueToValueMapTy ValueMap;

    for (llvm::BasicBlock &BB : F) {
      for (llvm::Instruction &Instr : BB) {
        if (llvm::CallInst *CI = llvm::dyn_cast<llvm::CallInst>(&Instr)) {
          llvm::Function *Callee = CI->getCalledFunction();
          if (Callee && Callee->arg_empty() &&
              Callee->getReturnType()->isVoidTy()) {
            llvm::ValueToValueMapTy ValueMap;

            for (llvm::BasicBlock &CalleeBB : *Callee) {
              for (llvm::Instruction &Inst : CalleeBB) {
                if (Inst.isTerminator())
                  continue; // Пропустить терминаторные инструкции
                llvm::Instruction *NewInst = Inst.clone();
                if (!NewInst) {
                  llvm::errs() << "Error: Failed to clone instruction.\n";
                  return llvm::PreservedAnalyses::none();
                }
                NewInst->insertBefore(CI);
                ValueMap[&Inst] = NewInst;
              }
            }

            for (auto it = ValueMap.begin(); it != ValueMap.end(); ++it) {
              llvm::Instruction *NewInst =
                  llvm::dyn_cast<llvm::Instruction>(it->second);
              for (llvm::Use &Op : NewInst->operands()) {
                if (ValueMap.count(Op)) {
                  Op.set(ValueMap[Op]);
                }
              }
            }

            // Обновляем операнды инструкций в CallInst
            for (auto &Use : CI->uses()) {
              llvm::User *User = Use.getUser();
              for (unsigned i = 0; i < User->getNumOperands(); ++i) {
                if (ValueMap.count(User->getOperand(i))) {
                  User->getOperand(i)->replaceAllUsesWith(
                      ValueMap[User->getOperand(i)]);
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

    return llvm::PreservedAnalyses::none();
  }
};

} // end anonymous namespace

llvm::PassPluginLibraryInfo getSimonyanInliningPassPluginInfo() {
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

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getSimonyanInliningPassPluginInfo();
}