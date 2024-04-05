#include "llvm/IR/PassManager.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Function.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"

using namespace llvm;

namespace {

// Плагин для инлайнинга вызовов функций без аргументов и возвращаемых значений
struct SimonyanInliningPass : public PassInfoMixin<SimonyanInliningPass> {
  PreservedAnalyses run(Function &F) {
    SmallPtrSet<Instruction *, 8> CallsToRemove;

    // Перебираем все базовые блоки в функции
    for (BasicBlock &BB : F) {
      // Перебираем все инструкции в базовом блоке
      for (Instruction &Instr : BB) {
        // Если инструкция - вызов функции без аргументов и возвращаемого значения
        if (CallInst *CI = dyn_cast<CallInst>(&Instr)) {
          Function *Callee = CI->getCalledFunction();
          if (Callee && Callee->arg_empty() && Callee->getReturnType()->isVoidTy()) {
            // Клонируем тело вызываемой функции в месте вызова
            BasicBlock &EntryBlock = Callee->getEntryBlock();
            IRBuilder<> Builder(CI);
            for (Instruction &Inst : EntryBlock) {
              Builder.Insert(Inst.clone());
            }

            // Помечаем вызов для удаления
            CallsToRemove.insert(CI);
          }
        }
      }
    }

    // Удаляем вызовы функций
    for (Instruction *I : CallsToRemove) {
      I->eraseFromParent();
    }

    // Возвращаем, что анализы не изменены
    return PreservedAnalyses::none();
  }
};

} // end anonymous namespace

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK llvmGetPassPluginInfo() {
  return {
    LLVM_PLUGIN_API_VERSION, "SimonyanInliningPass", "v0.1",
    [](PassBuilder &PB) {
      PB.registerPipelineParsingCallback(
        [](StringRef Name, FunctionPassManager &FPM,
           ArrayRef<PassBuilder::PipelineElement>) {
          if (Name == "custom-inlining") {
            FPM.addPass(SimonyanInliningPass());
            return true;
          }
          return false;
        }
      );
    }
  };
}