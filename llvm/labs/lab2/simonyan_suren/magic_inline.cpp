#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace llvm;

namespace {

struct SimonyanInliningPass : public PassInfoMixin<SimonyanInliningPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    SmallPtrSet<CallInst *, 8> CallsToRemove;

    // Перебираем все базовые блоки в функции
    for (BasicBlock &BB : F) {
      // Перебираем все инструкции в базовом блоке
      for (Instruction &Instr : BB) {
        // Если инструкция - вызов функции без аргументов и возвращаемого
        // значения
        if (CallInst *CI = dyn_cast<CallInst>(&Instr)) {
          Function *Callee = CI->getCalledFunction();
          if (Callee && Callee->arg_empty() &&
              Callee->getReturnType()->isVoidTy()) {
            // Клонируем тело вызываемой функции в месте вызова
            std::map<Instruction *, Instruction *> InstructionMap;
            BasicBlock *InsertBlock = CI->getParent();
            for (BasicBlock &CalleeBB : *Callee) {
              BasicBlock *NewBlock =
                  BasicBlock::Create(F.getContext(), CalleeBB.getName(), &F);
              for (Instruction &Inst : CalleeBB) {
                Instruction *NewInst = Inst.clone();
                if (!NewInst) {
                  errs() << "Ошибка: Клонирование инструкции не удалось.\n";
                  return PreservedAnalyses::none();
                }
                NewInst->insertBefore(InsertBlock->getTerminator());
                InstructionMap[&Inst] = NewInst;
              }
            }

            // Обновляем операнды в новых инструкциях
            for (auto &InstMapping : InstructionMap) {
              for (unsigned i = 0, e = InstMapping.second->getNumOperands();
                   i != e; ++i) {
                if (Instruction *Op = dyn_cast<Instruction>(
                        InstMapping.second->getOperand(i))) {
                  if (InstructionMap.count(Op)) {
                    InstMapping.second->setOperand(i, InstructionMap[Op]);
                  }
                }
              }
            }

            // Заменяем вызов функции на переход к новому блоку
            BranchInst::Create(&F.getEntryBlock(), CI);

            // Помечаем вызов для удаления
            CallsToRemove.insert(CI);
          }
        }
      }
    }

    // Удаляем вызовы функций
    for (CallInst *CI : CallsToRemove) {
      CI->eraseFromParent();
    }

    // Возвращаем, что анализы не изменены
    return PreservedAnalyses::all();
  }
};

} // end anonymous namespace

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "SimonyanInliningPass", "v0.1",
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "simonyan-inlining") {
                    FPM.addPass(SimonyanInliningPass());
                    return true;
                  }
                  return false;
                });
          }};
}