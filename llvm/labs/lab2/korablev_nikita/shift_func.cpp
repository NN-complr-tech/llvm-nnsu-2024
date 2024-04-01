#include "llvm/IR/Function.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"
#include <stack>

namespace {

struct ReplaceMultToShift : llvm::PassInfoMixin<ReplaceMultToShift> {
  
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM) {
    std::stack<llvm::Instruction *> inst_list;
    
    for (llvm::BasicBlock &BB : F) {
      for (llvm::Instruction &I : BB) {
        
        // I.print(llvm::outs()); llvm::outs() << "\n";

        // if (auto *BO = llvm::dyn_cast<llvm::BinaryOperator>(&I)) {
        //   BO->print(llvm::outs()); llvm::outs() << "\n";
        // }

        // Является ли "I" бинарным оператом
        // llvm::outs() << "---- BO start ----\n";
        if (llvm::BinaryOperator *BO = llvm::dyn_cast<llvm::BinaryOperator>(&I)) {
          // llvm::outs() << "---- BO in ----\n";
          // llvm::IRBuilder<> builder(BO);
          
          if (BO->getOpcode() == llvm::Instruction::BinaryOps::Mul) {
            // BO->getOperand(0)->print(llvm::outs()); llvm::outs() << "\n";
            // BO->getOperand(1)->print(llvm::outs());
            // llvm::outs() << "\nTEST1::::: "  << "\n";
            llvm::ConstantInt *LHS = llvm::dyn_cast<llvm::ConstantInt>(BO->getOperand(0));
            llvm::ConstantInt *RHS = llvm::dyn_cast<llvm::ConstantInt>(BO->getOperand(1));

            if (LHS || RHS) {
              llvm::IRBuilder<> builder(BO);
              // llvm::outs() << "---- IF in ----\n";
              if (LHS != nullptr && LHS->getValue().isPowerOf2()) {
                llvm::outs() << "---- LHS is not nullptr ----\n";
                // llvm::Value *llvmValue = llvm::ConstantInt::get(llvm::IntegerType::get(F.getContext(), 32), llvm::APInt(32, LHS->getValue().exactLogBase2()));
                BO->replaceAllUsesWith(llvm::BinaryOperator::Create(llvm::Instruction::Shl, BO->getOperand(0), BO->getOperand(1), "shiftInst", BO));
                llvm::outs() << "---- LHS end ----\n";
              } else if (RHS != nullptr && RHS->getValue().isPowerOf2()) {
                llvm::outs() << "---- RHS is not nullptr ----\n";
                // llvm::Value *llvmValue = llvm::ConstantInt::get(llvm::IntegerType::get(F.getContext(), 32), llvm::APInt(32, RHS->getValue().exactLogBase2()));
                BO->replaceAllUsesWith(llvm::BinaryOperator::Create(llvm::Instruction::Shl, BO->getOperand(1), BO->getOperand(0), "shiftInst", BO));
                // BO->replaceAllUsesWith(builder.CreateShl(BO->getOperand(1), llvm::ConstantInt::get(BO->getType(), lg1)));
              // Удаляем Mul
                llvm::outs() << "---- RHS end ----\n";
              }
              // llvm::outs() << "---- IF out ----\n";
              inst_list.push(&I);
            }
          }
        }
      }
      while(!inst_list.empty()) {
        inst_list.top()->eraseFromParent();
        inst_list.pop();
      }
    }
    
    // llvm::outs() << "---- Out start ----\n";
    auto PA = llvm::PreservedAnalyses::all();
    PA.abandon<llvm::LoopAnalysis>();
    return PA;
  }
};

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "Replace-Mult-Shift", LLVM_VERSION_STRING,
    [](llvm::PassBuilder &PB) {
      PB.registerPipelineParsingCallback(
        [](llvm::StringRef Name, llvm::FunctionPassManager &FPM,
           llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
          if (Name == "korablev-replace-mul-shift") {
            FPM.addPass(ReplaceMultToShift());
            return true;
          }
          return false;
        });
    }
  };
}
