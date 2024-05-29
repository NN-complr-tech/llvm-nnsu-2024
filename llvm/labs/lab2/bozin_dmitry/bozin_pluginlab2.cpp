#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Value.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/Casting.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <map>
#include <vector>
#include "llvm/IR/PassManager.h"
#include "llvm/IR/InstrTypes.h"

class InstructionCloner {
public:
    InstructionCloner(llvm::LLVMContext &Context) : Context(Context) {}

    void cloneInstructions(llvm::Function *SourceFunc, llvm::Function *TargetFunc) {
        llvm::ValueToValueMapTy VMap;
        llvm::SmallVector<llvm::ReturnInst *, 8> Returns; // Store return instructions

        // Clone the body of the source function into the target function
        llvm::CloneFunctionInto(TargetFunc, SourceFunc, VMap, llvm::CloneFunctionChangeType::LocalChangesOnly, Returns, "", nullptr);

        // Fix the return instructions in the cloned function
        for (auto *RI : Returns) {
            llvm::ReturnInst::Create(Context, RI->getReturnValue(), RI);
            RI->eraseFromParent();
        }
    }

private:
    llvm::LLVMContext &Context;
};


class BozinInlinePass
    : public llvm::PassInfoMixin<BozinInlinePass> {
public:
  llvm::PreservedAnalyses run(llvm::Function &Func,
                              llvm::FunctionAnalysisManager &) {
    std::vector<llvm::CallInst*> CallsToInline;

        for (auto &BB : Func) {
            for (auto &Inst : llvm::make_early_inc_range(BB)) {
                if (auto *CallInst = llvm::dyn_cast<llvm::CallInst>(&Inst)) {
                    llvm::Function *Callee = CallInst->getCalledFunction();
                    if (Callee && Callee->arg_empty() && Callee->getReturnType()->isVoidTy()) {
                        llvm::outs() << "Inlined function '"
                               << Callee->getName() << "'!\n";
                        CallInst->eraseFromParent();
                    }
                }
            }
        }  
    return llvm::PreservedAnalyses::all();}};
llvm::PassPluginLibraryInfo
getBozinInlinePluginPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "BozinInlinePass", "0.1",
          [](llvm::PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](llvm::StringRef Name, llvm::FunctionPassManager &PM,
                   llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
                  if (Name == "bozin-inline") {
                    PM.addPass(BozinInlinePass());
                    return true;
                  }
                  return false;
                });
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getBozinInlinePluginPluginInfo();
}
    