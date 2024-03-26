#define DEBUG_TYPE "opCounter"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

#include "llvm/ADT/PriorityWorklist.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Transforms/Utils/LoopUtils.h"


using namespace llvm;

class MyLoopPass : public PassInfoMixin<MyLoopPass> {
  public:
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) {
        const auto &LI = FAM.getAnalysis<LoopAnalysis>(const_cast<Function &>(F));
        // LoopInfo &LI = AM.getResult<LoopAnalysis>(F);

        errs() << "Printing analysis 'Loop Access Analysis' for function '" << F.getName()
      << "':\n";

        // SmallPriorityWorklist<Loop *, 4> Worklist;
        // appendLoopsToWorklist(LI, Worklist);
        // while (!Worklist.empty()) {
        //   Loop *L = Worklist.pop_back_val();
        //   OS.indent(2) << L->getHeader()->getName() << ":\n";
        //   LAIs.getInfo(*L).print(OS, 4);
        // }

        // for (Loop *L : LI) {
        //     BasicBlock *header = L->getHeader();
        //     BasicBlock *preheader = L->getLoopPreheader();

        //     if (!preheader) {
        //         preheader = BasicBlock::Create(F.getContext(), "loop_preheader", &F);
        //         BranchInst::Create(header, preheader);
        //     }

        //     CallInst::Create(F.getParent()->getOrInsertFunction("loop_start", Type::getVoidTy(F.getContext())), "", preheader->getTerminator());

        //     for (BasicBlock *BB : L->blocks()) {
        //         for (Instruction &I : *BB) {
        //             if (BranchInst *BI = dyn_cast<BranchInst>(&I)) {
        //                 if (BI->isUnconditional()) {
        //                     if (BI->getSuccessor(0) == header) {
        //                         CallInst::Create(F.getParent()->getOrInsertFunction("loop_end", Type::getVoidTy(F.getContext())), "", BI);
        //                     }
        //                 }
        //             }
        //         }
        //     }
        // }

        return PreservedAnalyses::all();
    }
};

// class CountOp : public PassInfoMixin<CountOp> {
//   std::map<std::string, int> opCounter;
// public:
//   PreservedAnalyses run(Function &F,
//                                       FunctionAnalysisManager &AM) {

//   errs() << "Function " << F.getName() << '\n';
//     for (Function::iterator bb = F.begin(), e = F.end(); bb != e; ++bb) {
//       for (BasicBlock::iterator i = bb->begin(), e = bb->end(); i != e; ++i) {
//         if (opCounter.find(i->getOpcodeName()) == opCounter.end()) {
//           opCounter[i->getOpcodeName()] = 1;
//         } else {
//           opCounter[i->getOpcodeName()] += 1;
//         }
//       }
//     }
//     std::map<std::string, int>::iterator i = opCounter.begin();
//     std::map<std::string, int>::iterator e = opCounter.end();
//     while (i != e) {
//       errs() << i->first << ": " << i->second << "\n";
//       i++;
//     }
//     errs() << "\n";
//     opCounter.clear();
//   return PreservedAnalyses::all();
// }
// };

// char MyLoopPass::ID = 0;

/* New PM Registration */
llvm::PassPluginLibraryInfo getMyLoopPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "MyLoopPass", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerVectorizerStartEPCallback(
                [](llvm::FunctionPassManager &PM, OptimizationLevel Level) {
                  PM.addPass(MyLoopPass());
                });
            PB.registerPipelineParsingCallback(
                [](StringRef Name, llvm::FunctionPassManager &PM,
                   ArrayRef<llvm::PassBuilder::PipelineElement>) {
                  if (Name == "loop-pass") {
                    PM.addPass(MyLoopPass());
                    return true;
                  }
                  return false;
                });
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getMyLoopPassPluginInfo();
}  

// char CountOp::ID = 0;
// static RegisterPass<CountOp> X("opCounter", "Counts opcodes per functions");