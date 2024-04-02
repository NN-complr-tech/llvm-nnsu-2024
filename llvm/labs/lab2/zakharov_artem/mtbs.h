#ifndef LLVM_MTBS_H
#define LLVM_MTBS_H
#include "llvm/IR/PassManager.h"

namespace llvm {

class MTBSPass : public PassInfoMixin<MTBSPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

private:
  int getLog2(Value *Op);
};

} // namespace llvm

#endif // LLVM_MTBS_H
