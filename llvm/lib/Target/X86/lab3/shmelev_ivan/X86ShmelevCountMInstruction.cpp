#include "X86.h"
#include "X86InstrInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

namespace {

class X86ShmelevCountMInstruction : public MachineFunctionPass {
public:
  static char ID;
  X86ShmelevCountMInstruction() : MachineFunctionPass(ID) {}
  bool runOnMachineFunction(llvm::MachineFunction &mFunction) override {
    auto *globalVar = mFunction.getFunction().getParent()->getNamedGlobal("ic");
    if (!globalVar) {
      return false;
    }

    auto debugLoc = mFunction.front().begin()->getDebugLoc();
    auto *targetInstrInfo = mFunction.getSubtarget().getInstrInfo();

    for (auto &basicBlock : mFunction) {
      int cnt = std::distance(basicBlock.begin(), basicBlock.end());
      auto place = basicBlock.getFirstTerminator();
      if (place != basicBlock.end() && place != basicBlock.begin() &&
          place->getOpcode() >= X86::JCC_1 &&
          place->getOpcode() <= X86::JCC_4) {
        --place;
      }
      BuildMI(basicBlock, place, debugLoc, targetInstrInfo->get(X86::ADD64mi32))
          .addReg(0)
          .addImm(1)
          .addReg(0)
          .addGlobalAddress(globalVar)
          .addReg(0)
          .addImm(cnt);
    }
    return true;
  }
};
} // namespace

char X86ShmelevCountMInstruction::ID = 0;
static RegisterPass<X86ShmelevCountMInstruction>
    X("x86-shmelev-count-minstruction",
      "Counts the number of machine instructions executed during"
      "the execution of the function",
      false, false);