#include "X86.h"
#include "X86InstrInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

namespace {

class X86ZakharovInstCounterPass : public MachineFunctionPass {
public:
  static char ID;
  X86ZakharovInstCounterPass() : MachineFunctionPass(ID) {}
  bool runOnMachineFunction(llvm::MachineFunction &MF) override {
    auto *GV = MF.getFunction().getParent()->getNamedGlobal("ic");
    if (!GV) {
      return false;
    }
    auto *TIF = MF.getSubtarget().getInstrInfo();
    MachineInstrBuilder MIB = BuildMI(MF.back(), MF.back().back(), DebugLoc(),
                                      TIF->get(X86::MOV64mi32));
    MIB.addReg(0);
    MIB.addImm(1);
    MIB.addReg(0);
    MIB.addGlobalAddress(GV);
    MIB.addReg(0);
    MIB.addImm(MF.getInstructionCount());
    return true;
  }
};
} // namespace

char X86ZakharovInstCounterPass::ID = 0;
FunctionPass *llvm::createX86ZakharovInstCounterPass() {
  return new X86ZakharovInstCounterPass();
}

INITIALIZE_PASS(X86ZakharovInstCounterPass, "X86ZakharovInstCounterPass",
                "Counting the number of instructions in a function", false,
                false)
