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

    auto DL = MF.front().begin()->getDebugLoc();
    auto *TII = MF.getSubtarget().getInstrInfo();

    for (auto &MBB : MF) {
      int Counter = std::distance(MBB.begin(), MBB.end());
      auto Place = MBB.getFirstTerminator();
      if (Place != MBB.begin()) {
        --Place;
      }
      BuildMI(MBB, Place, DL, TII->get(X86::ADD64mi32))
          .addReg(0)
          .addImm(1)
          .addReg(0)
          .addGlobalAddress(GV)
          .addReg(0)
          .addImm(Counter);
    }
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
