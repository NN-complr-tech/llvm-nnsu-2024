	#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

using namespace llvm;

namespace {

class X86SimonyanMICounterPass : public MachineFunctionPass {
public:
  static char ID;

  X86SimonyanMICounterPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
    DebugLoc DL3 = MF.front().begin()->getDebugLoc();

    for (auto &MBB : MF) {
      unsigned count = 0;
      for (auto &MI : MBB) {
        if (!MI.isDebugInstr())
          ++count;
      }

      BuildMI(MBB, MBB.getFirstTerminator(), DL3, TII->get(X86::ADD64ri32))
          .addImm(count)
          .addExternalSymbol("ic");
    }

    return true;
  }
};

char X86SimonyanMICounterPass::ID = 0;

} // end anonymous namespace

static RegisterPass<X86SimonyanMICounterPass>
    X("x86-simonyan-mi-counter", "X86 Count of machine instructions pass",
      false, false);