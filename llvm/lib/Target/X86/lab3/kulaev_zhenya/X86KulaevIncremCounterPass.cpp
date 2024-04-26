#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

#define PASS_NAME "x86-kulaev-increm-counter-pass"

namespace {
class X86KulaevIncremCounterPass : public MachineFunctionPass {
public:
  static char ID;
  X86KulaevIncremCounterPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MachineFunctionRunning) override;

private:
  void addInReg(MachineFunction &mi, DebugLoc dl, const TargetInstrInfo *TII,
                unsigned registreIc, MachineBasicBlock *Return);
  void updateCount(DebugLoc dl, MachineFunction &mf, const TargetInstrInfo *ti,
                   unsigned registreIc, MachineBasicBlock *Return);
};
} // namespace

bool X86KulaevIncremCounterPass::runOnMachineFunction(
    MachineFunction &MachineFunctionRunning) {
  const TargetInstrInfo *TargetInstructionInfo =
      MachineFunctionRunning.getSubtarget().getInstrInfo();
  DebugLoc DL3 = MachineFunctionRunning.front().begin()->getDebugLoc();

  unsigned icReg = MachineFunctionRunning.getRegInfo().createVirtualRegister(
      &X86::GR64RegClass);

  MachineBasicBlock *ReturnBlock = nullptr;

  updateCount(DL3, MachineFunctionRunning, TargetInstructionInfo, icReg, ReturnBlock);
  addInReg(MachineFunctionRunning, DL3, TargetInstructionInfo, icReg, ReturnBlock);

  return true;
}

void X86KulaevIncremCounterPass::updateCount(DebugLoc dl, MachineFunction &mf,
                                             const TargetInstrInfo *ti,
                                             unsigned registreIc, MachineBasicBlock *Return) {
  for (auto &MBasicBlock : mf) {
    unsigned count = 0;
    for (auto &MInstruction : MBasicBlock) {
      ++count;
      if (MInstruction.isReturn()) {
          Return = &MBasicBlock;
      }
    }

    // updating the counter
    BuildMI(MBasicBlock, MBasicBlock.getFirstTerminator(), dl,
            ti->get(X86::ADD64ri32), registreIc)
        .addReg(registreIc)
        .addImm(count);
  }
}

void X86KulaevIncremCounterPass::addInReg(MachineFunction &mi, DebugLoc dl,
                                          const TargetInstrInfo *TII,
                                          unsigned registreIc, MachineBasicBlock *Return) {
  if (!Return) {
      Return = &mi.back();
  }
  // adding a register
  BuildMI(mi.back(), mi.back().getFirstTerminator(), dl, TII->get(X86::MOV64mr))
      .addReg(registreIc)
      .addExternalSymbol("ic");
}

char X86KulaevIncremCounterPass::ID = 0;

static RegisterPass<X86KulaevIncremCounterPass>
    X(PASS_NAME,
      "Сounts the number of executed instructions in our function", false, false);