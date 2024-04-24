#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/Register.h"

using namespace llvm;

#define MY_COUNTINSTRUCTIONS_PASS_NAME "x86-count-machine-instructions"
#define MY_COUNTINSTRUCTIONS_PASS_DESC "X86 Count Machine Instructions Pass"

namespace {
class X86BodrovCountInstructionsPass : public MachineFunctionPass {
public:
  static char ID;
  X86BodrovCountInstructionsPass() : MachineFunctionPass(ID) {}
  bool runOnMachineFunction(MachineFunction &MF) override;
};

char X86BodrovCountInstructionsPass::ID = 0;

bool X86BodrovCountInstructionsPass::runOnMachineFunction(MachineFunction &MF) {
  // Get the global variable to store the counter
  Module *M = MF.getFunction().getParent();
  GlobalVariable *CounterVar = M->getGlobalVariable("ic");
  if (!CounterVar) {
    // If global variable doesn't exist, create it
    CounterVar =
        new GlobalVariable(*M, IntegerType::get(M->getContext(), 32), false,
                           GlobalValue::ExternalLinkage, nullptr, "ic");
  }

  // Get the target instruction info
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();

  DebugLoc DL3 = MF.front().begin()->getDebugLoc();

  // Get the register for storing the counter
  unsigned icReg = X86::R10;

  // Iterate over all basic blocks in the function
  for (auto &MBB : MF) {
    unsigned InstructionCount =
        0; // Counter to store the number of instructions
    for (auto &MI : MBB) {
      // Increment the instruction count, excluding the instruction to increment
      // the counter itself
      if (MI.getOpcode() !=
          X86::MOV32ri) { // Exclude the instruction to increment the counter
        ++InstructionCount;
      }
    }

    // Update the counter
    BuildMI(MBB, MBB.getFirstTerminator(), DL3, TII->get(X86::ADD64ri32), icReg)
        .addReg(icReg)
        .addImm(InstructionCount);
  }

  // Write to the global variable "ic"
  BuildMI(MF.back(), MF.back().begin(), DL3, TII->get(X86::MOV64mr))
      .addReg(icReg)
      .addExternalSymbol("ic");

  return true;
}

} // namespace

static RegisterPass<X86BodrovCountInstructionsPass>
    X(MY_COUNTINSTRUCTIONS_PASS_NAME, MY_COUNTINSTRUCTIONS_PASS_DESC, false,
      false);