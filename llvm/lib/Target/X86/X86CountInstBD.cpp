#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

#define X86_COUNTINSTRUCTIONS_PASS_DESC "X86 Count Instructions Pass"
#define X86_COUNTINSTRUCTIONS_PASS_NAME "x86-count-machine-instr"

namespace {
class X86BodrovCountInstructionsPass : public MachineFunctionPass {
public:
  static char ID;
  X86BodrovCountInstructionsPass() : MachineFunctionPass(ID) {}
  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  StringRef getPassName() const override {
    return X86_COUNTINSTRUCTIONS_PASS_DESC;
  }
};
} // namespace

char X86BodrovCountInstructionsPass::ID = 0;

FunctionPass *llvm::createX86BodrovCountInstructionsPass() {
  return new X86BodrovCountInstructionsPass();
}

INITIALIZE_PASS(X86BodrovCountInstructionsPass, X86_COUNTINSTRUCTIONS_PASS_NAME,
                X86_COUNTINSTRUCTIONS_PASS_DESC, false, false)

bool X86BodrovCountInstructionsPass::runOnMachineFunction(MachineFunction &MF) {
  // Check if global variable 'ic' exists
  GlobalVariable *ICGV = MF.getFunction().getParent()->getNamedGlobal("ic");

  // Get the target information from the subtarget
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();

  // If 'ic' does not exist, create it
  if (!ICGV) {
    Module *M = MF.getFunction().getParent();
    ICGV = new GlobalVariable(
        *M, Type::getInt64Ty(M->getContext()), false,
        GlobalValue::InternalLinkage,
        ConstantInt::get(Type::getInt64Ty(M->getContext()), 0), "ic");
  }

  // Create a new virtual register
  unsigned vmregister =
      MF.getRegInfo().createVirtualRegister(&X86::GR64RegClass);

  // Initialize instruction counter
  BuildMI(MF.front(), MF.front().begin(), DebugLoc(), TII->get(X86::MOV64ri),
          vmregister)
      .addImm(0);

  // Count number of machine instructions performed
  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      // Exclude the instruction counter increment
      if (MI.getOpcode() != X86::MOV64mi32 && MI.getOpcode() != X86::ADD64ri32)
        BuildMI(MBB, MI, DebugLoc(), TII->get(X86::ADD64ri32))
            .addReg(vmregister)
            .addImm(1);
    }
  }

  // Store the instruction count in the global variable 'ic'
  BuildMI(MF.front(), MF.front().begin(), DebugLoc(), TII->get(X86::MOV64mr))
      .addReg(vmregister)
      .addImm(0) // No segment
      .addReg(0) // Base register
      .addImm(1) // Scale
      .addReg(0) // Index register
      .addImm(0) // Displacement
      .addMemOperand(MF.getMachineMemOperand(
          MachinePointerInfo::getGOT(MF), MachineMemOperand::MOStore, 8,
          MF.getDataLayout().getPrefTypeAlign(
              Type::getInt64Ty(MF.getFunction().getContext()))));

  return true;
}
