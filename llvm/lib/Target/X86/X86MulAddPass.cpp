#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

#define X86_MULADD_PASS_DESC "X86 muladd pass"
#define X86_MULADD_PASS_NAME "x86-muladd"

namespace {
class X86MulAddPass : public MachineFunctionPass {
public:
  static char ID;

  X86MulAddPass() : MachineFunctionPass(ID) {
    initializeX86MulAddPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return X86_MULADD_PASS_DESC; }
};

char X86MulAddPass::ID = 0;

bool X86MulAddPass::runOnMachineFunction(MachineFunction &machineFunc) {
  const TargetInstrInfo *instrInfo = machineFunc.getSubtarget().getInstrInfo();
  bool changed = false;

  for (auto &block : machineFunc) {
    for (auto it = block.begin(); it != block.end(); ++it) {
      if (it->getOpcode() == X86::MULPDrr) {
        auto instrMul = it;
        auto instrNext = std::next(instrMul);
        if (instrNext->getOpcode() == X86::ADDPDrr) {
          if (instrMul->getOperand(0).getReg() ==
              instrNext->getOperand(1).getReg()) {
            --it;
            MachineInstr &MI = *instrMul;
            MachineInstrBuilder MIB =
                BuildMI(block, MI, MI.getDebugLoc(),
                        instrInfo->get(X86::VFMADD213PDZ128r));
            MIB.addReg(instrNext->getOperand(0).getReg(), RegState::Define);
            MIB.addReg(instrMul->getOperand(1).getReg());
            MIB.addReg(instrMul->getOperand(2).getReg());
            MIB.addReg(instrNext->getOperand(2).getReg());
            instrMul->eraseFromParent();
            instrNext->eraseFromParent();
            changed = true;
          }
        }
      }
    }
  }

  return changed;
}

} // namespace

INITIALIZE_PASS(X86MulAddPass, X86_MULADD_PASS_NAME, X86_MULADD_PASS_NAME,
                false, false)

namespace llvm {
FunctionPass *createX86MulAddPassPass() { return new X86MulAddPass(); }
} // namespace llvm