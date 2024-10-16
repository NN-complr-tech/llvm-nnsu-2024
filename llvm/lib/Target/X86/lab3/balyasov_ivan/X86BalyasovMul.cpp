#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsX86.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
class BalyasovMulPass : public MachineFunctionPass {
public:
  static char ID;
  BalyasovMulPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    SmallVector<MachineInstr *> InstructionsToDelete;
    bool Changed = false;

    for (auto &MBB : MF) {
      for (auto MI = MBB.begin(); MI != MBB.end(); ++MI) {
        if (MI->getOpcode() == X86::MULPDrr) {
          Changed |= processInstruction(MF, MBB, MI, InstructionsToDelete);
        }
      }
    }

    for (auto *Instr : InstructionsToDelete)
      Instr->eraseFromParent();

    return Changed;
  }

private:
  bool processInstruction(MachineFunction &MF, MachineBasicBlock &MBB,
                          MachineBasicBlock::iterator MI,
                          SmallVector<MachineInstr *> &InstructionsToDelete) {
    Register MulDestReg = MI->getOperand(0).getReg();

    for (auto NextMI = std::next(MI); NextMI != MBB.end(); ++NextMI) {
      if (NextMI->getOpcode() == X86::ADDPDrr &&
          usesRegister(NextMI, MulDestReg)) {
        if (!hasInterveningDependency(NextMI, MulDestReg, MBB)) {
          replaceWithFusedInstruction(MF, MBB, MI, NextMI, MulDestReg);
          InstructionsToDelete.push_back(&*MI);
          InstructionsToDelete.push_back(&*NextMI);
          return true;
        }
        break;
      } else if (usesRegister(NextMI, MulDestReg)) {
        break;
      }
    }

    return false;
  }

  bool usesRegister(MachineBasicBlock::iterator MI, Register Reg) {
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      if (MI->getOperand(i).getReg() == Reg) {
        return true;
      }
    }
    return false;
  }

  bool hasInterveningDependency(MachineBasicBlock::iterator NextMI,
                                Register Reg, MachineBasicBlock &MBB) {
    if (NextMI->getOperand(0).getReg() != Reg) {
      for (auto CheckMI = std::next(NextMI); CheckMI != MBB.end(); ++CheckMI) {
        if (usesRegister(CheckMI, Reg)) {
          return true;
        }
      }
    }

    return (NextMI->getOperand(1).getReg() == Reg &&
            NextMI->getOperand(2).getReg() == Reg);
  }

  void replaceWithFusedInstruction(MachineFunction &MF, MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MI,
                                   MachineBasicBlock::iterator NextMI,
                                   Register Reg) {
    MachineInstrBuilder Builder =
        BuildMI(MBB, MI, MI->getDebugLoc(),
                MF.getSubtarget().getInstrInfo()->get(X86::VFMADD213PDr));
    Builder.addReg(NextMI->getOperand(0).getReg(), RegState::Define);
    Builder.addReg(MI->getOperand(1).getReg());
    Builder.addReg(MI->getOperand(2).getReg());
    Builder.addReg(NextMI->getOperand(2).getReg());
  }
};

char BalyasovMulPass::ID = 0;
static RegisterPass<BalyasovMulPass> X("x86-balyasov-mul-pass",
                                       "x86 Balyasov Intrinsics Pass");
} // namespace
