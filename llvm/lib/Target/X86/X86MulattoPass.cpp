#include "X86.h"
#include "X86InstrInfo.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/Register.h"

#include <map>

using namespace llvm;

namespace {
class X86MulattoPass : public MachineFunctionPass {
public:
  static char ID;

  X86MulattoPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "X86 Mulatto Pass"; }
};
} // namespace

char X86MulattoPass::ID = 0;

bool X86MulattoPass::runOnMachineFunction(MachineFunction &MF) {
  bool changed = false;

  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();

  for (MachineBasicBlock &MBB : MF) {
    std::map<MachineInstr *, MachineInstr *> instCandidates;
    for (auto MI = MBB.begin(); MI != MBB.end(); ++MI) {
      MachineInstr *MULPD = &(*MI);
      if (MULPD->getOpcode() == X86::MULPDrr || MULPD->getOpcode() == X86::MULPDrm) {
        for (auto MIAfterMul = std::next(MI); MIAfterMul != MBB.end(); ++MIAfterMul) {
          if (MIAfterMul->readsRegister(MULPD->getOperand(0).getReg())) {
            if (MIAfterMul->getOpcode() == X86::ADDPDrr || MIAfterMul->getOpcode() == X86::ADDPDrm) {
              instCandidates.insert({MULPD, &(*MIAfterMul)});
            }
            break;
          }
        }
      }
    }

    for (auto candidate : instCandidates) {
      MachineInstr *mulInstr = candidate.first;
      MachineInstr *addInstr = candidate.second;

      MachineBasicBlock &MBB = *mulInstr->getParent();

      BuildMI(MBB, mulInstr, mulInstr->getDebugLoc(), TII->get(X86::VFMADD213PDr))
        .addReg(addInstr->getOperand(0).getReg())
        .addReg(mulInstr->getOperand(1).getReg())
        .addReg(mulInstr->getOperand(2).getReg())
        .addReg(addInstr->getOperand(2).getReg());

      mulInstr->eraseFromParent();
      addInstr->eraseFromParent();

      changed = true;
    }
  }

  return changed;
}

INITIALIZE_PASS(X86MulattoPass, "x86-mulatto-pass", "X86 Mulatto Pass", false,
                false)

FunctionPass *llvm::createX86MulattoPass() { return new X86MulattoPass(); }
