#include "X86.h"
#include "X86InstrInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

class X86SoloninkoOptsPass : public MachineFunctionPass {
public:
  static inline char ID = 0;

  X86SoloninkoOptsPass() : MachineFunctionPass(ID) {}

private:

  void buildMI(llvm::MachineBasicBlock &MachineBlock, llvm::MachineBasicBlock::iterator &MulInstr, 
               const llvm::TargetInstrInfo *TII, llvm::MachineBasicBlock::iterator &MBIter) {
        llvm::MachineInstrBuilder MIB =
                    BuildMI(MachineBlock, *MulInstr, MulInstr->getDebugLoc(),
                            TII->get(llvm::X86::VFMADD213PDr));
                MIB.addReg(MBIter->getOperand(0).getReg(), llvm::RegState::Define);
                MIB.addReg(MulInstr->getOperand(1).getReg());
                MIB.addReg(MulInstr->getOperand(2).getReg());
                MIB.addReg(MBIter->getOperand(2).getReg());
                MBIter = MachineBlock.erase(MBIter);
                MulInstr->eraseFromParent();
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    for (llvm::MachineBasicBlock &MachineBlock : MF) {
      for (llvm::MachineBasicBlock::iterator MBIter = MachineBlock.begin();
           MBIter != MachineBlock.end();) {
        if (MBIter->getOpcode() != llvm::X86::MULPDrr) {
          ++MBIter;
        } else {
          llvm::MachineBasicBlock::iterator MulInstr = MBIter;
          ++MBIter;
          while (MBIter != MachineBlock.end()) {
            if (MBIter->getOpcode() == llvm::X86::ADDPDrr) {
              if (MulInstr->getOperand(0).getReg() ==
                  MBIter->getOperand(1).getReg()) {
                const llvm::TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
                buildMI(MachineBlock, MulInstr, TII, MBIter);
              }
            } else {
              break;
            }
            ++MBIter;
          }
        }
      }
    }
    return true;
  }
};

INITIALIZE_PASS(X86SoloninkoOptsPass, "x86-soloninko-lab3",
                "X86 Soloninko Mult Add Opts Pass", false, false)