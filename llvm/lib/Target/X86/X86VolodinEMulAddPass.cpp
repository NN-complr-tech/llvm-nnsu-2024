#include <algorithm>

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

#define DESC "X86 replace mul and add with muladd pass"
#define NAME "x86-volodin-replace-mul-add"

#define DEBUG_TYPE MULADD_NAME

using namespace llvm;

namespace {
class X86VolodinEMulAddPass : public MachineFunctionPass {
public:
  X86VolodinEMulAddPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  static char ID;

private:
  StringRef getPassName() const override { return DESC; }
};
} // end anonymous namespace

char X86VolodinEMulAddPass::ID = 0;

FunctionPass *llvm::createX86VolodinEMulAddPass() {
  return new X86VolodinEMulAddPass();
}

INITIALIZE_PASS(X86VolodinEMulAddPass, NAME, DESC, false, false)

bool findInstruction(std::vector<MachineInstr *> &MIvector,
                     MachineInstr *instruction) {
  auto result{std::find(begin(MIvector), end(MIvector), instruction)};
  if (result == end(MIvector))
    return false;
  else
    return true;
}

bool X86VolodinEMulAddPass::runOnMachineFunction(MachineFunction &MF) {
  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo &TII = *STI.getInstrInfo();

  bool Changed = false;
  std::vector<MachineInstr *> MIvector;

  for (MachineBasicBlock &MBB : MF) {
    for (auto iterator = MBB.begin(); iterator != MBB.end(); ++iterator) {
      if (iterator->getOpcode() == X86::MULPDrr) {
        auto multiplicaton = &(*iterator);
        for (auto iter = std::next(iterator); iter != MBB.end(); ++iter) {
          if (iter != MBB.end() && iter->getOpcode() == X86::ADDPDrr) {
            if (multiplicaton->getOperand(0).getReg() ==
                    iter->getOperand(1).getReg() ||
                multiplicaton->getOperand(0).getReg() ==
                    iter->getOperand(2).getReg()) {
              auto addition = &(*iter);
              MIMetadata MIMD(*iterator);
              MachineInstrBuilder MIB =
                  BuildMI(MBB, multiplicaton, MIMD, TII.get(X86::VFMADD213PDr));
              MIB.addReg(addition->getOperand(0).getReg(), RegState::Define);
              MIB.addReg(multiplicaton->getOperand(1).getReg());
              MIB.addReg(multiplicaton->getOperand(2).getReg());
              if (multiplicaton->getOperand(0).getReg() ==
                  iter->getOperand(1).getReg()) {
                MIB.addReg(addition->getOperand(2).getReg());
              } else if (multiplicaton->getOperand(0).getReg() ==
                         iter->getOperand(2).getReg()) {
                MIB.addReg(addition->getOperand(1).getReg());
              }
              if (!findInstruction(MIvector, multiplicaton)) {
                MIvector.push_back(multiplicaton);
              }
              if (!findInstruction(MIvector, addition)) {
                MIvector.push_back(addition);
              }
              Changed = true;
            }
          }
        }
      }
    }
  }
  
  for (auto &MI : MIvector) {
    MI->eraseFromParent();
  }
  return Changed;
}