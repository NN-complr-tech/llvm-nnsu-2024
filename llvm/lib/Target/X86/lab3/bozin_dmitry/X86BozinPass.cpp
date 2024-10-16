#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include <utility>
#include <vector>

using namespace llvm;

namespace {
class X86MulAddPass : public MachineFunctionPass {
public:
  static char ID;
  X86MulAddPass() : MachineFunctionPass(ID) {}
  bool runOnMachineFunction(MachineFunction &MF) override {
    const TargetInstrInfo *info = MF.getSubtarget().getInstrInfo();
    std::vector<std::pair<MachineInstr *, MachineInstr *>> delete_instr;
    bool parameter = false;
    bool reg = false;

    for (auto &MBB : MF) {
      MachineInstr *multi_instr = nullptr;
      MachineInstr *addic_instr = nullptr;
      Register reg_multi;
      Register reg_addic_1;
      Register reg_addic_2;
      for (auto &instr : MBB) {
        if (instr.getOpcode() == X86::MULPDrr) {
          multi_instr = &instr;
          auto next_instr = std::next(instr.getIterator());
          for (next_instr; next_instr != MBB.end(); ++next_instr) {
            if (next_instr->getOpcode() == X86::ADDPDrr) {
              addic_instr = &*next_instr;
              reg_multi = multi_instr->getOperand(0).getReg();
              reg_addic_1 = addic_instr->getOperand(1).getReg();
              reg_addic_2 = addic_instr->getOperand(2).getReg();
              if (reg_multi == reg_addic_1 || reg_multi == reg_addic_2) {
                delete_instr.emplace_back(multi_instr, addic_instr);
                parameter = true;
                if (reg_multi == reg_addic_1) {
                  reg = false;
                } else {
                  reg = true;
                }
                break;
              }
            } else if (next_instr->definesRegister(
                           multi_instr->getOperand(0).getReg())) {
              break;
            }
          }
        }
      }
    }

    for (auto &[mult, addc] : delete_instr) {
      MachineInstrBuilder builder =
          BuildMI(*mult->getParent(), *mult, mult->getDebugLoc(),
                  info->get(X86::VFMADD213PDZ128r));
      builder.addReg(addc->getOperand(0).getReg(), RegState::Define);
      builder.addReg(mult->getOperand(1).getReg());
      builder.addReg(mult->getOperand(2).getReg());
      if (reg) {
        builder.addReg(addc->getOperand(1).getReg());
      } else {
        builder.addReg(addc->getOperand(2).getReg());
      }
      mult->eraseFromParent();
      addc->eraseFromParent();
    }

    return parameter;
  }
};
} // namespace

char X86MulAddPass::ID = 0;
static RegisterPass<X86MulAddPass> X("x86-mul-add-bozin-pass", "X86 Bozin Pass",
                                     false, false);
