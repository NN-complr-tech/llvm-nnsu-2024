#include "X86.h"
#include "X86InstrInfo.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/Register.h"

#include <iostream>
#include <typeinfo>
#include <vector>

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
  std::vector<std::pair<MachineInstr *, std::vector<MachineInstr *>>> candidates;

  // обходим все блоки
  for (MachineBasicBlock &MBB : MF) {

    // проходим все инструкции блока в поисках умножений
    for (auto MI = MBB.begin(); MI != MBB.end(); ++MI) {
      MachineInstr *MULPD = &(*MI);

      // является ли умножением рассматриваемя инструкция
      if (MULPD->getOpcode() == X86::MULPDrr || MULPD->getOpcode() == X86::MULPDrm) {
        outs() << "found mul\n";
        candidates.push_back({MULPD, std::vector<MachineInstr *>()});

        // начинаем обход всех инструкции после умножения
        outs() << "started find loop\n";
        for (auto MIam = std::next(MI); MIam != MBB.end(); ++MIam) {
          
          // проверка на модификацию/чтение регистра результата умножения

          // регистр изменили
          if (MIam->modifiesRegister(MULPD->getOperand(0).getReg())) {
            outs() << "reg mod by ";
            if (MIam->getOpcode() == X86::ADDPDrr || MIam->getOpcode() == X86::ADDPDrm) {

              // если изменило сложение, то сохраняем операцию и завершаем обход
              outs() << "add\n";
              candidates.back().second.push_back(&(*MIam));
              break;
            } else {
              
              // если поменяло что-то другое, просто завершаем обход
              outs() << "smth but not add\n";
              break;
            }
          }

          // регистр прочитали
          if (MIam->readsRegister(MULPD->getOperand(0).getReg())) {
            outs() << "reg read by ";
            if (MIam->getOpcode() == X86::ADDPDrr || MIam->getOpcode() == X86::ADDPDrm) {
              
              // если прочитало сложение, то сохраняем операцию и продолжаем обход
              outs() << "add\n";
              candidates.back().second.push_back(&(*MIam));
            } else {
              
              // если поменяло что-то другое, то просто идем дальше
              outs() << "smth but not add\n";
            }
          }
        }

        // если в результате обхода не нашли сложений, то ликвидируем кандидата
        if (candidates.back().second.size() == 0) {
          
          // печать
          outs() << "didn't find pair for:\n";
          outs() << *(candidates.back().first) << "\n";
          outs() << "--------------------\n\n";

          candidates.pop_back();

          // печать
        } else {
          outs() << "\nMULL:\n";
          outs() << *(candidates.back().first) << "\n";
          outs() << "ADD(s):\n";
          for (auto add : candidates.back().second) {
            outs() << *add << "\n";      
          }
          outs() << "--------------------\n\n";
        }
      }
    }
  }

  // for (const auto &pair : candidates) {

  // }

  // for (const auto &pair : candidates) {
  //   outs() << "Умножение: " << TII->getName(pair.first->getOpcode()) << "\n";
  //   outs() << "Сложение(я): "; 
  //   for (auto add : pair.second) {
  //     outs() << TII->getName(add->getOpcode()) << " | ";
  //   }
  //   outs() << "\n";
  // }

  return changed;
}

INITIALIZE_PASS(X86MulattoPass, "x86-mulatto-pass", "X86 Mulatto Pass", false,
                false)

FunctionPass *llvm::createX86MulattoPass() { return new X86MulattoPass(); }