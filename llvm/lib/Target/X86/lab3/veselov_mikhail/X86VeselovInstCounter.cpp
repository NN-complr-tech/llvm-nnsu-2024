#include "X86.h"
#include "X86InstrInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

namespace {

class X86VeselovInstCounter : public MachineFunctionPass {
public:
  static char ID;
  X86VeselovInstCounter() : MachineFunctionPass(ID) {}
  bool runOnMachineFunction(llvm::MachineFunction &MF) override;
};
} // namespace

char X86VeselovInstCounter::ID = 0;

bool X86VeselovInstCounter::runOnMachineFunction(llvm::MachineFunction &MF) {
  // Получаем глобальную переменную 'ic'
  auto *GV = MF.getFunction().getParent()->getNamedGlobal("ic");
  if (!GV) {
    // Если 'ic' не существует, создаем ее
    LLVMContext &context = MF.getFunction().getParent()->getContext();
    GV = new GlobalVariable(*MF.getFunction().getParent(), IntegerType::get(context, 64), false,
                            GlobalValue::ExternalLinkage, nullptr, "ic");
    GV->setAlignment(Align(8));
    // Если глобальная переменная 'ic' не может быть создана, возвращаем false
    if (!GV) {
      return false;
    }
  }

  auto DL = MF.front().begin()->getDebugLoc();
  auto *TII = MF.getSubtarget().getInstrInfo();

  for (auto &MBB : MF) {
    // Подсчитываем количество инструкций в базовом блоке
    int Counter = std::distance(MBB.begin(), MBB.end());
    auto Place = MBB.getFirstTerminator();
    if (Place != MBB.end() && Place != MBB.begin() &&
          Place->getOpcode() >= X86::JCC_1 &&
          Place->getOpcode() <= X86::JCC_4) {
      --Place;
    }
    // Создаем новую инструкцию, которая добавляет значение Counter к глобальной переменной 'ic'
    BuildMI(MBB, Place, DL, TII->get(X86::ADD64mi32))
        .addReg(0)
        .addImm(1)
        .addReg(0)
        .addGlobalAddress(GV)
        .addReg(0)
        .addImm(Counter);
  }
  return true;
}

static RegisterPass<X86VeselovInstCounter>
    X("x86-veselov-inst-cnt", "Instruction counter", false, false);