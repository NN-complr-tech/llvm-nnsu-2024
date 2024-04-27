#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

using namespace llvm;

namespace {

class X86SimonyanMICounterPass : public MachineFunctionPass {
public:
  static char ID;

  X86SimonyanMICounterPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
    DebugLoc DL3 = MF.front().begin()->getDebugLoc();

    // Создаем новый виртуальный регистр
    unsigned icReg = MF.getRegInfo().createVirtualRegister(&X86::GR64RegClass);

    // Проверяем, существует ли глобальная переменная 'ic'
    Module &M = *MF.getFunction().getParent();
    GlobalVariable *gvar = M.getGlobalVariable("ic");

    // Если 'ic' не существует, создаем ее
    if (!gvar) {
      LLVMContext &context = M.getContext();
      gvar = new GlobalVariable(M, IntegerType::get(context, 64), false,
                                GlobalValue::ExternalLinkage, nullptr, "ic");
      gvar->setAlignment(Align(8));
    }

    for (auto &MBB : MF) {
      unsigned count = 0;
      for (auto &MI : MBB) {
        if (!MI.isDebugInstr())
          ++count;
      }

      // Обновляем счетчик
      BuildMI(MBB, MBB.getFirstTerminator(), DL3, TII->get(X86::ADD64ri32),
              icReg)
          .addReg(icReg)
          .addImm(count);
    }

    for (auto &MBB : MF) {
      if (MBB.getFirstTerminator() != MBB.end()) {
        // Write to global variable ic
        BuildMI(MBB, MBB.getFirstTerminator(), DL3, TII->get(X86::MOV64mr))
            .addReg(icReg)
            .addExternalSymbol("ic");
      }
    }

    return true;
  }

};

char X86SimonyanMICounterPass::ID = 0;

} // end anonymous namespace

static RegisterPass<X86SimonyanMICounterPass>
    X("x86-simonyan-mi-counter",
      "X86 Count of machine instructions pass", false, false);