#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

namespace {

class X86KanakovMICounter : public MachineFunctionPass {
public:
  static char ID;
  X86KanakovMICounter() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
    DebugLoc DL = MF.front().begin()->getDebugLoc();

    Module &module = *MF.getFunction().getParent();
    GlobalVariable *var = module.getGlobalVariable("ic");

    if (!var) {
      var = new GlobalVariable(module,
                               IntegerType::get(module.getContext(), 64), false,
                               GlobalValue::ExternalLinkage, nullptr, "ic");
    }

    for (auto &BB : MF) {
      size_t count = std::distance(BB.begin(), BB.end());

      BuildMI(BB, BB.getFirstTerminator(), DL, TII->get(X86::ADD64mi32))
          .addGlobalAddress(var, 0, X86II::MO_NO_FLAG)
          .addImm(count);
    }

    return true;
  }
};
} // namespace

char X86KanakovMICounter::ID = 0;
static RegisterPass<X86KanakovMICounter>
    X("kanakov_mi_counter", "X86 Count number of machine instructions pass",
      false, false);
