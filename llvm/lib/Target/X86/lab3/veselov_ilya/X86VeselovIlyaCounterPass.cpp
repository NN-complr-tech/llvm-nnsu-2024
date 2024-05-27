#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

class X86VeselovIlyaCounterPass : public MachineFunctionPass {
public:
  static char ID;
  X86VeselovIlyaCounterPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &mFunc) override {
    Module &mod = *mFunc.getFunction().getParent();
    LLVMContext &context = mod.getContext();
    GlobalVariable *gVar = mod.getGlobalVariable("ic");
    if (!gVar) {
      gVar = new GlobalVariable(mod, Type::getInt64Ty(context), false,
                                GlobalValue::ExternalLinkage, nullptr, "ic");
      gVar->setInitializer(ConstantInt::get(Type::getInt64Ty(context), 0));
    }
    const TargetInstrInfo *tii = mFunc.getSubtarget().getInstrInfo();
    DebugLoc dl = mFunc.front().begin()->getDebugLoc();

    for (auto &mbb : mFunc) {
      unsigned InstrCount = 0;
      for (auto &mi : mbb) {
        if (!mi.isDebugInstr())
          ++InstrCount;
      }
      if (InstrCount > 0) {
        MachineBasicBlock::iterator mi = mbb.getFirstTerminator();
        Register tempReg =
            mFunc.getRegInfo().createVirtualRegister(&X86::GR64RegClass);

        BuildMI(mbb, mi, dl, tii->get(X86::MOV64rm), tempReg)
            .addReg(X86::RIP)
            .addImm(0)
            .addReg(0)
            .addGlobalAddress(gVar)
            .addReg(0);

        BuildMI(mbb, mi, dl, tii->get(X86::ADD64mi32))
            .addReg(tempReg)
            .addImm(0)
            .addReg(0)
            .addImm(0)
            .addImm(InstrCount);
      }
    }
    return true;
  }
};

char X86VeselovIlyaCounterPass::ID = 0;

static RegisterPass<X86VeselovIlyaCounterPass>
    X("x86-veselov-ilya-counter-pass",
      "Count number of machine instructions performed during execution of a "
      "function (excluding instruction counter increment)",
      false, false);
