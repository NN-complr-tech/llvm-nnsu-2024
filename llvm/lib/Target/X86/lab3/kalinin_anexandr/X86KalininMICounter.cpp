#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

using namespace llvm;

namespace {

class X86SKalininMICounterPass : public MachineFunctionPass {
public:
  static char ID;

  X86SKalininMICounterPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    DebugLoc debugLocation = MF.front().begin()->getDebugLoc();
    const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
    Module &module = *MF.getFunction().getParent();
    GlobalVariable *instructionCounter =
        module.getNamedGlobal("instruction_count");

    if (!instructionCounter) {
      LLVMContext &context = module.getContext();
      instructionCounter = new GlobalVariable(
          module, IntegerType::get(context, 64), false,
          GlobalValue::ExternalLinkage, nullptr, "instruction_count");
    }

    for (auto &MBB : MF) {
      int instructionCount = std::distance(MBB.begin(), MBB.end());
      auto insertionPoint = MBB.getFirstTerminator();

      BuildMI(MBB, insertionPoint, debugLocation, TII->get(X86::ADD64mi32))
          .addReg(0)
          .addImm(1)
          .addReg(0)
          .addGlobalAddress(instructionCounter)
          .addImm(instructionCount);
    }

    return true;
  }
};

char X86SKalininMICounterPass::ID = 0;

} // end anonymous namespace

static RegisterPass<X86SKalininMICounterPass>
    X("x86-kalinin-mi-counter", "X86 Count of machine instructions pass", false,
      false);