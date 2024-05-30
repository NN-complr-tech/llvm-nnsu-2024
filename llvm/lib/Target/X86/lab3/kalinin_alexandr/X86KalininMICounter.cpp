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

  bool runOnMachineFunction(MachineFunction &machineFunction) override {
    const TargetInstrInfo *targetInstrInfo =
        machineFunction.getSubtarget().getInstrInfo();
    DebugLoc debugLocation = machineFunction.front().begin()->getDebugLoc();

    Module &module = *machineFunction.getFunction().getParent();
    GlobalVariable *globalVar = module.getGlobalVariable("ic");

    if (!globalVar) {
      LLVMContext &llvmContext = module.getContext();
      globalVar =
          new GlobalVariable(module, IntegerType::get(llvmContext, 64), false,
                             GlobalValue::ExternalLinkage, nullptr, "ic");
    }

    for (auto &basicBlock : machineFunction) {
      unsigned instructionCount = 0;
      for (auto &instr : basicBlock) {
        if (!instr.isDebugInstr())
          ++instructionCount;
      }

      BuildMI(basicBlock, basicBlock.getFirstTerminator(), debugLocation,
              targetInstrInfo->get(X86::ADD64ri32))
          .addGlobalAddress(globalVar, 0, X86II::MO_NO_FLAG)
          .addImm(instructionCount);
    }

    return true;
  }
};

char X86SKalininMICounterPass::ID = 0;

} // end anonymous namespace

// Register the pass
static RegisterPass<X86SKalininMICounterPass>
    X("x86-kalinin-mi-counter", "X86 Count of machine instructions pass", false,
      false);
