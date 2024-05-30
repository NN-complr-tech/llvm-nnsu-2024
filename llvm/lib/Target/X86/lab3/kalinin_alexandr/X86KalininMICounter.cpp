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
    DebugLoc debugLoc = machineFunction.front().begin()->getDebugLoc();
    const TargetInstrInfo *targetInstrInfo =
        machineFunction.getSubtarget().getInstrInfo();
    Module &module = *machineFunction.getFunction().getParent();
    GlobalVariable *globalCounter = module.getGlobalVariable("ic");

    if (!globalCounter) {
      LLVMContext &llvmContext = module.getContext();
      globalCounter =
          new GlobalVariable(module, IntegerType::get(llvmContext, 64), false,
                             GlobalValue::ExternalLinkage, nullptr, "ic");
    }

    for (auto &basicBlock : machineFunction) {
      int instructionCount =
          std::distance(basicBlock.begin(), basicBlock.end());
      auto insertPoint = basicBlock.getFirstTerminator();

      if (insertPoint != basicBlock.end() &&
          insertPoint != basicBlock.begin() &&
          insertPoint->getOpcode() >= X86::JCC_1 &&
          insertPoint->getOpcode() <= X86::JCC_4) {
        --insertPoint;
      }

      BuildMI(basicBlock, insertPoint, debugLoc,
              targetInstrInfo->get(X86::ADD64mi32))
          .addReg(0)
          .addImm(1)
          .addReg(0)
          .addGlobalAddress(globalCounter)
          .addReg(0)
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