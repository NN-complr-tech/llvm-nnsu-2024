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
    DebugLoc debugLocation = machineFunction.front().begin()->getDebugLoc();
    const TargetInstrInfo *targetInstrInfo =
        machineFunction.getSubtarget().getInstrInfo();
    Module &module = *machineFunction.getFunction().getParent();
    GlobalVariable *instructionCounter =
        module.getNamedGlobal("instruction_counter");

    if (!instructionCounter) {
      LLVMContext &context = module.getContext();
      instructionCounter = new GlobalVariable(
          module, IntegerType::get(context, 64), false,
          GlobalValue::ExternalLinkage, nullptr, "instruction_counter");
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

      BuildMI(basicBlock, insertPoint, debugLocation,
              targetInstrInfo->get(X86::ADD64mi32))
          .addReg(0)
          .addImm(1)
          .addReg(0)
          .addGlobalAddress(instructionCounter)
          .addReg(0)
          .addImm(instructionCount);
    }

    return true;
  }
};

char X86SimonyanMICounterPass::ID = 0;

} // end anonymous namespace

static RegisterPass<X86SKalininMICounterPass>
    X("x86-kalinin-mi-counter", "X86 Count of machine instructions pass", false,
      false);