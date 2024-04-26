#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

class X86NoginCountInstsPass : public MachineFunctionPass {
public:
  static inline char ID = 0;

  X86NoginCountInstsPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MachFunc) override {
    DebugLoc DebugLocation = MachFunc.front().begin()->getDebugLoc();
    const TargetInstrInfo *InstrInfo = MachFunc.getSubtarget().getInstrInfo();
    size_t Count = 0;

    MachineBasicBlock *ReturnBlock = nullptr;
    for (auto &BasicBlock : MachFunc) {
      Count += countInstrBasBlock(BasicBlock);
      for (auto &Instr : BasicBlock) {
        if (Instr.isReturn()) {
          ReturnBlock = &BasicBlock;
          DebugLocation = Instr.getDebugLoc();
          break;
        }
      }
    }

    if (!ReturnBlock) {
      ReturnBlock = &MachFunc.back();
      DebugLocation = ReturnBlock->begin()->getDebugLoc();
    }

    BuildMI(*ReturnBlock, ReturnBlock->getFirstTerminator(),
            DebugLocation, InstrInfo->get(X86::MOV64mr))
        .addImm(Count)
        .addExternalSymbol("ic");

    return true;
  }

private:
  size_t countInstrBasBlock(MachineBasicBlock &BasicBlock) {
    size_t Count = 0;
    for (auto &CurInstr : BasicBlock) {
      Count++;
    }
    return Count;
  }
};

INITIALIZE_PASS(X86NoginCountInstsPass, "x86-nogin-count-insts",
                "A pass counting the number of X86 machine instructions", false,
                false)