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

  size_t countInstrBasBlock(MachineBasicBlock &BasicBlock) {
    size_t Count = 0;
    for (auto &CurInstr : BasicBlock) {
      Count++;
    }
    return Count;
  }

  bool runOnMachineFunction(MachineFunction &MachFunc) override {
    DebugLoc DebugLocation = MachFunc.front().begin()->getDebugLoc();
    const TargetInstrInfo *InstrInfo = MachFunc.getSubtarget().getInstrInfo();
    size_t CountInstsReg =
        MachFunc.getRegInfo().createVirtualRegister(&X86::GR64RegClass);

    for (auto &BasicBlock : MachFunc) {
      size_t Count = countInstrBasBlock(BasicBlock);
      BuildMI(BasicBlock, BasicBlock.getFirstTerminator(), DebugLocation,
              InstrInfo->get(X86::ADD64ri32), CountInstsReg)
          .addReg(CountInstsReg)
          .addImm(Count);
    }

    BuildMI(MachFunc.back(), MachFunc.back().getFirstTerminator(),
            DebugLocation, InstrInfo->get(X86::MOV64mr))
        .addReg(CountInstsReg)
        .addExternalSymbol("ic");

    return true;
  }
};

INITIALIZE_PASS(X86NoginCountInstsPass, "x86-nogin-count-insts",
                "A pass counting the number of X86 machine instructions", false,
                false)
