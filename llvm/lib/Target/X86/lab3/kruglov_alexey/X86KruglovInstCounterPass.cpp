#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"


using namespace llvm;

namespace {
class X86KruglovCntPass : public MachineFunctionPass {
public:
  static inline char ID = 0;

  X86KruglovCntPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    DebugLoc DebugLocation = MF.front().begin()->getDebugLoc();
    const TargetInstrInfo *InstrInfo = MF.getSubtarget().getInstrInfo();

    for (auto &MBB : MF) {
      size_t Count = 0;
      for (auto &MI : MBB) {
        Count++;
      }
        
      BuildMI(MBB, MBB.getFirstTerminator(), DebugLocation,
              InstrInfo->get(X86::ADD64ri32))
          .addImm(Count)
          .addExternalSymbol("ic");
    }
    return true;
  }
};

} // end anonymous namespace

static RegisterPass<X86KruglovCntPass>
    X("x86-kruglov-cnt-pass", "Instruction counter pass", false, false);