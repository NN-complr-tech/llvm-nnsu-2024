#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

using namespace llvm;

namespace {

    class X86PivovarovMICounter : public MachineFunctionPass {
    public:
        static char ID;

        X86PivovarovMICounter() : MachineFunctionPass(ID) {}

        bool runOnMachineFunction(MachineFunction& machineFunction) override {
            DebugLoc debugLoc = machineFunction.front().begin()->getDebugLoc();
            const TargetInstrInfo* targetInstrInfo =
                machineFunction.getSubtarget().getInstrInfo();
            Module& module = *machineFunction.getFunction().getParent();
            GlobalVariable* globalCounter = module.getGlobalVariable("ic");

            for (auto& basicBlock : machineFunction) {
                MachineBasicBlock::iterator insertPos = basicBlock.begin();
                while (insertPos != basicBlock.end() &&
                    (insertPos->isPHI() || insertPos->isDebugValue())) {
                    ++insertPos;
                }

                size_t instructionCount =
                    std::distance(basicBlock.begin(), basicBlock.end());

                BuildMI(basicBlock, insertPos, debugLoc,
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

    char X86PivovarovMICounter::ID = 0;

}

static RegisterPass<X86PivovarovMICounter>
X("x86-pivovarov-mi-counter", "X86 Count of machine instructions pass", false,
    false);
