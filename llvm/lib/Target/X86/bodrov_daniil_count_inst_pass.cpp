#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/Register.h"
#include <map>
using namespace llvm;

#define DEBUG_TYPE "count-instructions"

namespace {
    struct CountInstructions : public MachineFunctionPass {
        static char ID;
        CountInstructions() : MachineFunctionPass(ID) {}

        bool runOnMachineFunction(MachineFunction &MF) override {
            unsigned long InstrCount = 0;
            for (auto &MBB : MF) {
                for (auto &MI : MBB) {
                    // Exclude debug instructions
                    if (!MI.isDebugInstr())
                        ++InstrCount;
                }
            }

            errs() << "Number of machine instructions executed in function '"
                   << MF.getName() << "': " << InstrCount << "\n";
            return false;
        }
    };
} // end anonymous namespace

char CountInstructions::ID = 0;

using llvm::PassRegistry;
using llvm::PassInfo;
using llvm::callDefaultCtor;

void llvm::initializeCountInstructionsPass(llvm::PassRegistry &);

INITIALIZE_PASS(CountInstructions, "count-instructions", "Count machine instructions executed during function execution", false, false);

FunctionPass *llvm::createCountInstructions() {
    return new CountInstructions();
}

