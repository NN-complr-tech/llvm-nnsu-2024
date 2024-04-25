#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

namespace {
class X86ShishkinaMICounterPass : public MachineFunctionPass {
public:
    static char ID;
    
    X86ShishkinaMICounterPass() : MachineFunctionPass(ID) {}

    bool runOnMachineFunction(MachineFunction &MF) override {
        const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
        DebugLoc DL3 = MF.front().begin()->getDebugLoc();

        Module &M = *MF.getFunction().getParent();
        GlobalVariable *gvar = M.getGlobalVariable("ic");

        if (!gvar) {
            LLVMContext &context = M.getContext();
            gvar = new GlobalVariable(M, IntegerType::get(context, 64), false,
                                      GlobalValue::ExternalLinkage, nullptr, "ic");
            gvar->setAlignment(Align(8));
        }

        // createGlobalICVariable(MF);

        for (auto &MBB : MF) {
            unsigned count = 0;
            for (auto &MI : MBB) {
                if (!MI.isDebugInstr())
                    ++count;
            }

            BuildMI(MBB, MBB.getFirstTerminator(), DL3, TII->get(X86::ADD64ri32))
                .addGlobalAddress(gvar, 0, X86II::MO_NO_FLAG)
                .addImm(count);
        }

        return true;
    }

private:
    void createGlobalICVariable(MachineFunction &MF) {
        // Module &M = *MF.getFunction().getParent();
        // GlobalVariable *gvar = M.getGlobalVariable("ic");

        // if (!gvar) {
        //     LLVMContext &context = M.getContext();
        //     gvar = new GlobalVariable(M, IntegerType::get(context, 64), false,
        //                               GlobalValue::ExternalLinkage, nullptr, "ic");
        //     gvar->setAlignment(Align(8));
        // }
    }
};
} // end anonymous namespace

char X86ShishkinaMICounterPass::ID = 0;

INITIALIZE_PASS(X86ShishkinaMICounterPass, "x86-shishkina-mi-counter", "X86 Count number of machine instructions pass",
                false, false)