#include "X86.h"
#include "X86InstrInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

namespace {

class X86MirzakhmedovCountPassIns : public MachineFunctionPass {
public:
  static char ID;
  X86MirzakhmedovCountPassIns() : MachineFunctionPass(ID) {}
  bool runOnMachineFunction(llvm::MachineFunction &machine_func) override;
};
} // namespace

char X86MirzakhmedovCountPassIns::ID = 0;

bool X86MirzakhmedovCountPassIns::runOnMachineFunction(
    llvm::MachineFunction &machine_func) {
  auto *glob_var = machine_func.getFunction().getParent()->getNamedGlobal("ic");
  if (!glob_var) {
    LLVMContext &con = machine_func.getFunction().getParent()->getContext();
    glob_var = new GlobalVariable(*machine_func.getFunction().getParent(),
                                  IntegerType::get(con, 64), false,
                                  GlobalValue::ExternalLinkage, nullptr, "ic");
    glob_var->setAlignment(Align(8));
    if (!glob_var) {
      return false;
    }
  }

  auto d_log = machine_func.front().begin()->getDebugLoc();
  auto *target_inst_inf = machine_func.getSubtarget().getInstrInfo();

  for (auto &b_block : machine_func) {
    int cnt = std::distance(b_block.begin(), b_block.end());
    auto location = b_block.getFirstTerminator();
    if (location != b_block.end() && location != b_block.begin() &&
        location->getOpcode() >= X86::JCC_1 &&
        location->getOpcode() <= X86::JCC_4) {
      --location;
    }

    BuildMI(b_block, location, d_log, target_inst_inf->get(X86::ADD64mi32))
        .addReg(0)
        .addImm(1)
        .addReg(0)
        .addGlobalAddress(glob_var)
        .addReg(0)
        .addImm(cnt);
  }
  return true;
}

static RegisterPass<X86MirzakhmedovCountPassIns>
    X("x86-mirzakhmedov-cnt-pass",
      "A pass that counts the number of X86 machine instructions", false,
      false);