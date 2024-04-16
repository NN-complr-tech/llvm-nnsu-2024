#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/Register.h"
#include <map>

#define AVOIDCALL_DESC "X86 Kulikov FMA"
#define AVOIDCALL_NAME "x86-kulikov-fma"

#define DEBUG_TYPE AVOIDCALL_NAME

using namespace llvm;
// X86MaddToMPass
namespace {
class X86KulikovFMAPass : public MachineFunctionPass {
public:
  X86KulikovFMAPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  static char ID;

private:
  StringRef getPassName() const override { return AVOIDCALL_DESC; }
};
} // namespace

char X86KulikovFMAPass::ID = 0;

FunctionPass *llvm::createX86KulikovFMAPass() { return new X86KulikovFMAPass(); }

INITIALIZE_PASS(X86KulikovFMAPass, AVOIDCALL_NAME, AVOIDCALL_DESC, false, false)

bool X86KulikovFMAPass::runOnMachineFunction(MachineFunction &MF) {
  bool Modified = false;
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();

  for (auto &MBB : MF) {
    std::map<llvm::Register, std::pair<MachineInstr *, MachineInstr *>> useMap;
    for (auto I = MBB.begin(); I != MBB.end(); I++) {
      if (I->getOpcode() == X86::MULPDrr) {
        useMap.insert({I->getOperand(0).getReg(), {&(*I), nullptr}});
      } else {
        for (unsigned i = 1; i < I->getNumOperands(); i++) {
          llvm::MachineOperand &MO = I->getOperand(i);
          if (MO.isReg() && MO.getReg().isVirtual()) {
            auto r = MO.getReg();
            auto fnd = useMap.find(r);
            if (fnd != useMap.end()) {
              if (fnd->second.second) {
                useMap.erase(r);
              } else {
                useMap.insert_or_assign(
                    r, std::make_pair(fnd->second.first, &(*I)));
              }
            }
          }
        }
      }
    }
    for (auto u : useMap) {
      if (u.second.second && u.second.second->getOpcode() == X86::ADDPDrr) {
        auto &MulInstr = *u.second.first; // a = b * c;
        auto &AddInstr =
            *u.second.second; // d = e(a) + f | d = e + f(a);  => d = b * c + e

        auto a = MulInstr.getOperand(0).getReg();
        auto b = MulInstr.getOperand(1).getReg();
        auto c = MulInstr.getOperand(2).getReg();
        auto d = AddInstr.getOperand(0).getReg();
        auto e = AddInstr.getOperand(1).getReg();
        auto f = AddInstr.getOperand(2).getReg();
        if (e == a) {
          e = f;
        }

        MIMetadata MIMD(AddInstr);
        BuildMI(MBB, AddInstr, MIMD, TII->get(X86::VFMADD213PDr)) // or X86::VFMADDPD4rr
            .addReg(d, RegState::Define)
            .addReg(b)
            .addReg(c)
            .addReg(e);
        MulInstr.eraseFromParent();
        AddInstr.eraseFromParent();
        Modified = true;
      }
    }
  }

  return Modified;
}
