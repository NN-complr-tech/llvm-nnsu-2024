define dso_local noundef <2 x double> @_Z12muladd_test1Dv2_dS_S_(<2 x double> noundef %a, <2 x double> noundef %b, <2 x double> noundef %c) {
entry:
  %a.addr = alloca <2 x double>, align 16
  %b.addr = alloca <2 x double>, align 16
  %c.addr = alloca <2 x double>, align 16
  store <2 x double> %a, ptr %a.addr, align 16
  store <2 x double> %b, ptr %b.addr, align 16
  store <2 x double> %c, ptr %c.addr, align 16
  %0 = load <2 x double>, ptr %a.addr, align 16
  %1 = load <2 x double>, ptr %b.addr, align 16
  %2 = load <2 x double>, ptr %c.addr, align 16
  %3 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %0, <2 x double> %1, <2 x double> %2)
  ret <2 x double> %3
}

declare <2 x double> @llvm.fmuladd.v2f64(<2 x double>, <2 x double>, <2 x double>)

define dso_local noundef <2 x double> @_Z12muladd_test2Dv2_dS_S_(<2 x double> noundef %a, <2 x double> noundef %b, <2 x double> noundef %c) {
entry:
  %a.addr = alloca <2 x double>, align 16
  %b.addr = alloca <2 x double>, align 16
  %c.addr = alloca <2 x double>, align 16
  %tmp = alloca <2 x double>, align 16
  store <2 x double> %a, ptr %a.addr, align 16
  store <2 x double> %b, ptr %b.addr, align 16
  store <2 x double> %c, ptr %c.addr, align 16
  %0 = load <2 x double>, ptr %a.addr, align 16
  %1 = load <2 x double>, ptr %b.addr, align 16
  %mul = fmul <2 x double> %0, %1
  store <2 x double> %mul, ptr %tmp, align 16
  %2 = load <2 x double>, ptr %tmp, align 16
  %3 = load <2 x double>, ptr %c.addr, align 16
  %add = fadd <2 x double> %2, %3
  store <2 x double> %add, ptr %tmp, align 16
  %4 = load <2 x double>, ptr %tmp, align 16
  ret <2 x double> %4
}

define dso_local noundef <2 x double> @_Z12muladd_test3Dv2_dS_S_(<2 x double> noundef %a, <2 x double> noundef %b, <2 x double> noundef %c) {
entry:
  %a.addr = alloca <2 x double>, align 16
  %b.addr = alloca <2 x double>, align 16
  %c.addr = alloca <2 x double>, align 16
  %tmp = alloca <2 x double>, align 16
  store <2 x double> %a, ptr %a.addr, align 16
  store <2 x double> %b, ptr %b.addr, align 16
  store <2 x double> %c, ptr %c.addr, align 16
  %0 = load <2 x double>, ptr %a.addr, align 16
  %1 = load <2 x double>, ptr %c.addr, align 16
  %2 = load <2 x double>, ptr %b.addr, align 16
  %3 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %0, <2 x double> %1, <2 x double> %2)
  store <2 x double> %3, ptr %tmp, align 16
  %4 = load <2 x double>, ptr %tmp, align 16
  %5 = load <2 x double>, ptr %c.addr, align 16
  %6 = load <2 x double>, ptr %b.addr, align 16
  %7 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %4, <2 x double> %5, <2 x double> %6)
  ret <2 x double> %7
}