// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/AlexseevMulAddMergePass%shlibext --pass-pipeline="builtin.module(alexseev_mul_add_merge)" %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>} {
  // double c = a * 6.0 + b;
  llvm.func local_unnamed_addr @foo1(%arg0: f64 {llvm.noundef}, %arg1: f64 {llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", "nounwind", "willreturn", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(6.000000e+00 : f64) : f64
    // CHECK-NOT:  %1 = llvm.fmul %arg0, %0  : f64
    // CHECK-NOT:  %2 = llvm.fadd %1, %arg1  : f64
    // CHECK:      %1 = llvm.intr.fma(%arg0, %0, %arg1)  : (f64, f64, f64) -> f64
    %1 = llvm.fmul %arg0, %0  : f64
    %2 = llvm.fadd %1, %arg1  : f64
    llvm.return
  }
  
  // double c = a + 10.0 * b;
  llvm.func local_unnamed_addr @foo2(%arg0: f64 {llvm.noundef}, %arg1: f64 {llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", "nounwind", "willreturn", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(10.000000e+00 : f64) : f64
    // CHECK-NOT:  %1 = llvm.fmul %arg1, %0  : f64
    // CHECK-NOT:  %2 = llvm.fadd %1, %arg0  : f64
    // CHECK:      %1 = llvm.intr.fma(%arg1, %0, %arg0)  : (f64, f64, f64) -> f64
    %1 = llvm.fmul %arg1, %0  : f64
    %2 = llvm.fadd %1, %arg0  : f64
    llvm.return
  }
  
  //     double c = a * b;
  //     double d = c + e;
  //     double f = c + x;
  llvm.func local_unnamed_addr @foo3(%arg0: f64 {llvm.noundef}, %arg1: f64 {llvm.noundef}, %arg2: f64 {llvm.noundef}, %arg3: f64 {llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", "nounwind", "willreturn", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    // CHECK-NOT: %0 = llvm.fmul %arg0, %arg1  : f64
    // CHECK-NOT: %1 = llvm.fadd %0, %arg2  : f64
    // CHECK:     %0 = llvm.intr.fma(%arg0, %arg1, %arg2)  : (f64, f64, f64) -> f64
    %0 = llvm.fmul %arg0, %arg1  : f64
    %1 = llvm.fadd %0, %arg2  : f64
    // CHECK-NOT: %2 = llvm.fadd %0, %arg3  : f64
    // CHECK:     %1 = llvm.intr.fma(%arg0, %arg1, %arg3)  : (f64, f64, f64) -> f64
    %2 = llvm.fadd %0, %arg3  : f64
    llvm.return
  }
}