// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/NoginFusedMultAddPass%shlibext --pass-pipeline="builtin.module(fused-mult-add)" %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>} {
  llvm.func @_Z5func1dd(%arg0: f64 {llvm.noundef}, %arg1: f64 {llvm.noundef}) attributes {passthrough = ["mustprogress", "noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(3.000000e+00 : f64) : f64
    %2 = llvm.alloca %0 x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %3 = llvm.alloca %0 x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %4 = llvm.alloca %0 x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %2 {alignment = 8 : i64} : f64, !llvm.ptr
    llvm.store %arg1, %3 {alignment = 8 : i64} : f64, !llvm.ptr
    %5 = llvm.load %2 {alignment = 8 : i64} : !llvm.ptr -> f64
    %6 = llvm.fmul %5, %1  : f64
    %7 = llvm.load %3 {alignment = 8 : i64} : !llvm.ptr -> f64
    %8 = llvm.fadd %6, %7  : f64
    // CHECK: %5 = llvm.load %2 {alignment = 8 : i64} : !llvm.ptr -> f64
    // CHECK-NEXT: %6 = llvm.load %3 {alignment = 8 : i64} : !llvm.ptr -> f64
    // CHECK-NEXT: %7 = llvm.intr.fma(%5, %1, %6)  : (f64, f64, f64) -> f64
    // CHECK-NOT: %6 = llvm.fmul %5, %1  : f64
    // CHECK-NOT: %8 = llvm.fadd %6, %7  : f64
    llvm.store %8, %4 {alignment = 8 : i64} : f64, !llvm.ptr
    llvm.return
  }
  llvm.func @_Z5func2dd(%arg0: f64 {llvm.noundef}, %arg1: f64 {llvm.noundef}) attributes {passthrough = ["mustprogress", "noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(3.000000e+00 : f64) : f64
    %2 = llvm.alloca %0 x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %3 = llvm.alloca %0 x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %4 = llvm.alloca %0 x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %2 {alignment = 8 : i64} : f64, !llvm.ptr
    llvm.store %arg1, %3 {alignment = 8 : i64} : f64, !llvm.ptr
    %5 = llvm.load %2 {alignment = 8 : i64} : !llvm.ptr -> f64
    %6 = llvm.fmul %5, %1  : f64
    %7 = llvm.load %3 {alignment = 8 : i64} : !llvm.ptr -> f64
    %8 = llvm.fsub %6, %7  : f64
    // CHECK: %5 = llvm.load %2 {alignment = 8 : i64} : !llvm.ptr -> f64
    // CHECK-NEXT: %6 = llvm.fmul %5, %1  : f64
    // CHECK-NEXT: %7 = llvm.load %3 {alignment = 8 : i64} : !llvm.ptr -> f64
    // CHECK-NEXT: %8 = llvm.fsub %6, %7  : f64
    // CHECK-NOT: llvm.intr.fma
    llvm.store %8, %4 {alignment = 8 : i64} : f64, !llvm.ptr
    llvm.return
  }
  llvm.func @_Z5func3dd(%arg0: f64 {llvm.noundef}, %arg1: f64 {llvm.noundef}) attributes {passthrough = ["mustprogress", "noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(3.000000e+00 : f64) : f64
    %2 = llvm.alloca %0 x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %3 = llvm.alloca %0 x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %4 = llvm.alloca %0 x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %2 {alignment = 8 : i64} : f64, !llvm.ptr
    llvm.store %arg1, %3 {alignment = 8 : i64} : f64, !llvm.ptr
    %5 = llvm.load %2 {alignment = 8 : i64} : !llvm.ptr -> f64
    %6 = llvm.load %3 {alignment = 8 : i64} : !llvm.ptr -> f64
    %7 = llvm.fmul %1, %6  : f64
    %8 = llvm.fadd %5, %7  : f64
    // CHECK: %5 = llvm.load %2 {alignment = 8 : i64} : !llvm.ptr -> f64
    // CHECK-NEXT: %6 = llvm.load %3 {alignment = 8 : i64} : !llvm.ptr -> f64
    // CHECK-NEXT: %7 = llvm.intr.fma(%1, %6, %5)  : (f64, f64, f64) -> f64
    // CHECK-NOT: %7 = llvm.fmul %1, %6  : f64
    // CHECK-NOT: %8 = llvm.fadd %5, %7  : f64
    llvm.store %8, %4 {alignment = 8 : i64} : f64, !llvm.ptr
    llvm.return
  }
  llvm.func @_Z5func4dd(%arg0: f64 {llvm.noundef}, %arg1: f64 {llvm.noundef}) attributes {passthrough = ["mustprogress", "noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(3.000000e+00 : f64) : f64
    %2 = llvm.mlir.constant(6.000000e+00 : f64) : f64
    %3 = llvm.alloca %0 x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %4 = llvm.alloca %0 x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %5 = llvm.alloca %0 x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %3 {alignment = 8 : i64} : f64, !llvm.ptr
    llvm.store %arg1, %4 {alignment = 8 : i64} : f64, !llvm.ptr
    %6 = llvm.load %3 {alignment = 8 : i64} : !llvm.ptr -> f64
    %7 = llvm.fmul %6, %1  : f64
    %8 = llvm.load %4 {alignment = 8 : i64} : !llvm.ptr -> f64
    %9 = llvm.fsub %7, %8  : f64
    %10 = llvm.fadd %9, %2  : f64
    // CHECK: %6 = llvm.load %3 {alignment = 8 : i64} : !llvm.ptr -> f64
    // CHECK-NEXT: %7 = llvm.fmul %6, %1  : f64
    // CHECK-NEXT: %8 = llvm.load %4 {alignment = 8 : i64} : !llvm.ptr -> f64
    // CHECK-NEXT: %9 = llvm.fsub %7, %8  : f64
    // CHECK-NEXT: %10 = llvm.fadd %9, %2  : f64
    // CHECK-NOT: llvm.intr.fma
    llvm.store %10, %5 {alignment = 8 : i64} : f64, !llvm.ptr
    llvm.return
  }
}
