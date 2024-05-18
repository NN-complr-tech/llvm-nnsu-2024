// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/KulaevZhenyaFMAPass%shlibext --pass-pipeline="builtin.module(kulaev_zhenya_fma)" %s | FileCheck %s

// void func1(double c){
//     double a = 3 + c * 4;
// }

// void func2(double c, double b, double f){
//     double a = c + b * f;
// }

// void func3(double b, double f){
//     double a = 3 + b * f;
// }

// void func4(double с){
//     double a = 3 - 5 * с;
// }

// void func5(double с){
//     double a = 3 + 5 / с;
// }

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>} {
  llvm.func @_Z5func1d(%arg0: f64 {llvm.noundef}) attributes {passthrough = ["mustprogress", "noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(4.000000e+00 : f64) : f64
    %2 = llvm.mlir.constant(3.000000e+00 : f64) : f64
    %3 = llvm.alloca %0 x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %4 = llvm.alloca %0 x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %3 {alignment = 8 : i64} : f64, !llvm.ptr
    %5 = llvm.load %3 {alignment = 8 : i64} : !llvm.ptr -> f64
    %6 = llvm.fmul %5, %1  : f64
    %7 = llvm.fadd %2, %6  : f64
    // CHECK-NOT: %6 = llvm.fmul %5, %1  : f64
    // CHECK-NOT: %7 = llvm.fadd %2, %6  : f64
    // CHECK: %6 = llvm.intr.fma(%5, %1, %2)  : (f64, f64, f64) -> f64
    llvm.store %7, %4 {alignment = 8 : i64} : f64, !llvm.ptr
    llvm.return
  }
  llvm.func @_Z5func2ddd(%arg0: f64 {llvm.noundef}, %arg1: f64 {llvm.noundef}, %arg2: f64 {llvm.noundef}) attributes {passthrough = ["mustprogress", "noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %2 = llvm.alloca %0 x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %3 = llvm.alloca %0 x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %4 = llvm.alloca %0 x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %1 {alignment = 8 : i64} : f64, !llvm.ptr
    llvm.store %arg1, %2 {alignment = 8 : i64} : f64, !llvm.ptr
    llvm.store %arg2, %3 {alignment = 8 : i64} : f64, !llvm.ptr
    %5 = llvm.load %1 {alignment = 8 : i64} : !llvm.ptr -> f64
    %6 = llvm.load %2 {alignment = 8 : i64} : !llvm.ptr -> f64
    %7 = llvm.load %3 {alignment = 8 : i64} : !llvm.ptr -> f64
    %8 = llvm.fmul %6, %7  : f64
    %9 = llvm.fadd %5, %8  : f64
    // CHECK-NOT: %6 = llvm.fmul %5, %1  : f64
    // CHECK-NOT: %7 = llvm.fadd %2, %6  : f64
    // CHECK: %8 = llvm.intr.fma(%6, %7, %5)  : (f64, f64, f64) -> f64
    llvm.store %9, %4 {alignment = 8 : i64} : f64, !llvm.ptr
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
    %7 = llvm.fmul %5, %6  : f64
    %8 = llvm.fadd %1, %7  : f64
    // CHECK-NOT: %8 = llvm.intr.fma(%1, %4, %2)  : (f64, f64, f64) -> f64
    llvm.store %8, %4 {alignment = 8 : i64} : f64, !llvm.ptr
    llvm.return
  }
  llvm.func @_Z5func4d(%arg0: f64 {llvm.noundef}) attributes {passthrough = ["mustprogress", "noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(5.000000e+00 : f64) : f64
    %2 = llvm.mlir.constant(3.000000e+00 : f64) : f64
    %3 = llvm.alloca %0 x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %4 = llvm.alloca %0 x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %3 {alignment = 8 : i64} : f64, !llvm.ptr
    %5 = llvm.load %3 {alignment = 8 : i64} : !llvm.ptr -> f64
    %6 = llvm.fmul %1, %5  : f64
    %7 = llvm.fsub %2, %6  : f64
    // CHECK: %6 = llvm.fmul %1, %5  : f64
    // CHECK-NEXT: %7 = llvm.fsub %2, %6  : f64
    // CHECK-NOT: %8 = llvm.intr.fma(%1, %5, %6)  : (f64, f64, f64) -> f64
    llvm.store %7, %4 {alignment = 8 : i64} : f64, !llvm.ptr
    llvm.return
  }
  llvm.func @_Z5func5d(%arg0: f64 {llvm.noundef}) attributes {passthrough = ["mustprogress", "noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(5.000000e+00 : f64) : f64
    %2 = llvm.mlir.constant(3.000000e+00 : f64) : f64
    %3 = llvm.alloca %0 x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %4 = llvm.alloca %0 x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %3 {alignment = 8 : i64} : f64, !llvm.ptr
    %5 = llvm.load %3 {alignment = 8 : i64} : !llvm.ptr -> f64
    %6 = llvm.fdiv %1, %5  : f64
    %7 = llvm.fadd %2, %6  : f64
    // CHECK: %6 = llvm.fdiv %1, %5  : f64
    // CHECK-NEXT: %7 = llvm.fadd %2, %6  : f64
    // CHECK-NOT: %8 = llvm.intr.fma(%1, %5, %6)  : (f64, f64, f64) -> f64
    llvm.store %7, %4 {alignment = 8 : i64} : f64, !llvm.ptr
    llvm.return
  }
}
