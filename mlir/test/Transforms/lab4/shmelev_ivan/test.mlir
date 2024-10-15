// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/cntCallFuncShmelev%shlibext --pass-pipeline="builtin.module(cntCallFuncShmelev)" %s | FileCheck %s
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>} {
  llvm.func @_Z6simplev() -> (i32 {llvm.noundef}) attributes {passthrough = ["mustprogress", "noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    // CHECK: llvm.func @_Z6simplev() -> (i32 {llvm.noundef}) attributes {{.*}}cnt_call = 11 : i32
    %0 = llvm.mlir.constant(5 : i32) : i32
    llvm.return %0 : i32
  }
  llvm.func @_Z12discriminantddd(%arg0: f64 {llvm.noundef}, %arg1: f64 {llvm.noundef}, %arg2: f64 {llvm.noundef}) -> (f64 {llvm.noundef}) attributes {passthrough = ["mustprogress", "noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    // CHECK: llvm.func @_Z12discriminantddd(%arg0: f64 {llvm.noundef}, %arg1: f64 {llvm.noundef}, %arg2: f64 {llvm.noundef}) -> (f64 {llvm.noundef}) attributes {{.*}}cnt_call = 2 : i32
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(4.000000e+00 : f64) : f64
    %2 = llvm.alloca %0 x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %3 = llvm.alloca %0 x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %4 = llvm.alloca %0 x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %2 {alignment = 8 : i64} : f64, !llvm.ptr
    llvm.store %arg1, %3 {alignment = 8 : i64} : f64, !llvm.ptr
    llvm.store %arg2, %4 {alignment = 8 : i64} : f64, !llvm.ptr
    %5 = llvm.load %3 {alignment = 8 : i64} : !llvm.ptr -> f64
    %6 = llvm.load %3 {alignment = 8 : i64} : !llvm.ptr -> f64
    %7 = llvm.load %2 {alignment = 8 : i64} : !llvm.ptr -> f64
    %8 = llvm.fmul %1, %7  : f64
    %9 = llvm.load %4 {alignment = 8 : i64} : !llvm.ptr -> f64
    %10 = llvm.fmul %8, %9  : f64
    %11 = llvm.fneg %10  : f64
    %12 = llvm.intr.fmuladd(%5, %6, %11)  : (f64, f64, f64) -> f64
    llvm.return %12 : f64
  }
  llvm.func @_Z18checkDiscriminantsv() -> (i1 {llvm.noundef, llvm.zeroext}) attributes {passthrough = ["mustprogress", "noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    // CHECK: llvm.func @_Z18checkDiscriminantsv() -> (i1 {llvm.noundef, llvm.zeroext}) attributes {{.*}}cnt_call = 1 : i32
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(2.000000e+00 : f64) : f64
    %2 = llvm.mlir.constant(-1.000000e+00 : f64) : f64
    %3 = llvm.mlir.constant(-5.500000e+00 : f64) : f64
    %4 = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %5 = llvm.mlir.constant(4.000000e+00 : f64) : f64
    %6 = llvm.mlir.constant(3.000000e+00 : f64) : f64
    %7 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %8 = llvm.mlir.constant(false) : i1
    %9 = llvm.mlir.constant(true) : i1
    %10 = llvm.alloca %0 x i1 {alignment = 1 : i64} : (i32) -> !llvm.ptr
    %11 = llvm.alloca %0 x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %12 = llvm.alloca %0 x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %13 = llvm.call @_Z12discriminantddd(%1, %2, %3) : (f64, f64, f64) -> f64
    llvm.store %13, %11 {alignment = 8 : i64} : f64, !llvm.ptr
    %14 = llvm.call @_Z12discriminantddd(%4, %5, %6) : (f64, f64, f64) -> f64
    llvm.store %14, %12 {alignment = 8 : i64} : f64, !llvm.ptr
    %15 = llvm.load %11 {alignment = 8 : i64} : !llvm.ptr -> f64
    %16 = llvm.fcmp "oge" %15, %7 : f64
    llvm.cond_br %16, ^bb1, ^bb3
  ^bb1:  // pred: ^bb0
    %17 = llvm.load %12 {alignment = 8 : i64} : !llvm.ptr -> f64
    %18 = llvm.fcmp "oge" %17, %7 : f64
    llvm.cond_br %18, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    llvm.store %9, %10 {alignment = 1 : i64} : i1, !llvm.ptr
    llvm.br ^bb4
  ^bb3:  // 2 preds: ^bb0, ^bb1
    llvm.store %8, %10 {alignment = 1 : i64} : i1, !llvm.ptr
    llvm.br ^bb4
  ^bb4:  // 2 preds: ^bb2, ^bb3
    %19 = llvm.load %10 {alignment = 1 : i64} : !llvm.ptr -> i1
    llvm.return %19 : i1
  }
  llvm.func @_Z7someonev() -> (i32 {llvm.noundef}) attributes {passthrough = ["mustprogress", "noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    // CHECK: llvm.func @_Z7someonev() -> (i32 {llvm.noundef}) attributes {{.*}}cnt_call = 0 : i32
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(30 : i32) : i32
    %2 = llvm.alloca %0 x i8 {alignment = 1 : i64} : (i32) -> !llvm.ptr
    %3 = llvm.call @_Z18checkDiscriminantsv() : () -> i1
    %4 = llvm.zext %3 : i1 to i8
    llvm.store %4, %2 {alignment = 1 : i64} : i8, !llvm.ptr
    llvm.return %1 : i32
  }
  llvm.func @_Z10operationsv() -> (i32 {llvm.noundef}) attributes {passthrough = ["mustprogress", "noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    // CHECK: llvm.func @_Z10operationsv() -> (i32 {llvm.noundef}) attributes {{.*}}cnt_call = 1 : i32
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(40 : i32) : i32
    %2 = llvm.mlir.constant(90 : i32) : i32
    %3 = llvm.mlir.constant(10 : i32) : i32
    %4 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %5 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %6 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %7 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %1, %4 {alignment = 4 : i64} : i32, !llvm.ptr
    %8 = llvm.load %4 {alignment = 4 : i64} : !llvm.ptr -> i32
    %9 = llvm.add %8, %2  : i32
    llvm.store %9, %5 {alignment = 4 : i64} : i32, !llvm.ptr
    %10 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> i32
    %11 = llvm.load %4 {alignment = 4 : i64} : !llvm.ptr -> i32
    %12 = llvm.add %10, %11  : i32
    %13 = llvm.add %12, %3  : i32
    llvm.store %13, %6 {alignment = 4 : i64} : i32, !llvm.ptr
    %14 = llvm.load %4 {alignment = 4 : i64} : !llvm.ptr -> i32
    %15 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> i32
    %16 = llvm.add %14, %15  : i32
    %17 = llvm.load %6 {alignment = 4 : i64} : !llvm.ptr -> i32
    %18 = llvm.add %16, %17  : i32
    llvm.store %18, %7 {alignment = 4 : i64} : i32, !llvm.ptr
    %19 = llvm.load %7 {alignment = 4 : i64} : !llvm.ptr -> i32
    llvm.return %19 : i32
  }
  llvm.func @_Z18lots_of_challengesv() attributes {passthrough = ["mustprogress", "noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    // CHECK: llvm.func @_Z18lots_of_challengesv() attributes {{.*}}cnt_call = 0 : i32
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %2 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %3 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %4 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %5 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %6 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %7 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %8 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %9 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %10 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %11 = llvm.call @_Z6simplev() : () -> i32
    llvm.store %11, %1 {alignment = 4 : i64} : i32, !llvm.ptr
    %12 = llvm.call @_Z6simplev() : () -> i32
    llvm.store %12, %2 {alignment = 4 : i64} : i32, !llvm.ptr
    %13 = llvm.call @_Z6simplev() : () -> i32
    llvm.store %13, %3 {alignment = 4 : i64} : i32, !llvm.ptr
    %14 = llvm.call @_Z6simplev() : () -> i32
    llvm.store %14, %4 {alignment = 4 : i64} : i32, !llvm.ptr
    %15 = llvm.call @_Z6simplev() : () -> i32
    llvm.store %15, %5 {alignment = 4 : i64} : i32, !llvm.ptr
    %16 = llvm.call @_Z6simplev() : () -> i32
    llvm.store %16, %6 {alignment = 4 : i64} : i32, !llvm.ptr
    %17 = llvm.call @_Z6simplev() : () -> i32
    llvm.store %17, %7 {alignment = 4 : i64} : i32, !llvm.ptr
    %18 = llvm.call @_Z6simplev() : () -> i32
    llvm.store %18, %8 {alignment = 4 : i64} : i32, !llvm.ptr
    %19 = llvm.call @_Z6simplev() : () -> i32
    llvm.store %19, %9 {alignment = 4 : i64} : i32, !llvm.ptr
    %20 = llvm.call @_Z6simplev() : () -> i32
    llvm.store %20, %10 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.return
  }
  llvm.func @_Z8anythingv() -> (i32 {llvm.noundef}) attributes {passthrough = ["mustprogress", "noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    // CHECK: llvm.func @_Z8anythingv() -> (i32 {llvm.noundef}) attributes {{.*}}cnt_call = 0 : i32
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %2 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %3 = llvm.call @_Z6simplev() : () -> i32
    llvm.store %3, %1 {alignment = 4 : i64} : i32, !llvm.ptr
    %4 = llvm.call @_Z10operationsv() : () -> i32
    llvm.store %4, %2 {alignment = 4 : i64} : i32, !llvm.ptr
    %5 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
    %6 = llvm.load %2 {alignment = 4 : i64} : !llvm.ptr -> i32
    %7 = llvm.mul %5, %6  : i32
    llvm.return %7 : i32
  }
}
