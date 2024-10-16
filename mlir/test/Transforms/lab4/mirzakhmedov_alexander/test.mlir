// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/MirzakhmedovCountFCalls%shlibext --pass-pipeline="builtin.module(MirzakhmedovCountFCalls)" %s | FileCheck %s
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>} {
  llvm.func @_Z4baseii(%arg0: i32 {llvm.noundef}, %arg1: i32 {llvm.noundef}) -> (i32 {llvm.noundef}) attributes {passthrough = ["mustprogress", "noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    // CHECK: llvm.func @_Z4baseii(%arg0: i32 {llvm.noundef}, %arg1: i32 {llvm.noundef}) -> (i32 {llvm.noundef}) attributes {{.*}}"call-count" = 3 : i32
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %2 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %1 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.store %arg1, %2 {alignment = 4 : i64} : i32, !llvm.ptr
    %3 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
    %4 = llvm.load %2 {alignment = 4 : i64} : !llvm.ptr -> i32
    %5 = llvm.mul %3, %4  : i32
    llvm.return %5 : i32
  }
  llvm.func @_Z9usingbasev() -> (i32 {llvm.noundef}) attributes {passthrough = ["mustprogress", "noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    // CHECK: llvm.func @_Z9usingbasev() -> (i32 {llvm.noundef}) attributes {{.*}}"call-count" = 0 : i32
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(5 : i32) : i32
    %2 = llvm.mlir.constant(10 : i32) : i32
    %3 = llvm.mlir.constant(15 : i32) : i32
    %4 = llvm.mlir.constant(3 : i32) : i32
    %5 = llvm.mlir.constant(9 : i32) : i32
    %6 = llvm.mlir.constant(69 : i32) : i32
    %7 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %8 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %9 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %10 = llvm.call @_Z4baseii(%0, %1) : (i32, i32) -> i32
    llvm.store %10, %7 {alignment = 4 : i64} : i32, !llvm.ptr
    %11 = llvm.call @_Z4baseii(%2, %3) : (i32, i32) -> i32
    llvm.store %11, %8 {alignment = 4 : i64} : i32, !llvm.ptr
    %12 = llvm.call @_Z4baseii(%4, %5) : (i32, i32) -> i32
    llvm.store %12, %9 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.return %6 : i32
  }
  llvm.func @_Z6twentyv() -> (i32 {llvm.noundef}) attributes {passthrough = ["mustprogress", "noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    // CHECK: llvm.func @_Z6twentyv() -> (i32 {llvm.noundef}) attributes {{.*}}"call-count" = 1 : i32
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(5 : i32) : i32
    %2 = llvm.mlir.constant(20 : i32) : i32
    %3 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %1, %3 {alignment = 4 : i64} : i32, !llvm.ptr
    %4 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
    %5 = llvm.add %4, %2  : i32
    llvm.store %5, %3 {alignment = 4 : i64} : i32, !llvm.ptr
    %6 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
    llvm.return %6 : i32
  }
  llvm.func @_Z7nothingv() attributes {passthrough = ["mustprogress", "noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    // CHECK: llvm.func @_Z7nothingv() attributes {{.*}}"call-count" = 0 : i32
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %2 = llvm.call @_Z6twentyv() : () -> i32
    llvm.store %2, %1 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.return
  }
  llvm.func @_Z9retsmthngv() -> (f64 {llvm.noundef}) attributes {passthrough = ["mustprogress", "noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    // CHECK: llvm.func @_Z9retsmthngv() -> (f64 {llvm.noundef}) attributes {{.*}}"call-count" = 4 : i32,
    %0 = llvm.mlir.constant(2.500000e+00 : f64) : f64
    llvm.return %0 : f64
  }
  llvm.func @_Z13collectThreesv() attributes {passthrough = ["mustprogress", "noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    // CHECK: llvm.func @_Z13collectThreesv() attributes {{.*}}"call-count" = 0 : i32
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %2 = llvm.alloca %0 x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %3 = llvm.alloca %0 x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %4 = llvm.alloca %0 x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %5 = llvm.call @_Z9retsmthngv() : () -> f64
    llvm.store %5, %1 {alignment = 8 : i64} : f64, !llvm.ptr
    %6 = llvm.call @_Z9retsmthngv() : () -> f64
    llvm.store %6, %2 {alignment = 8 : i64} : f64, !llvm.ptr
    %7 = llvm.call @_Z9retsmthngv() : () -> f64
    llvm.store %7, %3 {alignment = 8 : i64} : f64, !llvm.ptr
    %8 = llvm.call @_Z9retsmthngv() : () -> f64
    llvm.store %8, %4 {alignment = 8 : i64} : f64, !llvm.ptr
    llvm.return
  }
}


