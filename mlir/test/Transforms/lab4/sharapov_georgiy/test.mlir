// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/sCeilDiv%shlibext --pass-pipeline="builtin.module(sharapov_ceildiv)" %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>} {
  llvm.func @ceildivsi(%arg0: i32, %arg1: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    // CHECK: llvm.func @ceildivsi(%arg0: i32, %arg1: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    // CHECK-NEXT: %c1_i32 = arith.constant 1 : i32
    // CHECK-NEXT: %0 = arith.addi %arg0, %arg1 : i32
    // CHECK-NEXT: %1 = arith.subi %0, %c1_i32 : i32
    // CHECK-NEXT: %2 = arith.divsi %1, %arg1 : i32
    // CHECK-NOT: %0 = arith.ceildivsi %arg0, %arg1 : i32
    %0 = arith.ceildivsi %arg0, %arg1 : i32
    llvm.return %0 : i32
  }
  llvm.func @ceildivui(%arg0: i32, %arg1: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    // CHECK: llvm.func @ceildivui(%arg0: i32, %arg1: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    // CHECK-NEXT: %c1_i32 = arith.constant 1 : i32
    // CHECK-NEXT: %0 = arith.addi %arg0, %arg1 : i32
    // CHECK-NEXT: %1 = arith.subi %0, %c1_i32 : i32
    // CHECK-NEXT: %2 = arith.divui %1, %arg1 : i32
    // CHECK-NOT: %0 = arith.ceildivui %arg0, %arg1 : i32
    %0 = arith.ceildivui %arg0, %arg1 : i32
    llvm.return %0 : i32
  }
  llvm.func @example(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = llvm.call @ceildivsi(%arg0, %arg1) : (i32, i32) -> i32
    %1 = llvm.call @ceildivui(%arg2, %arg3) : (i32, i32) -> i32
    %2 = arith.addi %0, %1 : i32
    llvm.return %2 : i32
  }
}
