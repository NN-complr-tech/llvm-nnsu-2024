// RUN: split-file %s %t
// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/BalyasovMaxDepPass%shlibext --pass-pipeline="builtin.module(func.func(BalyasovMaxDepPass))" %t/func1.mlir | FileCheck %t/func1.mlir
// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/BalyasovMaxDepPass%shlibext --pass-pipeline="builtin.module(func.func(BalyasovMaxDepPass))" %t/func2.mlir | FileCheck %t/func2.mlir
// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/BalyasovMaxDepPass%shlibext --pass-pipeline="builtin.module(func.func(BalyasovMaxDepPass))" %t/func3.mlir | FileCheck %t/func3.mlir
// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/BalyasovMaxDepPass%shlibext --pass-pipeline="builtin.module(func.func(BalyasovMaxDepPass))" %t/func4.mlir | FileCheck %t/func4.mlir
// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/BalyasovMaxDepPass%shlibext --pass-pipeline="builtin.module(func.func(BalyasovMaxDepPass))" %t/func5.mlir | FileCheck %t/func5.mlir

//--- func1.mlir
func.func @func1(%arg0: i32) -> i32 {
// CHECK: func.func @func1(%arg0: i32) -> i32 attributes {maxDepth = 1 : i32}
  %0 = arith.muli %arg0, %arg0 : i32
  func.return %0 : i32
}

//--- func2.mlir
func.func @func2() {
// CHECK: func.func @func2() attributes {maxDepth = 2 : i32}
  %cond = arith.constant 1 : i1
    %0 = scf.if %cond -> (i1) {
        scf.yield %cond : i1
    } else {
        scf.yield %cond : i1
    }
    func.return
}

//--- func3.mlir
func.func @func3() {
// CHECK: func.func @func3() attributes {maxDepth = 3 : i32}
  %c3_i32 = arith.constant 3 : i32
  %c5_i32 = arith.constant 5 : i32
  %0 = arith.constant 0 : i32
  %1 = arith.constant 1 : i32
  %2 = arith.subi %c5_i32, %1 : i32
  %3 = arith.subi %c3_i32, %1 : i32
  %4 = arith.cmpi sgt, %c5_i32, %0 : i32
  scf.if %4 {
    %5 = arith.cmpi sgt, %c3_i32, %0 : i32
    scf.if %5 {
      %6 = arith.subi %c5_i32, %1 : i32
      %7 = arith.subi %3, %1 : i32
    }
    %8 = arith.subi %3, %1 : i32
  }
  func.return
}

//--- func4.mlir
func.func @func4() {
// CHECK: func.func @func4(%arg0: i32, %arg1: i32, %arg2: i32) attributes {maxDepth = 4 : i32}
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32
  %cmp_a = arith.cmpi "sgt", %arg0, %c0 : i32
  %cmp_b = arith.cmpi "sgt", %arg1, %c0 : i32
  %cmp_c = arith.cmpi "sgt", %arg2, %c0 : i32
  scf.if %cmp_a {
    scf.if %cmp_b {
      scf.if %cmp_c {
        func.return %c3 : i32  
      } else {
        func.return %c4 : i32  
      }
    } else {
      func.return %c2 : i32  
    }
  } 
  else {
    func.return %c1 : i32 
  }
  func.return %c0 : i32
}
