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
// CHECK: func.func @func4() attributes {maxDepth = 4 : i32}
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.mlir.constant(0 : i32) : i32
  %2 = llvm.mlir.constant(2 : i32) : i32
  %3 = llvm.mlir.constant(4 : i32) : i32
  %4 = llvm.mlir.constant(3 : i32) : i32
  %5 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  %6 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  %7 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  %8 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  llvm.store %arg0, %6 : i32, !llvm.ptr
  llvm.store %arg1, %7 : i32, !llvm.ptr
  llvm.store %arg2, %8 : i32, !llvm.ptr
  %9 = llvm.load %6 : !llvm.ptr -> i32
  %10 = llvm.icmp "sgt" %9, %1 : i32
  llvm.cond_br %10, ^bb1, ^bb6
^bb1:  // pred: ^bb0
  %11 = llvm.load %7 : !llvm.ptr -> i32
  %12 = llvm.icmp "sgt" %11, %1 : i32
  llvm.cond_br %12, ^bb2, ^bb5
^bb2:  // pred: ^bb1
  %13 = llvm.load %8 : !llvm.ptr -> i32
  %14 = llvm.icmp "sgt" %13, %1 : i32
  llvm.cond_br %14, ^bb3, ^bb4
^bb3:  // pred: ^bb2
  llvm.store %4, %5 : i32, !llvm.ptr
  llvm.br ^bb7
^bb4:  // pred: ^bb2
  llvm.store %3, %5 : i32, !llvm.ptr
  llvm.br ^bb7
^bb5:  // pred: ^bb1
  llvm.store %2, %5 : i32, !llvm.ptr
  llvm.br ^bb7
^bb6:  // pred: ^bb0
  llvm.store %0, %5 : i32, !llvm.ptr
  llvm.br ^bb7
^bb7:  // 4 preds: ^bb3, ^bb4, ^bb5, ^bb6
  %15 = llvm.load %5 : !llvm.ptr -> i32
  llvm.return %15 : i32
}

//--- func5.mlir
func.func @func5() {
// CHECK: func.func @func5() attributes {maxDepth = 3 : i32}
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.mlir.constant(2 : i32) : i32
  %2 = llvm.mlir.constant(0 : i32) : i32
  %3 = llvm.mlir.constant(-10 : i32) : i32
  %4 = llvm.mlir.constant(10 : i32) : i32
  %5 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  %6 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  %7 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  llvm.store %arg0, %6 : i32, !llvm.ptr
  llvm.store %arg1, %7 : i32, !llvm.ptr
  %8 = llvm.load %6 : !llvm.ptr -> i32
  %9 = llvm.srem %8, %1  : i32
  %10 = llvm.icmp "eq" %9, %2 : i32
  llvm.cond_br %10, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
  llvm.store %4, %5 : i32, !llvm.ptr
  llvm.br ^bb3
  ^bb2:  // pred: ^bb0
  llvm.store %3, %5 : i32, !llvm.ptr
  llvm.br ^bb3
  ^bb3:  // 2 preds: ^bb1, ^bb2
  %11 = llvm.load %5 : !llvm.ptr -> i32
  llvm.return %11 : i32
}
