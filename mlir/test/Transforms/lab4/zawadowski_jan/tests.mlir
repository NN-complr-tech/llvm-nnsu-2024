// RUN: split-file %s %t
// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/ZawadaMaxDepth%shlibext --pass-pipeline="builtin.module(func.func(ZawadaMaxDepth))" %t/funcDepth1.mlir | FileCheck %t/funcDepth1.mlir
// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/ZawadaMaxDepth%shlibext --pass-pipeline="builtin.module(func.func(ZawadaMaxDepth))" %t/funcDepth2.mlir | FileCheck %t/funcDepth2.mlir
// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/ZawadaMaxDepth%shlibext --pass-pipeline="builtin.module(func.func(ZawadaMaxDepth))" %t/funcDepth3.mlir | FileCheck %t/funcDepth3.mlir

//--- funcDepth1.mlir
func.func @funcDepth1() {
// CHECK: func.func @funcDepth1() attributes {maxDepth = 1 : i32}
  func.return
}

//--- funcDepth2.mlir
func.func @funcDepth2() {
// CHECK: func.func @funcDepth2() attributes {maxDepth = 2 : i32}
    %cond = arith.constant 1 : i1
    %0 = scf.if %cond -> (i1) {
        scf.yield %cond : i1
    } else {
        scf.yield %cond : i1
    }
    func.return
}

//--- funcDepth3.mlir
func.func @funcDepth3() {
// CHECK: func.func @funcDepth3() attributes {maxDepth = 3 : i32}
    %cond = arith.constant 1 : i1
    %0 = scf.if %cond -> (i1) {
        %1 = scf.if %cond -> (i1) {
            scf.yield %cond : i1
        } else {
            scf.yield %cond : i1
        }
        scf.yield %cond : i1
    } else {
        scf.yield %cond : i1
    }
    func.return
}