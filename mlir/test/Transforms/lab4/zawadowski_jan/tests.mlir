// RUN: split-file %s %t
// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/ZawadaMaxDepth%shlibext --pass-pipeline="builtin.module(func.func(ZawadaMaxDepth))" %t/funcDepth1.mlir | FileCheck %t/funcDepth1.mlir
// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/ZawadaMaxDepth%shlibext --pass-pipeline="builtin.module(func.func(ZawadaMaxDepth))" %t/funcDepth2.mlir | FileCheck %t/funcDepth2.mlir
// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/ZawadaMaxDepth%shlibext --pass-pipeline="builtin.module(func.func(ZawadaMaxDepth))" %t/funcDepth3.mlir | FileCheck %t/funcDepth3.mlir
// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/ZawadaMaxDepth%shlibext --pass-pipeline="builtin.module(func.func(ZawadaMaxDepth))" %t/funcDepth4.mlir | FileCheck %t/funcDepth4.mlir
// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/ZawadaMaxDepth%shlibext --pass-pipeline="builtin.module(func.func(ZawadaMaxDepth))" %t/funcDepth5.mlir | FileCheck %t/funcDepth5.mlir
// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/ZawadaMaxDepth%shlibext --pass-pipeline="builtin.module(func.func(ZawadaMaxDepth))" %t/funcDepth5WithLoop.mlir | FileCheck %t/funcDepth5WithLoop.mlir


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

//--- funcDepth4.mlir
func.func @funcDepth4() {
// CHECK: func.func @funcDepth4() attributes {maxDepth = 4 : i32}
    %cond = arith.constant 1 : i1
    %0 = scf.if %cond -> (i1) {
        %1 = scf.if %cond -> (i1) {
            %2 = scf.if %cond -> (i1) {
                scf.yield %cond : i1
            } else {
                scf.yield %cond : i1
            }
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

//--- funcDepth5.mlir
func.func @funcDepth5() {
// CHECK: func.func @funcDepth5() attributes {maxDepth = 5 : i32}
    %cond = arith.constant 1 : i1
    %0 = scf.if %cond -> (i1) {
        %1 = scf.if %cond -> (i1) {
            %2 = scf.if %cond -> (i1) {
                %3 = scf.if %cond -> (i1) {
                    scf.yield %cond : i1
                } else {
                    scf.yield %cond : i1
                }
                scf.yield %cond : i1
            } else {
                scf.yield %cond : i1
            }
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

//--- funcDepth5WithLoop.mlir
func.func @funcDepth5WithLoop() {
// CHECK: func.func @funcDepth5WithLoop() attributes {maxDepth = 5 : i32}
    %cond = arith.constant 1 : i1
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index

    scf.for %i = %c0 to %c10 step %c1 {
        %0 = scf.if %cond -> (i1) {
            %1 = scf.if %cond -> (i1) {
                %2 = scf.if %cond -> (i1) {
                    scf.yield %cond : i1
                } else {
                    scf.yield %cond : i1
                }
                scf.yield %cond : i1
            } else {
                scf.yield %cond : i1
            }
            scf.yield %cond : i1
        } else {
            scf.yield %cond : i1
        }
    }
    func.return
}