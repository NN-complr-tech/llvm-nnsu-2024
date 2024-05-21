// RUN: split-file %s %t
// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/dephCounter%shlibext --pass-pipeline="builtin.module(func.func(VasilevDepthCounter))" %t/one.mlir | FileCheck %t/one.mlir

//--- one.mlir
// CHECK: max_region_depth = 1
func.func @one() {
  func.return
}

//--- two.mlir
// CHECK: max_region_depth = 2
func.func @two() {
    %cond = arith.constant 1 : i1
    %0 = scf.if %cond -> (i1) {
        scf.yield %cond : i1
    } else {
        scf.yield %cond : i1
    }
    func.return
}

//--- three.mlir
// CHECK: max_region_depth = 3
func.func @three() {
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
