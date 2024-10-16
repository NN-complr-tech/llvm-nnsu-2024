// RUN: split-file %s %t
// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/ShipitsinDepthPass%shlibext --pass-pipeline="builtin.module(func.func(shipitsin-depth-pass))" %t/one.mlir | FileCheck %t/one.mlir
// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/ShipitsinDepthPass%shlibext --pass-pipeline="builtin.module(func.func(shipitsin-depth-pass))" %t/two.mlir | FileCheck %t/two.mlir
// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/ShipitsinDepthPass%shlibext --pass-pipeline="builtin.module(func.func(shipitsin-depth-pass))" %t/three.mlir | FileCheck %t/three.mlir

//--- one.mlir
module {
  // CHECK: func.func @one() attributes {maxDepth = 3 : i32} {
  func.func @one() {
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
}

//--- two.mlir
module {
  // CHECK: func.func @two() attributes {maxDepth = 4 : i32} {
  func.func @two() {
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
}

//--- three.mlir
module {
  // CHECK: func.func @three() attributes {maxDepth = 2 : i32} {
  func.func @three() {
    %cond = arith.constant 1 : i1
    %0 = scf.if %cond -> (i1) {
      scf.yield %cond : i1
    } else {
      scf.yield %cond : i1
    }
    func.return
  }
}