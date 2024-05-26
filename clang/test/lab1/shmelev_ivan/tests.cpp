// RUN: %clang_cc1 -load %llvmshlibdir/ShmelevPrintNamesPlugin%pluginext -plugin classexplorer %s 2>&1 | FileCheck %s


class foo1 {
  int a;
  double b;
  char c;
};
// CHECK: foo1
// CHECK-NEXT: |_a
// CHECK-NEXT: |_b
// CHECK-NEXT: |_c

struct foo2 {
  float a;
  char b;
  bool c;
};
// CHECK: foo2
// CHECK-NEXT: |_a
// CHECK-NEXT: |_b
// CHECK-NEXT: |_c

template <typename T> class foo3 { T var; };
// CHECK: foo3
// CHECK-NEXT: |_var

class foo4 {
  class inside {
    int z;
  };
};
// CHECK: foo4
// CHECK: inside
// CHECK-NEXT: |_z
