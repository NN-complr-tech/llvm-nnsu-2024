// RUN: %clang_cc1 -load %llvmshlibdir/PrintNamesPlugin%pluginext -plugin classprinter %s 1>&1 | FileCheck %s

class Test1 {
    int x;
    double y;
};
// CHECK: Test1
// CHECK-NEXT: |_x
// CHECK-NEXT: |_y

struct Test2 {
    float a;
    char b;
    bool c;
};
// CHECK: Test2
// CHECK-NEXT: |_a
// CHECK-NEXT: |_b
// CHECK-NEXT: |_c

template<typename T> class Test3 {
    T var;
};
// CHECK: Test3
// CHECK-NEXT: |_var

class Test4 {
    class Nested {
        int z;
    };
};
// CHECK: Test4
// CHECK: Nested
// CHECK-NEXT: |_z
