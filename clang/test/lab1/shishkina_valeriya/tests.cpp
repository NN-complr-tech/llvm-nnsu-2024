// RUN: %clang_cc1 -load %llvmshlibdir/PrintClassesNamePlugin%pluginext -plugin print-classes %s 1>&1 | FileCheck %s

// CHECK: test1
struct test1
{
    // CHECK-NEXT: |_ a
    int a;
    // CHECK-NEXT: |_ b
    int b;
};

// CHECK: test2
class test2
{
    // CHECK-NEXT: |_ arr
    double arr;
    // CHECK-NEXT: |_ b
    const int b = 2;
};

// CHECK: test3
class test3
{
    // CHECK-NEXT: |_ arg
    static int arg;
public:
    // CHECK-NEXT: |_ brr
    int brr = 2;
};

// CHECK: test4
struct test4{};