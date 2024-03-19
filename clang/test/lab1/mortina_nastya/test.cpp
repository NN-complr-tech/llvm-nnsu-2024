// RUN: %clang_cc1 -load %llvmshlibdir/PrintikClassPlugin%pluginext -plugin prin-elds %s 2>&1 | FileCheck %s

// CHECK: Empty
class Empty {};

// CHECK: MyClass
struct MyClass {
    // CHECK-NEXT: |_variable
    int variable;
};

// CHECK: Test
struct Test
{
    // CHECK-NEXT: |_A
    int A;
    // CHECK-NEXT: |_B
    int B;
};

// CHECK: TClass
template<typename T> class TClass {
    // CHECK-NEXT: |_TVar
    T TVar;
};


