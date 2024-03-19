// RUN: %clang_cc1 -load %llvmshlibdir/PrintClassesPlugin%pluginext -plugin print-classes %s 1>&1 | FileCheck %s

// CHECK: Empty
class Empty {};

// CHECK: MyStruct
struct MyStruct{
    // CHECK-NEXT: |_x
    int x;
    // CHECK-NEXT: |_y
    int y;
};

// CHECK: MyClass
class MyClass {
    // CHECK-NEXT: |_a_c
    int x_c;
    // CHECK-NEXT: |_b_c
    float y_c;
    // CHECK-NEXT: |_c_c
    double c_c;
};

// CHECK: outerClass
class outerClass {
    // CHECK: innerClass
    class innerClass {
        //CHECK-NEXT: |_value
        float value;
    };
};

// CHECK: TemplateClass
template<typename T> class TemplateClass {
    //CHECK-NEXT: Tvalue
    T Tvalue;
};
