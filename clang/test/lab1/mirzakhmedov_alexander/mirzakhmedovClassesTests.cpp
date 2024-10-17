// RUN: %clang_cc1 -load %llvmshlibdir/MirzakhmedovLabaOnePlugin%pluginext -plugin mirzakhmedov-classes-print %s 2>&1 | FileCheck %s
// CHECK: Point
class Point
{
    // CHECK-NEXT: |_ x
    int x;
    // CHECK-NEXT: |_ y
    int y;
};

// CHECK: Person
struct Person
{
    // CHECK-NEXT: |_ age
    unsigned char age;
    // CHECK-NEXT: |_ name
    char* name;
};

// CHECK: Car
class Car
{
    // CHECK-NEXT: |_ cnt
    static int cnt;
public:
    // CHECK-NEXT: |_ mode
    const int mode = 2;
};

// CHECK: House
struct House
{
    // CHECK-NEXT: |_ h
    double h;
    // CHECK-NEXT: |_ w
    double w;
};

// CHECK: Vehicle
class Vehicle
{
public:
    // CHECK-NEXT: |_ speed
    float speed;
    // CHECK-NEXT: |_ fuelType
    char fuelType;
};

// CHECK: Classout
class Classout {
    // CHECK-NEXT: |_ some
    int some;
    // CHECK-NEXT: |_ thing
    char thing;
    // CHECK-NEXT: nstdClass
    class nstdClass {
        //CHECK-NEXT: |_ x
        float x;
    };
};

template<typename T>
// CHECK: Tree
class Tree{
    // CHECK-NEXT: |_ n
    T n;
};

// CHECK: Empty
class Empty{};

// RUN: %clang_cc1 -load %llvmshlibdir/MirzakhmedovLabaOnePlugin%pluginext -plugin mirzakhmedov-classes-print -plugin-arg-mirzakhmedov-classes-print --help 1>&1 | FileCheck %s --check-prefix=HELP

// HELP: Display names of classes.
