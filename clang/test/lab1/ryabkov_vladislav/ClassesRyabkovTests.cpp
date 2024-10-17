// RUN: %clang_cc1 -load %llvmshlibdir/ClassesRyabkovLabOne%pluginext -plugin classes-ryabkov %s 2>&1 | FileCheck %s
// CHECK: Circle
class Circle
{
    // CHECK-NEXT: |_ PI
    const double PI = 3.141592;
    // CHECK-NEXT: |_ radius
    double radius;
};

// CHECK: Product
struct Product
{
    // CHECK-NEXT: |_ price
    unsigned int price;
    // CHECK-NEXT: |_ name
    char* name;
};

// CHECK: User
class User
{
    // CHECK-NEXT: |_ id
    int id;
public:
    // CHECK-NEXT: |_ age
    unsigned char age;
};

// CHECK: ComplexClass
struct ComplexClass
{
    // CHECK-NEXT: |_ floatingPoint
    const float floatingPoint;
    // CHECK-NEXT: |_ isSet
    static bool isSet;
};

// CHECK: outerClass
class outerClass {
    // CHECK-NEXT: |_ a
    int a;
    // CHECK-NEXT: |_ b
    char b;
    // CHECK-NEXT: innerClass
    class innerClass {
        //CHECK-NEXT: |_ var
        float var;
    };
};

// CHECK: Blank
struct Blank{};
 
// RUN: %clang_cc1 -load %llvmshlibdir/ClassesRyabkovLabOne%pluginext -plugin classes-ryabkov -plugin-arg-classes-ryabkov --help 1>&1 | FileCheck %s --check-prefix=HELP
 
// HELP: Classes and fields
