// RUN: split-file %s %t

// RUN: %clang_cc1 -load %llvmshlibdir/LebedevaRenamePlugin%pluginext -add-plugin lebedeva-rename-plugin\
// RUN: -plugin-arg-lebedeva-rename-plugin OldName="A" -plugin-arg-lebedeva-rename-plugin NewName="B" %t/test1.cpp 
// RUN: FileCheck %s < %t/test1.cpp --check-prefix=FIRST-CHECK

// FIRST-CHECK: class B {
// FIRST-CHECK-NEXT: public:
// FIRST-CHECK-NEXT:     B() {};
// FIRST-CHECK-NEXT:     ~B();
// FIRST-CHECK-NEXT: };

//--- test1.cpp
class A {
public:
    A() {};
    ~A();
};

// RUN: %clang_cc1 -load %llvmshlibdir/LebedevaRenamePlugin%pluginext -add-plugin lebedeva-rename-plugin\
// RUN: -plugin-arg-lebedeva-rename-plugin OldName="C" -plugin-arg-lebedeva-rename-plugin NewName="D" %t/test2.cpp
// RUN: FileCheck %s < %t/test2.cpp --check-prefix=SECOND-CHECK

// SECOND-CHECK: class D {
// SECOND-CHECK-NEXT: private:
// SECOND-CHECK-NEXT:     int a;
// SECOND-CHECK-NEXT: public:
// SECOND-CHECK-NEXT:     D() {}
// SECOND-CHECK-NEXT:     D(int a): a(a) {}
// SECOND-CHECK-NEXT:     ~D();
// SECOND-CHECK-NEXT:     int getA() {
// SECOND-CHECK-NEXT:         return a;
// SECOND-CHECK-NEXT:     }
// SECOND-CHECK-NEXT: };
// SECOND-CHECK-NEXT: void func() {
// SECOND-CHECK-NEXT:     D* c = new D();
// SECOND-CHECK-NEXT:     delete c;
// SECOND-CHECK-NEXT: }
