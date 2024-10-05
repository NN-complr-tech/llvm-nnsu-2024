// RUN: split-file %s %t

// RUN: %clang_cc1 -load %llvmshlibdir/KanakovRenameInd%pluginext -add-plugin kanakov-rename-plugin\
// RUN: -plugin-arg-kanakov-rename-plugin nameOld_="A" -plugin-arg-kanakov-rename-plugin nameNew_="B" %t/testClass.cpp 
// RUN: FileCheck %s < %t/testClass.cpp --check-prefix=FIRST-CHECK

// FIRST-CHECK: class B {
// FIRST-CHECK-NEXT: public:
// FIRST-CHECK-NEXT:     B() {}
// FIRST-CHECK-NEXT:     ~B();
// FIRST-CHECK-NEXT: };

//--- testClass.cpp
class A {
public:
  A() {}
  ~A();
};

// RUN: %clang_cc1 -load %llvmshlibdir/KanakovRenameInd%pluginext -add-plugin kanakov-rename-plugin\
// RUN: -plugin-arg-kanakov-rename-plugin nameOld_="func" -plugin-arg-kanakov-rename-plugin nameNew_="func1" %t/testFunc.cpp
// RUN: FileCheck %s < %t/testFunc.cpp --check-prefix=SECOND-CHECK

// SECOND-CHECK: void func1() {
// SECOND-CHECK-NEXT:    return;
// SECOND-CHECK-NEXT: }


//--- testFunc.cpp
void func() {
  return;
}

// RUN: %clang_cc1 -load %llvmshlibdir/KanakovRenameInd%pluginext -add-plugin kanakov-rename-plugin\
// RUN: -plugin-arg-kanakov-rename-plugin nameOld_="var" -plugin-arg-kanakov-rename-plugin nameNew_="newVar" %t/testVar.cpp
// RUN: FileCheck %s < %t/testVar.cpp --check-prefix=THIRD-CHECK

// THIRD-CHECK: int newVar = 0;

//--- testVar.cpp
int var = 0;

// RUN: %clang_cc1 -load %llvmshlibdir/KanakovRenameInd%pluginext -add-plugin kanakov-rename-plugin\
// RUN: -plugin-arg-kanakov-rename-plugin nameOld_="var" -plugin-arg-kanakov-rename-plugin nameNew_="newVar" %t/testVarFunc.cpp
// RUN: FileCheck %s < %t/testVarFunc.cpp --check-prefix=FOURTH-CHECK

// FOURTH-CHECK: void foo(int newVar) {}

//--- testVarFunc.cpp
void foo(int var) {}

// RUN: %clang_cc1 -load %llvmshlibdir/KanakovRenameInd%pluginext -add-plugin kanakov-rename-plugin\
// RUN: -plugin-arg-kanakov-rename-plugin nameOld_="A" -plugin-arg-kanakov-rename-plugin nameNew_="B" %t/testNew.cpp
// RUN: FileCheck %s < %t/testNew.cpp --check-prefix=FIFTH-CHECK

// FIFTH-CHECK: struct B {};
// FIFTH-CHECK-NEXT: void a() {
// FIFTH-CHECK-NEXT:    B* a = new B;
// FIFTH-CHECK-NEXT:    delete a;
// FIFTH-CHECK-NEXT: }

//--- testNew.cpp
struct A {};
void a() {
  A* a = new A;
  delete a;
}

// RUN: %clang_cc1 -load %llvmshlibdir/KanakovRenameInd%pluginext -add-plugin kanakov-rename-plugin\
// RUN: -plugin-arg-kanakov-rename-plugin nameOld_="var" -plugin-arg-kanakov-rename-plugin nameNew_="newVar" %t/testVarPtr.cpp
// RUN: FileCheck %s < %t/testVarPtr.cpp --check-prefix=SIXTH-CHECK

// SIXTH-CHECK: int a = 0;
// SIXTH-CHECK-NEXT: int* newVar = &a;

//--- testVarPtr.cpp
int a = 0;
int* var = &a;

// RUN: %clang_cc1 -load %llvmshlibdir/KanakovRenameInd%pluginext -add-plugin kanakov-rename-plugin\
// RUN: -plugin-arg-kanakov-rename-plugin nameOld_="var" -plugin-arg-kanakov-rename-plugin nameNew_="newVar" %t/testVarRef.cpp
// RUN: FileCheck %s < %t/testVarRef.cpp --check-prefix=SEVENTH-CHECK

// SEVENTH-CHECK: int a = 0;
// SEVENTH-CHECK-NEXT: int& newVar = a;

//--- testVarRef.cpp
int a = 0;
int& var = a;

// RUN: %clang_cc1 -load %llvmshlibdir/KanakovRenameInd%pluginext -add-plugin kanakov-rename-plugin\
// RUN: -plugin-arg-kanakov-rename-plugin nameOld_="var" -plugin-arg-kanakov-rename-plugin nameNew_="newVar" %t/testNotVar.cpp
// RUN: FileCheck %s < %t/testNotVar.cpp --check-prefix=EIGHTH-CHECK

// EIGHTH-CHECK: int a = 0;

//--- testNotVar.cpp
int a = 0;

// RUN: %clang_cc1 -load %llvmshlibdir/KanakovRenameInd%pluginext -add-plugin kanakov-rename-plugin\
// RUN: -plugin-arg-kanakov-rename-plugin nameOld_="A" -plugin-arg-kanakov-rename-plugin nameNew_="B" %t/testNotFunc.cpp
// RUN: FileCheck %s < %t/testNotFunc.cpp --check-prefix=NINETH-CHECK

// NINETH-CHECK: void function() {}

//--- testNotFunc.cpp
void function() {}