// RUN: split-file %s %t

// RUN: %clang_cc1 -load %llvmshlibdir/VeselRenamePlugin%pluginext\
// RUN: -add-plugin rename\
// RUN: -plugin-arg-rename cur-name=a\
// RUN: -plugin-arg-rename new-name=new_var %t/rename_var.cpp
// RUN: FileCheck %s < %t/rename_var.cpp --check-prefix=VAR

// VAR: int func() {
// VAR-NEXT: int new_var = 2, b = 2;
// VAR-NEXT: new_var = b + new_var;
// VAR-NEXT: new_var++;
// VAR-NEXT:  return new_var;
// VAR-NEXT: }

//--- rename_var.cpp
int func() {
  int a = 2, b = 2;
  a = b + a;
  a++;
  return a;
}

// RUN: %clang_cc1 -load %llvmshlibdir/VeselRenamePlugin%pluginext\
// RUN: -add-plugin rename\
// RUN: -plugin-arg-rename cur-name=c\
// RUN: -plugin-arg-rename new-name=new_var %t/rename_non_existent_var.cpp
// RUN: FileCheck %s < %t/rename_non_existent_var.cpp --check-prefix=NON_EXIST_VAR

// NON_EXIST_VAR: int func() {
// NON_EXIST_VAR-NEXT: int a = 2;
// NON_EXIST_VAR-NEXT: int b = 3;
// NON_EXIST_VAR-NEXT: b += a;
// NON_EXIST_VAR-NEXT: a++;
// NON_EXIST_VAR-NEXT:  return b - a;
// NON_EXIST_VAR-NEXT: }

//--- rename_non_existent_var.cpp
int func() {
  int a = 2;
  int b = 3;
  b += a;
  a++;
  return b - a;
}

// RUN: %clang_cc1 -load %llvmshlibdir/VeselRenamePlugin%pluginext\
// RUN: -add-plugin rename\
// RUN: -plugin-arg-rename cur-name=function\
// RUN: -plugin-arg-rename new-name=new_func %t/rename_func.cpp
// RUN: FileCheck %s < %t/rename_func.cpp --check-prefix=FUNC

// FUNC: int new_func(int param) {
// FUNC-NEXT: int a;
// FUNC-NEXT: a = 2;
// FUNC-NEXT: return param + a;
// FUNC-NEXT: }
// FUNC-NEXT: int other_func(){
// FUNC-NEXT: new_func(3);
// FUNC-NEXT: int a = new_func(2) + 3;
// FUNC-NEXT: return a;
// FUNC-NEXT: }

//--- rename_func.cpp
int function(int param) {
    int a;
    a = 2;
    return param + a;
}
int other_func(){
  function(3);
  int a = function(2) + 3;
  return a;
}

// RUN: %clang_cc1 -load %llvmshlibdir/VeselRenamePlugin%pluginext\
// RUN: -add-plugin rename\
// RUN: -plugin-arg-rename cur-name=function\
// RUN: -plugin-arg-rename new-name=f %t/rename_non_existent_func.cpp
// RUN: FileCheck %s < %t/rename_non_existent_func.cpp --check-prefix=NON_EXIST_FUNC

// NON_EXIST_FUNC: int func(int a) {
// NON_EXIST_FUNC-NEXT: int b = 2;
// NON_EXIST_FUNC-NEXT: return a + b;
// NON_EXIST_FUNC-NEXT: }
// NON_EXIST_FUNC: void func2() {
// NON_EXIST_FUNC-NEXT: int c = func(2);
// NON_EXIST_FUNC-NEXT: int b = func(c) + func(3);
// NON_EXIST_FUNC-NEXT: }

//--- rename_non_existent_func.cpp
int func(int a) {
  int b = 2;
  return a + b;
}

void func2() {
  int c = func(2);
  int b = func(c) + func(3);
}

// RUN: %clang_cc1 -load %llvmshlibdir/VeselRenamePlugin%pluginext\
// RUN: -add-plugin rename\
// RUN: -plugin-arg-rename cur-name=B\
// RUN: -plugin-arg-rename new-name=C %t/rename_non_existent_class.cpp
// RUN: FileCheck %s < %t/rename_non_existent_class.cpp --check-prefix=NON_EXIST_CLASS

// NON_EXIST_CLASS: class A{
// NON_EXIST_CLASS-NEXT: private:
// NON_EXIST_CLASS-NEXT: int var1;
// NON_EXIST_CLASS-NEXT: double var2;
// NON_EXIST_CLASS-NEXT: public:
// NON_EXIST_CLASS-NEXT: A() {};
// NON_EXIST_CLASS-NEXT: ~A() {};
// NON_EXIST_CLASS-NEXT: };
// NON_EXIST_CLASS: void func() {
// NON_EXIST_CLASS-NEXT: A var1;
// NON_EXIST_CLASS-NEXT: A* var2 = new A;
// NON_EXIST_CLASS-NEXT: delete var2;
// NON_EXIST_CLASS-NEXT: }

//--- rename_non_existent_class.cpp
class A{
 private:
  int var1;
  double var2;
 public:
 A() {};
 ~A() {};
};

void func() {
  A var1;
  A* var2 = new A;
  delete var2;
}
