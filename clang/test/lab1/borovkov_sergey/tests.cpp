// RUN: split-file %s %t

//--- rename_var.cpp

// RUN: %clang_cc1 -load %llvmshlibdir/renamedfilePlugin%pluginext\
// RUN: -add-plugin rename\
// RUN: -plugin-arg-rename type=var\
// RUN: -plugin-arg-rename cur-name=a\
// RUN: -plugin-arg-rename new-name=new_var %t/rename_var.cpp
// RUN: FileCheck %s < %t/rename_var.cpp --check-prefix=VAR

// VAR: int func() {
// VAR-NEXT: int new_var = 2, b = 2;
// VAR-NEXT: new_var = b + new_var;
// VAR-NEXT: new_var++;
// VAR-NEXT:  return new_var;
// VAR-NEXT: }


int func() {
  int a = 2, b = 2;
  a = b + a;
  a++;
  return a;
}

//--- rename_non_existent_var.cpp

// RUN: %clang_cc1 -load %llvmshlibdir/renamedfilePlugin%pluginext\
// RUN: -add-plugin rename\
// RUN: -plugin-arg-rename type=var\
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


int func() {
  int a = 2;
  int b = 3;
  b += a;
  a++;
  return b - a;
}

//--- rename_func.cpp

// RUN: %clang_cc1 -load %llvmshlibdir/renamedfilePlugin%pluginext\
// RUN: -add-plugin rename\
// RUN: -plugin-arg-rename type=func\
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

//--- rename_non_existent_func.cpp

// RUN: %clang_cc1 -load %llvmshlibdir/renamedfilePlugin%pluginext\
// RUN: -add-plugin rename\
// RUN: -plugin-arg-rename type=func\
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


int func(int a) {
  int b = 2;
  return a + b;
}

void func2() {
  int c = func(2);
  int b = func(c) + func(3);
}

//--- rename_class.cpp

// RUN: %clang_cc1 -load %llvmshlibdir/renamedfilePlugin%pluginext\
// RUN: -add-plugin rename\
// RUN: -plugin-arg-rename type=class\
// RUN: -plugin-arg-rename cur-name=Base\
// RUN: -plugin-arg-rename new-name=SimpleClass %t/rename_class.cpp
// RUN: FileCheck %s < %t/rename_class.cpp --check-prefix=CLASS

// CLASS: class SimpleClass{
// CLASS-NEXT: private:
// CLASS-NEXT: int a;
// CLASS-NEXT: int b;
// CLASS-NEXT: public:
// CLASS-NEXT: SimpleClass() {}
// CLASS-NEXT: SimpleClass(int a, int b): a(a), b(b) {}
// CLASS-NEXT: ~SimpleClass();
// CLASS-NEXT: };
// CLASS: void func() {
// CLASS-NEXT: SimpleClass a;
// CLASS-NEXT: SimpleClass* var = new SimpleClass(1, 2);
// CLASS-NEXT: delete var;
// CLASS-NEXT: }


class Base{
 private:
  int a;
  int b;
 public:
  Base() {}
  Base(int a, int b): a(a), b(b) {}
  ~Base();
};

void func() {
  Base a;
  Base* var = new Base(1, 2);
  delete var;
}

//--- rename_non_existent_class.cpp

// RUN: %clang_cc1 -load %llvmshlibdir/renamedfilePlugin%pluginext\
// RUN: -add-plugin rename\
// RUN: -plugin-arg-rename type=class\
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

// RUN: %clang_cc1 -load %llvmshlibdir/renamedfilePlugin%pluginext\
// RUN: -add-plugin rename\
// RUN: -plugin-arg-rename help\
// RUN: 2>&1 | FileCheck %s --check-prefix=HELP

// HELP: Specify three required arguments:
// HELP-NEXT: -plugin-arg-rename type=["var", "func", "class"]
// HELP-NEXT: -plugin-arg-rename cur-name="Current identifier name"
// HELP-NEXT: -plugin-arg-rename new-name="New identifier name"

// RUN: not %clang_cc1 -load %llvmshlibdir/renamedfilePlugin%pluginext\
// RUN: -add-plugin rename\
// RUN: -plugin-arg-rename cur-name=B\
// RUN: -plugin-arg-rename new-name=C\
// RUN: 2>&1 | FileCheck %s --check-prefix=ERROR

// RUN: not %clang_cc1 -load %llvmshlibdir/renamedfilePlugin%pluginext\
// RUN: -add-plugin rename\
// RUN: -plugin-arg-rename cur-name=B\
// RUN: -plugin-arg-rename new-name=C\
// RUN: -plugin-arg-rename param=val\
// RUN: 2>&1 | FileCheck %s --check-prefix=ERROR

// RUN: not %clang_cc1 -load %llvmshlibdir/renamedfilePlugin%pluginext\
// RUN: -add-plugin rename\
// RUN: -plugin-arg-rename type=undefined\
// RUN: -plugin-arg-rename cur-name=B\
// RUN: -plugin-arg-rename new-name=C\
// RUN: 2>&1 | FileCheck %s --check-prefix=ERROR

// RUN: not %clang_cc1 -load %llvmshlibdir/renamedfilePlugin%pluginext\
// RUN: -add-plugin rename\
// RUN: 2>&1 | FileCheck %s --check-prefix=ERROR

//ERROR: Invalid arguments
//ERROR-NEXT: Specify "-plugin-arg-rename help" for usage

