// RUN: split-file %s %t

// RUN: %clang_cc1 -load %llvmshlibdir/RenamePlugin%pluginext\
// RUN: -add-plugin rename\
// RUN: -plugin-arg-rename type=var\
// RUN: -plugin-arg-rename cur-name=a\
// RUN: -plugin-arg-rename new-name=new_var %t/rename_var.cpp
// RUN: diff %t/rename_var_expected.cpp %t/rename_var.cpp

// RUN: %clang_cc1 -load %llvmshlibdir/RenamePlugin%pluginext\
// RUN: -add-plugin rename\
// RUN: -plugin-arg-rename type=var\
// RUN: -plugin-arg-rename cur-name=c\
// RUN: -plugin-arg-rename new-name=new_var %t/rename_non_existent_var.cpp
// RUN: diff %t/rename_non_existent_var_expected.cpp %t/rename_non_existent_var.cpp

// RUN: %clang_cc1 -load %llvmshlibdir/RenamePlugin%pluginext\
// RUN: -add-plugin rename\
// RUN: -plugin-arg-rename type=func\
// RUN: -plugin-arg-rename cur-name=function\
// RUN: -plugin-arg-rename new-name=new_func %t/rename_func.cpp
// RUN: diff %t/rename_func_expected.cpp %t/rename_func.cpp

// RUN: %clang_cc1 -load %llvmshlibdir/RenamePlugin%pluginext\
// RUN: -add-plugin rename\
// RUN: -plugin-arg-rename type=func\
// RUN: -plugin-arg-rename cur-name=function\
// RUN: -plugin-arg-rename new-name=f %t/rename_non_existent_func.cpp
// RUN: diff %t/rename_non_existent_func_expected.cpp %t/rename_non_existent_func.cpp

// RUN: %clang_cc1 -load %llvmshlibdir/RenamePlugin%pluginext\
// RUN: -add-plugin rename\
// RUN: -plugin-arg-rename type=class\
// RUN: -plugin-arg-rename cur-name=Base\
// RUN: -plugin-arg-rename new-name=SimpleClass %t/rename_class.cpp
// RUN: diff %t/rename_class_expected.cpp %t/rename_class.cpp

// RUN: %clang_cc1 -load %llvmshlibdir/RenamePlugin%pluginext\
// RUN: -add-plugin rename\
// RUN: -plugin-arg-rename type=class\
// RUN: -plugin-arg-rename cur-name=B\
// RUN: -plugin-arg-rename new-name=C %t/rename_non_existent_class.cpp
// RUN: diff %t/rename_non_existent_class_expected.cpp %t/rename_non_existent_class.cpp

// RUN: %clang_cc1 -load %llvmshlibdir/RenamePlugin%pluginext\
// RUN: -add-plugin rename\
// RUN: -plugin-arg-rename help\
// RUN: 2>&1 | FileCheck %s --check-prefix=HELP

// RUN: not %clang_cc1 -load %llvmshlibdir/RenamePlugin%pluginext\
// RUN: -add-plugin rename\
// RUN: -plugin-arg-rename cur-name=B\
// RUN: -plugin-arg-rename new-name=C\
// RUN: 2>&1 | FileCheck %s --check-prefix=ERROR

// RUN: not %clang_cc1 -load %llvmshlibdir/RenamePlugin%pluginext\
// RUN: -add-plugin rename\
// RUN: -plugin-arg-rename cur-name=B\
// RUN: -plugin-arg-rename new-name=C\
// RUN: -plugin-arg-rename param=val\
// RUN: 2>&1 | FileCheck %s --check-prefix=ERROR

// RUN: not %clang_cc1 -load %llvmshlibdir/RenamePlugin%pluginext\
// RUN: -add-plugin rename\
// RUN: -plugin-arg-rename type=undefined\
// RUN: -plugin-arg-rename cur-name=B\
// RUN: -plugin-arg-rename new-name=C\
// RUN: 2>&1 | FileCheck %s --check-prefix=ERROR

// RUN: not %clang_cc1 -load %llvmshlibdir/RenamePlugin%pluginext\
// RUN: -add-plugin rename\
// RUN: 2>&1 | FileCheck %s --check-prefix=ERROR

// HELP: Specify three required arguments:
// HELP: -plugin-arg-rename type=["var", "func", "class"]
// HELP: -plugin-arg-rename cur-name="Current identifier name"
// HELP: -plugin-arg-rename new-name="New identifier name"

//ERROR: Invalid arguments
//ERROR: Specify "-plugin-arg-rename help" for usage

//--- rename_var.cpp
int func() {
  int a = 2, b = 2;
  a = b + a;
  a++;
  return a;
}
//--- rename_var_expected.cpp
int func() {
  int new_var = 2, b = 2;
  new_var = b + new_var;
  new_var++;
  return new_var;
}
//--- rename_non_existent_var.cpp
int func() {
  int a = 2;
  int b = 3;
  b += a;
  a++;
  return b - a;
}
//--- rename_non_existent_var_expected.cpp
int func() {
  int a = 2;
  int b = 3;
  b += a;
  a++;
  return b - a;
}
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
//--- rename_func_expected.cpp
int new_func(int param) {
    int a;
    a = 2;
    return param + a;
}
int other_func(){
  new_func(3);
  int a = new_func(2) + 3;
  return a;
}
//--- rename_non_existent_func.cpp
int func(int a) {
  int b = 2;
  return a + b;
}

void func2() {
  int c = func(2);
  int b = func(c) + func(3);
}
//--- rename_non_existent_func_expected.cpp
int func(int a) {
  int b = 2;
  return a + b;
}

void func2() {
  int c = func(2);
  int b = func(c) + func(3);
}
//--- rename_class.cpp
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
//--- rename_class_expected.cpp
class SimpleClass{
 private:
  int a;
  int b;
 public:
  SimpleClass() {}
  SimpleClass(int a, int b): a(a), b(b) {}
  ~SimpleClass();
};

void func() {
  SimpleClass a;
  SimpleClass* var = new SimpleClass(1, 2);
  delete var;
}
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
//--- rename_non_existent_class_expected.cpp
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
