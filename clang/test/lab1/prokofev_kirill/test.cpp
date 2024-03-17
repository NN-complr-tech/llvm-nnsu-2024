// RUN: split-file %s %t
// RUN: %clang_cc1 -load %llvmshlibdir/RenamePlugin%pluginext -add-plugin renameplug\
// RUN: -plugin-arg-renameplug oldName="Alpha"\
// RUN: -plugin-arg-renameplug newName="Beta" %t/class_test.cpp
// RUN: FileCheck %s < %t/class_test.cpp --check-prefix=CLASS

// CLASS: class Beta {
// CLASS-NEXT: public:
// CLASS-NEXT:   Beta() {};
// CLASS-NEXT:    ~Beta();
// CLASS-NEXT:};

//--- class_test.cpp
class Alpha {
public:
  Alpha(){};
  ~Alpha();
};

// RUN: %clang_cc1 -load %llvmshlibdir/RenamePlugin%pluginext -add-plugin renameplug\
// RUN: -plugin-arg-renameplug oldName="Foo"\
// RUN: -plugin-arg-renameplug newName="Function" %t/function_test.cpp
// RUN: FileCheck %s < %t/function_test.cpp --check-prefix=FUNCTION

// FUNCTION: int Function(int digit) {
// FUNCTION-NEXT:     return digit * 2;
// FUNCTION-NEXT: }

//--- function_test.cpp
int Foo(int digit) { return digit * 2; };

// RUN: %clang_cc1 -load %llvmshlibdir/RenamePlugin%pluginext -add-plugin renameplug\
// RUN: -plugin-arg-renameplug oldName="oldVar"\
// RUN: -plugin-arg-renameplug newName="newVar" %t/variable_test.cpp
// RUN: FileCheck %s < %t/variable_test.cpp --check-prefix=VARIABLE

// VARIABLE: int Foo(int newVar) {
// VARIABLE-NEXT:     return newVar * 2;
// VARIABLE-NEXT: }

//--- variable_test.cpp
int Foo(int oldVar) { return oldVar * 2; }

