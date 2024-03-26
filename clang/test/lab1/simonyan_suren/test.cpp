// RUN: %clang_cc1 -ast-dump -ast-dump-filter testFun -load %llvmshlibdir/AlwaysInlinePlugin%pluginext -add-plugin always-inlines-plugin %s 2>&1 | FileCheck %s

// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+test\.cpp:([0-9]+:[0-9]|[0-9]+), (line|col):([0-9]+:[0-9]|[0-9]+)> (line|col):([0-9]+:[0-9]|[0-9]+) testFunIf 'int \(int, int\)'}}
int testFunIf(int A, int B) {
  if (A < B) {
    A = B;
  }
  return 0;
}
// CHECK-NOT: `-AlwaysInlineAttr {{0[xX][0-9a-fA-F]+ <(line|col):([0-9]+:[0-9]|[0-9]+)> Implicit always_inline}}

// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+test\.cpp:([0-9]+:[0-9]|[0-9]+), (line|col):([0-9]+:[0-9]|[0-9]+)> (line|col):([0-9]+:[0-9]|[0-9]+) testFunWhile 'void \(int, int\)'}}
void testFunWhile(int A, int B) {
  {
    while(A < B) {
      A = B;
    }
  }
}
// CHECK-NOT: `-AlwaysInlineAttr {{0[xX][0-9a-fA-F]+ <(line|col):([0-9]+:[0-9]|[0-9]+)> Implicit always_inline}}

// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+test\.cpp:([0-9]+:[0-9]|[0-9]+), (line|col):([0-9]+:[0-9]|[0-9]+)> (line|col):([0-9]+:[0-9]|[0-9]+) testFunEmpty 'int \(\)'}}
int testFunEmpty() {}

// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+test\.cpp:([0-9]+:[0-9]|[0-9]+), (line|col):([0-9]+:[0-9]|[0-9]+)> (line|col):([0-9]+:[0-9]|[0-9]+) testFunFor 'int \(int\)'}}
int testFunFor(int A) {
  for (int i = 0; i < 0; i++){

  }

  return A;
}
// CHECK-NOT: `-AlwaysInlineAttr {{0[xX][0-9a-fA-F]+ <(line|col):([0-9]+:[0-9]|[0-9]+)> Implicit always_inline}}

// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+test\.cpp:([0-9]+:[0-9]|[0-9]+), (line|col):([0-9]+:[0-9]|[0-9]+)> (line|col):([0-9]+:[0-9]|[0-9]+) testFunSwitch 'int \(int\)'}}
int testFunSwitch(int A) {

    switch (A)
    {
    case 1:
      break;

    default:
      break;
    }
  return A;
}
// CHECK-NOT: `-AlwaysInlineAttr {{0[xX][0-9a-fA-F]+ <(line|col):([0-9]+:[0-9]|[0-9]+)> Implicit always_inline}}

