// RUN: %clang_cc1 -ast-dump -ast-dump-filter testFun -load %llvmshlibdir/AlwaysInlinePlugin%pluginext -add-plugin always-inlines-plugin %s 2>&1 | FileCheck %s

// CHECK: FunctionDecl {{.*}} testFunIf 'int \(int, int\)'
int testFunIf(int A, int B) {
  if (A < B) {
    A = B;
  }
  return 0;
}
// CHECK-NOT: `-AlwaysInlineAttr {{.*}} Implicit always_inline`

// CHECK: FunctionDecl {{.*}} testFunWhile 'void \(int, int\)'
void testFunWhile(int A, int B) {
  {
    while(A < B) {
      A = B;
    }
  }
}
// CHECK-NOT: `-AlwaysInlineAttr {{.*}} Implicit always_inline`

// CHECK: FunctionDecl {{.*}} testFunEmpty 'int \(\)'
int testFunEmpty() {}

// CHECK: FunctionDecl {{.*}} testFunFor 'int \(int\)'
int testFunFor(int A) {
  for (int i = 0; i < 0; i++){

  }

  return A;
}
// CHECK-NOT: `-AlwaysInlineAttr {{.*}} Implicit always_inline`

// CHECK: FunctionDecl {{.*}} testFunSwitch 'int \(int\)'
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
// CHECK-NOT: `-AlwaysInlineAttr {{.*}} Implicit always_inline`