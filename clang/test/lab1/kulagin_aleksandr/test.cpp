// RUN: %clang_cc1 -ast-dump -ast-dump-filter foo -load %llvmshlibdir/kulaginAlwaysInline%pluginext -add-plugin always-inline %s | FileCheck %s
// COM: %clang_cc1 -load %llvmshlibdir/kulaginAlwaysInline%pluginext -add-plugin always-inline %s -emit-llvm -o - | FileCheck %s

// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+test\.cpp:[0-9]+:[0-9]+, line:[0-9]+:[0-9]+> line:[0-9]+:[0-9]+ foo1 'void \(\)'}}
void __attribute__((always_inline)) foo1() {
  return;
}
// CHECK: `-AlwaysInlineAttr {{0[xX][0-9a-fA-F]+ <line:[0-9]+:[0-9]+> always_inline}}

// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+test\.cpp:[0-9]+:[0-9]+, line:[0-9]+:[0-9]+> line:[0-9]+:[0-9]+ foo2 'void \(\)'}}
void foo2() {

}
// CHECK: `-AlwaysInlineAttr {{0[xX][0-9a-fA-F]+ <line:[0-9]+:[0-9]+> always_inline}}

// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+test\.cpp:[0-9]+:[0-9]+, line:[0-9]+:[0-9]+> line:[0-9]+:[0-9]+ foo3 'void \(\)'}}
void foo3() {
  if (true) {
    int c;
  }
}
// CHECK-NOT: `-AlwaysInlineAttr {{0[xX][0-9a-fA-F]+ <line:[0-9]+:[0-9]+> always_inline}}

// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+test\.cpp:[0-9]+:[0-9]+, line:[0-9]+:[0-9]+> line:[0-9]+:[0-9]+ foo4 'void \(\)'}}
// COM: test the rest after -emit-llvm is resolved

void foo4() {
  while (true) {

  }
}

void foo5() {
  for (int i = 0; i < 5; i++) {

  }
}

void foo6() {
  while (true) {
    if (false) {
      int d = -1;
      if (d) {
        return;
      }
    }
  }
}

int foo7(int a, int b) {
  int c;
  c = a+b;
  return c;
}
