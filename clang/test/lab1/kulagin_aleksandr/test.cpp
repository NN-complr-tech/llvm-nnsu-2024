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
  bool f = true;
  if (f) {
    f = false;
  }
}
// CHECK-NOT: `-AlwaysInlineAttr {{0[xX][0-9a-fA-F]+ <line:[0-9]+:[0-9]+> always_inline}}

// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+test\.cpp:[0-9]+:[0-9]+, line:[0-9]+:[0-9]+> line:[0-9]+:[0-9]+ foo4 'void \(\)'}}
void foo4() {
  while (true) {

  }
}
// CHECK-NOT: `-AlwaysInlineAttr {{0[xX][0-9a-fA-F]+ <line:[0-9]+:[0-9]+> always_inline}}

// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+test\.cpp:[0-9]+:[0-9]+, line:[0-9]+:[0-9]+> line:[0-9]+:[0-9]+ foo5 'void \(\)'}}
void foo5() {
  for (int i = 0; i < 5; i++) {

  }
}
// CHECK-NOT: `-AlwaysInlineAttr {{0[xX][0-9a-fA-F]+ <line:[0-9]+:[0-9]+> always_inline}}

// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+test\.cpp:[0-9]+:[0-9]+, line:[0-9]+:[0-9]+> line:[0-9]+:[0-9]+ foo6 'void \(\)'}}
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
// CHECK-NOT: `-AlwaysInlineAttr {{0[xX][0-9a-fA-F]+ <line:[0-9]+:[0-9]+> always_inline}}

// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+test\.cpp:[0-9]+:[0-9]+, line:[0-9]+:[0-9]+> line:[0-9]+:[0-9]+ foo7 'void \(\)'}}
void foo7() {
  do {

  } while(true);
}
// CHECK-NOT: `-AlwaysInlineAttr {{0[xX][0-9a-fA-F]+ <line:[0-9]+:[0-9]+> always_inline}}

// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+test\.cpp:[0-9]+:[0-9]+, line:[0-9]+:[0-9]+> line:[0-9]+:[0-9]+ foo8 'void \(\)'}}
void foo8() {
  int a = 1;
  switch (a) {
    case 1:
      break;
    default:
      break;
  }
}
// CHECK-NOT: `-AlwaysInlineAttr {{0[xX][0-9a-fA-F]+ <line:[0-9]+:[0-9]+> always_inline}}

// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+test\.cpp:[0-9]+:[0-9]+, line:[0-9]+:[0-9]+> line:[0-9]+:[0-9]+ foo9 'int \(int, int\)'}}
int foo9(int a, int b) {
  int c;
  c = a+b;
  return c;
}
// CHECK: `-AlwaysInlineAttr {{0[xX][0-9a-fA-F]+ <line:[0-9]+:[0-9]+> always_inline}}
