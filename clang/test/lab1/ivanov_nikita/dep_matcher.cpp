// RUN: %clang_cc1 -load %llvmshlibdir/clangDepMatcher%pluginext -plugin deprecated-match %s 2>&1 | FileCheck %s
// REQUIRES: plugins


void foo_deprecated(int a, int b);

// CHECK: warning: Deprecated in function name
void deprecated(int c);

// CHECK: warning: Deprecated in function name
void abc();

// CHECK-NOT: warning: Deprecated in function name