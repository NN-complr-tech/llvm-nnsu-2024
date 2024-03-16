// RUN: %clang_cc1 -load %llvmshlibdir/DeprecationPlugin%pluginext -plugin deprecation-plugin %s 2>&1 | FileCheck %s

// CHECK-NOT: warning: Deprecated function found here
int somebody();

// CHECK-NOT: warning: Deprecated function found here
int oncedepre();

// CHECK: warning: Deprecated function found here
int toldeprecatedme();

// CHECK-NOT: warning: Deprecated function found here
int theworld();

// CHECK: warning: Deprecated function found here
int deprecatedme();