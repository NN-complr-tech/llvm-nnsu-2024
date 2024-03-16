// RUN: %clang_cc1 -load %llvmshlibdir/DeprecationPlugin%pluginext -plugin deprecation-plugin %s 2>&1 | FileCheck %s

// CHECK-NOT: warning: Deprecated function found here
void somebody();

// CHECK-NOT: warning: Deprecated function found here
void oncedepre();

// CHECK: warning: Deprecated function found here
void toldeprecatedme();

// CHECK-NOT: warning: Deprecated function found here
void theworld();

// CHECK: warning: Deprecated function found here
void deprecatedme();