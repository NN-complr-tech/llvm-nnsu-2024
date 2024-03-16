// RUN: %clang_cc1 -load %llvmshlibdir/depWarningPlugin%pluginext -plugin deprecated-warning %s 2>&1 | FileCheck %s

// CHECK: warning: Function contains 'deprecated' in its name
void deprecated();

// CHECK: warning: Function contains 'deprecated' in its name
void deprecated123();

// CHECK: warning: Function contains 'deprecated' in its name
void AAAdeprecatedOOO();

// CHECK-NOT: warning: Function contains 'deprecated' in its name
void something();

// CHECK-NOT: warning: Function contains 'deprecated' in its name
void deprecate_d();

// CHECK-NOT: warning: Function contains 'deprecated' in its name
void deprecate();
