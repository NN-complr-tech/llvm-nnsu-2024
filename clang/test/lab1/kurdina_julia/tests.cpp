// RUN: %clang_cc1 -load %llvmshlibdir/addWarning%pluginext -plugin warn_dep %s 2>&1 | FileCheck %s

// CHECK: warning: Deprecated is contain in function name
void deprecated();

// CHECK: warning: Deprecated is contain in function name
void function_name_is_deprecated();

// CHECK-NOT: warning: Deprecated is contain in function name
void function();

// CHECK-NOT: warning: Deprecated is contain in function name
void function_depr();
