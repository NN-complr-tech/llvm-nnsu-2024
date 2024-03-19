// RUN: %clang_cc1 -load %llvmshlibdir/LysanovaDepWarnPlugin%pluginext -plugin depWarning %s 2>&1 | FileCheck %s

// CHECK: warning: Function have deprecated in its name!
void deprecated();

// CHECK: warning: Function have deprecated in its name!
void deprecated123();

// CHECK-NOT: warning: Function have deprecated in its name!
void abcdf();

// CHECK-NOT: warning: Function have deprecated in its name!
void eprecated();

// RUN: %clang_cc1 -load %llvmshlibdir/LysanovaDepWarnPlugin%pluginext -plugin depWarning -plugin-arg-depWarning -help %s 2>&1 | FileCheck %s

//CHECK: This plugin throws warning if func name contains 'deprecated'
