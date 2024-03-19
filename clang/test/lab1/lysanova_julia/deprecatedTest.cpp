// RUN: %clang_cc1 -load %llvmshlibdir/LysanovaDepWarnPlugin%pluginext -plugin depWarning %s 2>&1 | FileCheck %s

// CHECK: warning: This plugin throws warning if func name contains 'deprecated'
void deprecated();

// CHECK: warning: This plugin throws warning if func name contains 'deprecated'
void deprecated123();

// CHECK-NOT: warning: This plugin throws warning if func name contains 'deprecated'
void abcdf();

// CHECK-NOT: warning: This plugin throws warning if func name contains 'deprecated'
void eprecated();

// RUN: %clang_cc1 -load %llvmshlibdir/LysanovaDepWarnPlugin%pluginext -plugin depWarning -plugin-arg-depWarning -help %s 2>&1 | FileCheck %s

//CHECK: This plugin throws warning if func name contains 'deprecated'
