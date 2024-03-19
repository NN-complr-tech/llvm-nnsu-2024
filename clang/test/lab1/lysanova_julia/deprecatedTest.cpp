// RUN: %clang_cc1 -load %llvmshlibdir/LysanovaDepWarnPlugin%pluginext -plugin depWarning %s 2>&1 | FileCheck %s

// CHECK: warning: Function contains 'deprecated' in its name
void deprecated();

// CHECK: warning: Function contains 'deprecated' in its name
void deprecated123();

// CHECK: warning: Function contains 'deprecated' in its name
void aaAdeprecatedOoo();

// CHECK-NOT: warning: Function contains 'deprecated' in its name
void something();

// CHECK-NOT: warning: Function contains 'deprecated' in its name
void deprecatend();

// CHECK-NOT: warning: Function contains 'deprecated' in its name
void deprecate();

// CHECK-NOT: warning: Function contains 'deprecated' in its name
void deprecatedFunc();

// RUN: %clang_cc1 -load %llvmshlibdir/LysanovaDepWarnPlugin%pluginext -plugin depWarning -plugin-arg-depWarning -help %s 2>&1 | FileCheck %s

//CHECK: This plugin throws warning if func name contains 'deprecated'
