// RUN: %clang_cc1 -load %llvmshlibdir/AlwaysInlinePluginAkopyan%pluginext -plugin always-inline %s 2>&1 | FileCheck %s

// CHECK: __attribute__((always_inline)) FuncInline
int FuncInline(){ return 10; }

void FuncNotInline() {
  if (true) {
    return;
  }
  return;
}