// RUN: %clang_cc1 -load %llvmshlibdir/AttrubuteAlwaysPlugin%pluginext -plugin always_inlines-plugin %s 1>&1 | FileCheck %s

// CHECK: __attribute__((always_inline)) sum
int sum(int a, int b) { return a + b; }

void checkAlw(int a, int b) {
  if (a < b) {
    a = b;
  }
}

// CHECK: __attribute__((always_inline)) checkEmpty
void checkEmpty() { ; }

void ifEqual(char a, char b) {
  if (a != b) {
  }
}