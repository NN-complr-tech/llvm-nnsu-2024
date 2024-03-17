// RUN: %clang_cc1 -load %llvmshlibdir/AttrubuteAlwaysPlugin%pluginext -plugin always_inlines-plugin %s 1>&1 | FileCheck %s

// CHECK: Added attribute always_inline in sum
int sum(int A, int B) { return A + B; }

// CHECK: checkAlw not suitable for the attribute
void checkAlw(int A, int B) {
  {
    if (A > B) {
      A = B;
    }
  }
}

// CHECK: Added attribute always_inline in checkEmpty
void checkEmpty() {}

// CHECK: ifEqual not suitable for the attribute
void ifEqual(char A, char B) {
  {
    {
      {
        {
          while (true) {
          }
          {}
        }
      }
    }
  }
}