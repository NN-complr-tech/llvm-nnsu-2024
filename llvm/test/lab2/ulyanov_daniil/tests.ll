; RUN: opt -load-pass-plugin=%llvmshlibdir/ulyanovMulToBitShiftPlugin%shlibext -passes=ulyanovMulToBitShiftPlugin -S %s | FileCheck %s

;int foo1() {
;  int a = 3;
;  int b = 3 * 2;
;  return 16 * b;
;}

define dso_local noundef i32 @_Z4foo1v() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  store i32 3, ptr %1, align 4
  store i32 6, ptr %2, align 4
  %3 = load i32, ptr %2, align 4
  %4 = mul nsw i32 16, %3
  ret i32 %4
}

; CHECK-LABEL @_Z4foo1v()
; CHECK: %1 = alloca i32, align 4
; CHECK-NEXT: %2 = alloca i32, align 4
; CHECK-NEXT: store i32 3, ptr %1, align 4
; CHECK-NEXT: store i32 6, ptr %2, align 4
; CHECK-NEXT: %3 = load i32, ptr %2, align 4
; CHECK-NEXT: %4 = shl i32 %3, 4
; CHECK-NEXT: ret i32 %4