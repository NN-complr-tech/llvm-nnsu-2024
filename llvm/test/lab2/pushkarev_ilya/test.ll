; RUN: split-file %s %t
; RUN: opt -passes="pushkarev-function-inlining" -S %t/test1.ll | FileCheck %t/test1.ll

;--- test1.ll
; COM: Simple magic inline check. Expect inline

;void foo() {
;  float a = 1.0f;
;  a += 1.0f;
;}
;
;void bar() {
;  int a = 0;
;  foo();
;  a++;
;}

; COM: The result is:

;define dso_local void @_Z3foov() {
;  %1 = alloca float, align 4
;  store float 1.000000e+00, ptr %1, align 4
;  %2 = load float, ptr %1, align 4
;  %3 = fadd float %2, 1.000000e+00
;  store float %3, ptr %1, align 4
;  ret void
;}
;
;define dso_local void @_Z3barv() {
;.split:
;  %0 = alloca i32, align 4
;  store i32 0, ptr %0, align 4
;  br label %1
;
;1:                                                ; preds = %.split
;  %2 = alloca float, align 4
;  store float 1.000000e+00, ptr %2, align 4
;  %3 = load float, ptr %2, align 4
;  %4 = fadd float %3, 1.000000e+00
;  store float %4, ptr %2, align 4
;  br label %5
;
;5:                                                ; preds = %1
;  %6 = load i32, ptr %0, align 4
;  %7 = add nsw i32 %6, 1
;  store i32 %7, ptr %0, align 4
;  ret void
;}

; COM: Begin checking

; CHECK: define dso_local void @_Z3barv() {
; CHECK: store i32 0, ptr %0, align 4
; CHECK-NEXT: br label %1
; CHECK: %2 = alloca float, align 4
; CHECK-NEXT: store float 1.000000e+00, ptr %2, align 4
; CHECK-NEXT: %3 = load float, ptr %2, align 4
; CHECK-NEXT: %4 = fadd float %3, 1.000000e+00
; CHECK-NEXT: store float %4, ptr %2, align 4
; CHECK-NEXT: br label %5
; COM: After inline function...
; CHECK: %6 = load i32, ptr %0, align 4

define dso_local void @_Z3foov() {
  %1 = alloca float, align 4
  store float 1.000000e+00, float* %1, align 4
  %2 = load float, float* %1, align 4
  %3 = fadd float %2, 1.000000e+00
  store float %3, float* %1, align 4
  ret void
}

define dso_local void @_Z3barv() {
  %1 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  call void @_Z3foov()
  %2 = load i32, i32* %1, align 4
  %3 = add nsw i32 %2, 1
  store i32 %3, i32* %1, align 4
  ret void
}
