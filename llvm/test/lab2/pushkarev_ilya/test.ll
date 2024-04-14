
; RUN: opt -load-pass-plugin %llvmshlibdir/PushkarevFunctionInliningPass%pluginext\
; RUN: -passes=pushkarev-function-inlining -S %s | FileCheck %s


; COM: Simple inline check. Expect inline

;void func()
;{
;    int a = 0;
;    a += 1;
;}
;
;int foo(int num)
;{
;    func();
;    return num;
;}

define dso_local void @_Z4funcv() #0 {
entry:
  %a = alloca i32, align 4
  store i32 0, ptr %a, align 4
  %0 = load i32, ptr %a, align 4
  %add = add nsw i32 %0, 1
  store i32 %add, ptr %a, align 4
  ret void
}

define dso_local noundef i32 @_Z3fooi(i32 noundef %num) #0 {
entry:
  %num.addr = alloca i32, align 4
  store i32 %num, ptr %num.addr, align 4
  call void @_Z4funcv()
  %0 = load i32, ptr %num.addr, align 4
  ret i32 %0
}

; CHECK: define dso_local void @_Z4funcv() {
; CHECK: entry:
; CHECK-NEXT:   %a = alloca i32, align 4
; CHECK-NEXT:   store i32 0, ptr %a, align 4
; CHECK-NEXT:   %0 = load i32, ptr %a, align 4
; CHECK-NEXT:   %add = add nsw i32 %0, 1
; CHECK-NEXT:   store i32 %add, ptr %a, align 4
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define dso_local noundef i32 @_Z3fooi(i32 noundef %num) {
; CHECK: entry.split:
; CHECK-NEXT:   %num.addr = alloca i32, align 4
; CHECK-NEXT:   store i32 %num, ptr %num.addr, align 4
; CHECK-NEXT:   br label %0
; CHECK: 0:                                  ; preds = %entry.split
; CHECK-NEXT:   %1 = alloca i32, align 4
; CHECK-NEXT:   store i32 0, ptr %1, align 4
; CHECK-NEXT:   %2 = load i32, ptr %1, align 4
; CHECK-NEXT:   %3 = add nsw i32 %2, 1
; CHECK-NEXT:   store i32 %3, ptr %1, align 4
; CHECK-NEXT:   br label %entry
; CHECK: entry:                                  ; preds = %0
; CHECK-NEXT:   %4 = load i32, ptr %num.addr, align 4
; CHECK-NEXT:   ret i32 %4
; CHECK-NEXT: }

; --------------------------------------------------------------------
