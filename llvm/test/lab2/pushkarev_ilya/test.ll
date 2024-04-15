
; RUN: opt -load-pass-plugin %llvmshlibdir/PushkarevFunctionInliningPass%pluginext\
; RUN: -passes=pushkarev-function-inlining -S %s | FileCheck %s

;--------------------
;|      TEST 1      |
;--------------------

;void void_no_arg() //expect inline
;{
;    int a = 0;
;    a += 1;
;}

;int foo1(int num)//expect inline
;{
;    void_no_arg();
;    return num;
;}

define dso_local void @_Z11void_no_argv() #0 {
  %1 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  %2 = load i32, i32* %1, align 4
  %3 = add nsw i32 %2, 1
  store i32 %3, i32* %1, align 4
  ret void
}

define dso_local noundef i32 @_Z4foo1i(i32 noundef %0) #0 {
  %2 = alloca i32, align 4
  store i32 %0, i32* %2, align 4
  call void @_Z11void_no_argv()
  %3 = load i32, i32* %2, align 4
  ret i32 %3
}

; CHECK: define dso_local void @_Z11void_no_argv() {
; CHECK-NEXT:   %1 = alloca i32, align 4
; CHECK-NEXT:   store i32 0, ptr %1, align 4
; CHECK-NEXT:   %2 = load i32, ptr %1, align 4
; CHECK-NEXT:   %3 = add nsw i32 %2, 1
; CHECK-NEXT:   store i32 %3, ptr %1, align 4
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define dso_local noundef i32 @_Z4foo1i(i32 noundef %0) {
; CHECK: .split:
; CHECK-NEXT:   %1 = alloca i32, align 4
; CHECK-NEXT:   store i32 %0, ptr %1, align 4
; CHECK-NEXT:   br label %2

; CHECK: 2:                                                ; preds = %.split
; CHECK-NEXT:   %3 = alloca i32, align 4
; CHECK-NEXT:   store i32 0, ptr %3, align 4
; CHECK-NEXT:   %4 = load i32, ptr %3, align 4
; CHECK-NEXT:   %5 = add nsw i32 %4, 1
; CHECK-NEXT:   store i32 %5, ptr %3, align 4
; CHECK-NEXT:   br label %6

; CHECK: 6:                                                ; preds = %2
; CHECK-NEXT:   %7 = load i32, ptr %1, align 4
; CHECK-NEXT:   ret i32 %7
; CHECK-NEXT: }

;--------------------
;|      TEST 2      |
;--------------------

define dso_local void @_Z8void_argii(i32 noundef %0, i32 noundef %1) #0 {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store i32 %0, i32* %3, align 4
  store i32 %1, i32* %4, align 4
  %6 = load i32, i32* %3, align 4
  %7 = load i32, i32* %4, align 4
  %8 = add nsw i32 %6, %7
  store i32 %8, i32* %5, align 4
  ret void
}


define dso_local noundef i32 @_Z4foo2i(i32 noundef %0) #0 {
  %2 = alloca i32, align 4
  store i32 %0, i32* %2, align 4
  %3 = load i32, i32* %2, align 4
  %4 = load i32, i32* %2, align 4
  call void @_Z8void_argii(i32 noundef %3, i32 noundef %4)
  %5 = load i32, i32* %2, align 4
  ret i32 %5
}

; CHECK: define dso_local void @_Z8void_argii(i32 noundef %0, i32 noundef %1) {
; CHECK-NEXT:   %3 = alloca i32, align 4
; CHECK-NEXT:   %4 = alloca i32, align 4
; CHECK-NEXT:   %5 = alloca i32, align 4
; CHECK-NEXT:   store i32 %0, ptr %3, align 4
; CHECK-NEXT:   store i32 %1, ptr %4, align 4
; CHECK-NEXT:   %6 = load i32, ptr %3, align 4
; CHECK-NEXT:   %7 = load i32, ptr %4, align 4
; CHECK-NEXT:   %8 = add nsw i32 %6, %7
; CHECK-NEXT:   store i32 %8, ptr %5, align 4
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define dso_local noundef i32 @_Z4foo2i(i32 noundef %0) {
; CHECK-NEXT:   %2 = alloca i32, align 4
; CHECK-NEXT:   store i32 %0, ptr %2, align 4
; CHECK-NEXT:   %3 = load i32, ptr %2, align 4
; CHECK-NEXT:   %4 = load i32, ptr %2, align 4
; CHECK-NEXT:   call void @_Z8void_argii(i32 noundef %3, i32 noundef %4)
; CHECK-NEXT:   %5 = load i32, ptr %2, align 4
; CHECK-NEXT:   ret i32 %5
; CHECK-NEXT: }

;--------------------
;|      TEST 3      |
;--------------------

define dso_local noundef i32 @_Z10int_no_argv() #0 {
  %1 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  %2 = load i32, i32* %1, align 4
  ret i32 %2
}


define dso_local noundef i32 @_Z4foo3i(i32 noundef %0) #0 {
  %2 = alloca i32, align 4
  store i32 %0, i32* %2, align 4
  %3 = call noundef i32 @_Z10int_no_argv()
  store i32 %3, i32* %2, align 4
  %4 = load i32, i32* %2, align 4
  ret i32 %4
}

; CHECK: define dso_local noundef i32 @_Z10int_no_argv() {
; CHECK-NEXT:  %1 = alloca i32, align 4
; CHECK-NEXT:  store i32 0, ptr %1, align 4
; CHECK-NEXT:  %2 = load i32, ptr %1, align 4
; CHECK-NEXT:  ret i32 %2
; CHECK-NEXT:}

; CHECK: define dso_local noundef i32 @_Z4foo3i(i32 noundef %0) {
; CHECK-NEXT:  %2 = alloca i32, align 4
; CHECK-NEXT:  store i32 %0, ptr %2, align 4
; CHECK-NEXT:  %3 = call noundef i32 @_Z10int_no_argv()
; CHECK-NEXT:  store i32 %3, ptr %2, align 4
; CHECK-NEXT:  %4 = load i32, ptr %2, align 4
; CHECK-NEXT:  ret i32 %4
; CHECK-NEXT:}

;--------------------
;|      TEST 4      |
;--------------------

define dso_local noundef i32 @_Z7int_argii(i32 noundef %0, i32 noundef %1) #0 {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store i32 %0, i32* %3, align 4
  store i32 %1, i32* %4, align 4
  %6 = load i32, i32* %3, align 4
  %7 = load i32, i32* %4, align 4
  %8 = add nsw i32 %6, %7
  store i32 %8, i32* %5, align 4
  %9 = load i32, i32* %5, align 4
  ret i32 %9
}


define dso_local noundef i32 @_Z4foo4i(i32 noundef %0) #0 {
  %2 = alloca i32, align 4
  store i32 %0, i32* %2, align 4
  %3 = load i32, i32* %2, align 4
  %4 = load i32, i32* %2, align 4
  %5 = call noundef i32 @_Z7int_argii(i32 noundef %3, i32 noundef %4)
  %6 = load i32, i32* %2, align 4
  ret i32 %6
} 

; CHECK: define dso_local noundef i32 @_Z7int_argii(i32 noundef %0, i32 noundef %1) {
; CHECK-NEXT:  %3 = alloca i32, align 4
; CHECK-NEXT:  %4 = alloca i32, align 4
; CHECK-NEXT:  %5 = alloca i32, align 4
; CHECK-NEXT:  store i32 %0, ptr %3, align 4
; CHECK-NEXT:  store i32 %1, ptr %4, align 4
; CHECK-NEXT:  %6 = load i32, ptr %3, align 4
; CHECK-NEXT:  %7 = load i32, ptr %4, align 4
; CHECK-NEXT:  %8 = add nsw i32 %6, %7
; CHECK-NEXT:  store i32 %8, ptr %5, align 4
; CHECK-NEXT:  %9 = load i32, ptr %5, align 4
; CHECK-NEXT:  ret i32 %9
; CHECK-NEXT:}

; CHECK: define dso_local noundef i32 @_Z4foo4i(i32 noundef %0) {
; CHECK-NEXT:  %2 = alloca i32, align 4
; CHECK-NEXT:  store i32 %0, ptr %2, align 4
; CHECK-NEXT:  %3 = load i32, ptr %2, align 4
; CHECK-NEXT:  %4 = load i32, ptr %2, align 4
; CHECK-NEXT:  %5 = call noundef i32 @_Z7int_argii(i32 noundef %3, i32 noundef %4)
; CHECK-NEXT:  %6 = load i32, ptr %2, align 4
; CHECK-NEXT:  ret i32 %6
; CHECK-NEXT:}

;--------------------
;|      TEST 5      |
;--------------------

define dso_local void @_Z15void_calls_voidv() #0 {
  call void @_Z11void_no_argv()
  ret void
}


define dso_local noundef i32 @_Z4foo5i(i32 noundef %0) #0 {
  %2 = alloca i32, align 4
  store i32 %0, i32* %2, align 4
  call void @_Z15void_calls_voidv()
  %3 = load i32, i32* %2, align 4
  ret i32 %3
}

; CHECK: define dso_local void @_Z15void_calls_voidv() {
; CHECK: .split:
; CHECK-NEXT:  br label %0

; CHECK: 0:                                                ; preds = %.split
; CHECK-NEXT:  %1 = alloca i32, align 4
; CHECK-NEXT:  store i32 0, ptr %1, align 4
; CHECK-NEXT:  %2 = load i32, ptr %1, align 4
; CHECK-NEXT:  %3 = add nsw i32 %2, 1
; CHECK-NEXT:  store i32 %3, ptr %1, align 4
; CHECK-NEXT:  br label %4

; CHECK: 4:                                                ; preds = %0
; CHECK-NEXT:  ret void
; CHECK-NEXT:}

; CHECK: define dso_local noundef i32 @_Z4foo5i(i32 noundef %0) {
; CHECK: .split:
; CHECK-NEXT:  %1 = alloca i32, align 4
; CHECK-NEXT:  store i32 %0, ptr %1, align 4
; CHECK-NEXT:  br label %2

; CHECK: 2:                                                ; preds = %.split
; CHECK-NEXT:  br label %3

; CHECK: 3:                                                ; preds = %2
; CHECK-NEXT:  %4 = alloca i32, align 4
; CHECK-NEXT:  store i32 0, ptr %4, align 4
; CHECK-NEXT:  %5 = load i32, ptr %4, align 4
; CHECK-NEXT:  %6 = add nsw i32 %5, 1
; CHECK-NEXT:  store i32 %6, ptr %4, align 4
; CHECK-NEXT:  br label %7

; CHECK: 7:                                                ; preds = %3
; CHECK-NEXT:  br label %8

; CHECK: 8:                                                ; preds = %7
; CHECK-NEXT:  %9 = load i32, ptr %1, align 4
; CHECK-NEXT:  ret i32 %9
; CHECK-NEXT:}
