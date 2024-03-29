; RUN: opt -load-pass-plugin=%llvmshlibdir/MulToBitShift%shlibext -passes=MulToBitShift -S %s | FileCheck %s

;int f1(int a){
;    int c = a * 4;
;    return c;
;}

;void f2(){
;    int a = 3;
;    int c = a * 4;
;}

;void f3(){
;    int c = 3 * 4;
;}

;void f4(){
;    int c = 4 * 1;
;}

;void f5(){
;    int c = 3 * 1;
;}

;void f6(){
;    int c = 4 * 0;
;}

define dso_local i32 @f1(i32 noundef %0) #0 {
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32 %0, ptr %2, align 4
  %4 = load i32, ptr %2, align 4
  %5 = mul nsw i32 %4, 4
;  CHECK: %5 = shl i32 %4, 2
  store i32 %5, ptr %3, align 4
  %6 = load i32, ptr %3, align 4
  ret i32 %6
}

define dso_local void @f2() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  store i32 3, ptr %1, align 4
  %3 = load i32, ptr %1, align 4
  %4 = mul nsw i32 %3, 4
;  CHECK: %4 = shl i32 %3, 2
  store i32 %4, ptr %2, align 4
  ret void
}


define dso_local void @f3() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  store i32 3, ptr %1, align 4
  %3 = load i32, ptr %1, align 4
  %4 = mul nsw i32 4, %3
; CHECK: %4 = shl i32 %3, 2
  store i32 %4, ptr %2, align 4
  ret void
}


define dso_local void @f4() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  store i32 4, ptr %1, align 4
  %3 = load i32, ptr %1, align 4
  %4 = mul nsw i32 %3, 1
;  CHECK: %4 = shl i32 %3, 0
  store i32 %4, ptr %2, align 4
  ret void
}


define dso_local void @f5() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  store i32 3, ptr %1, align 4
  %3 = load i32, ptr %1, align 4
  %4 = mul nsw i32 0, %3
  store i32 %4, ptr %2, align 4
  ret void
}