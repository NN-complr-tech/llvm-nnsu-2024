; RUN: opt -load-pass-plugin=%llvmshlibdir/BozinInlinePass%pluginext -passes=bozin-inline -S %s | FileCheck %s

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
