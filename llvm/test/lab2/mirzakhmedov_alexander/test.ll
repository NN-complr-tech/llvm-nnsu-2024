; RUN: opt -load-pass-plugin %llvmshlibdir/MirzakhmedovInstrumentFunc%pluginext -passes=instrument-func -S %s | FileCheck %s
define dso_local void @_Z16instrument_startv() #0 {
  ret void
}

define dso_local void @_Z14instrument_endv() #0 {
  ret void
}

define dso_local void @_Z6simplev() #0 {
  ret void
}

; CHECK-LABEL: @_Z6simplev
; CHECK: call void @instrument_start()
; CHECK-NEXT: call void @instrument_end()
; CHECK-NEXT: ret void

define dso_local noundef i32 @_Z9rectangleii(i32 noundef %0, i32 noundef %1) #0 {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 %0, ptr %3, align 4
  store i32 %1, ptr %4, align 4
  %5 = load i32, ptr %3, align 4
  %6 = load i32, ptr %4, align 4
  %7 = mul nsw i32 %5, %6
  ret i32 %7
}

; CHECK-LABEL: @_Z9rectangleii
; CHECK: call void @instrument_start()
; CHECK-NEXT: %3 = alloca i32, align 4
; CHECK-NEXT: %4 = alloca i32, align 4
; CHECK-NEXT: store i32 %0, ptr %3, align 4
; CHECK-NEXT: store i32 %1, ptr %4, align 4
; CHECK-NEXT: %5 = load i32, ptr %3, align 4
; CHECK-NEXT: %6 = load i32, ptr %4, align 4
; CHECK-NEXT: %7 = mul nsw i32 %5, %6
; CHECK-NEXT: call void @instrument_end()
; CHECK-NEXT: ret i32 %7

define dso_local noundef i32 @_Z3maxii(i32 noundef %0, i32 noundef %1) #0 {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store i32 %0, ptr %4, align 4
  store i32 %1, ptr %5, align 4
  %6 = load i32, ptr %4, align 4
  %7 = load i32, ptr %5, align 4
  %8 = icmp sge i32 %6, %7
  br i1 %8, label %9, label %11

9:                                                ; preds = %2
  %10 = load i32, ptr %4, align 4
  store i32 %10, ptr %3, align 4
  br label %13

11:                                               ; preds = %2
  %12 = load i32, ptr %5, align 4
  store i32 %12, ptr %3, align 4
  br label %13

13:                                               ; preds = %11, %9
  %14 = load i32, ptr %3, align 4
  ret i32 %14
}

; CHECK-LABEL: @_Z3maxii
; CHECK: call void @instrument_start()
; CHECK-NEXT: %3 = alloca i32, align 4
; CHECK-NEXT: %4 = alloca i32, align 4
; CHECK-NEXT: %5 = alloca i32, align 4
; CHECK-NEXT: store i32 %0, ptr %4, align 4
; CHECK-NEXT: store i32 %1, ptr %5, align 4
; CHECK-NEXT: %6 = load i32, ptr %4, align 4
; CHECK-NEXT: %7 = load i32, ptr %5, align 4
; CHECK-NEXT: %8 = icmp sge i32 %6, %7
; CHECK-NEXT: br i1 %8, label %9, label %11
; CHECK: 9:                                                ; preds = %2
; CHECK-NEXT: %10 = load i32, ptr %4, align 4
; CHECK-NEXT: store i32 %10, ptr %3, align 4
; CHECK-NEXT: br label %13
; CHECK: 11:                                               ; preds = %2
; CHECK-NEXT: %12 = load i32, ptr %5, align 4
; CHECK-NEXT: store i32 %12, ptr %3, align 4
; CHECK-NEXT: br label %13
; CHECK: 13:                                               ; preds = %11, %9
; CHECK-NEXT: %14 = load i32, ptr %3, align 4
; CHECK-NEXT: call void @instrument_end()
; CHECK-NEXT: ret i32 %14

define dso_local noundef double @_Z6circled(double noundef %0) #0 {
  %2 = alloca double, align 8
  store double %0, ptr %2, align 8
  %3 = load double, ptr %2, align 8
  %4 = fmul double %3, 0x400921FAFC8B007A
  ret double %4
}

; CHECK-LABEL: @_Z6circled
; CHECK: call void @instrument_start()
; CHECK-NEXT: %2 = alloca double, align 8
; CHECK-NEXT: store double %0, ptr %2, align 8
; CHECK-NEXT: %3 = load double, ptr %2, align 8
; CHECK-NEXT: %4 = fmul double %3, 0x400921FAFC8B007A
; CHECK-NEXT: call void @instrument_end()
; CHECK-NEXT: ret double %4