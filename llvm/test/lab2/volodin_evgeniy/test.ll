; RUN: opt -load-pass-plugin=%llvmshlibdir/InstrFuncVolodinE%shlibext -passes=instr-func-volodin -S %s | FileCheck %s

; int square(int x) {
;   return x * x;
; }

; void foo() {
;   return;
; }

; double max(double a, double b) {
;   if (a > b) {
;		  return a;
;	  } else {
;		  return b;
;	  }
; }

; CHECK-LABEL: @_Z6squarei
; CHECK: call void @instrument_start()
; CHECL-NEXT: %x.addr = alloca i32, align 4
; CHECK: call void @instrument_end()
; CHECK-NEXT: ret i32 %mul

define dso_local noundef i32 @_Z6squarei(i32 noundef %x) #0 {
entry:
  %x.addr = alloca i32, align 4
  store i32 %x, ptr %x.addr, align 4
  %0 = load i32, ptr %x.addr, align 4
  %1 = load i32, ptr %x.addr, align 4
  %mul = mul nsw i32 %0, %1
  ret i32 %mul
}

; CHECK-LABEL: @_Z3foov
; CHECK: call void @instrument_start()
; CHECL-NEXT: call void @instrument_end()
; CHECK-NEXT: ret void

define dso_local void @_Z3foov() #0 {
entry:
  ret void
}

; CHECK-LABEL: @_Z3maxdd
; CHECK: call void @instrument_start()
; CHECL-NEXT: %retval = alloca double, align 8
; CHECK: call void @instrument_end()
; CHECL-NEXT: ret double %4

define dso_local noundef double @_Z3maxdd(double noundef %a, double noundef %b) #0 {
entry:
  %retval = alloca double, align 8
  %a.addr = alloca double, align 8
  %b.addr = alloca double, align 8
  store double %a, ptr %a.addr, align 8
  store double %b, ptr %b.addr, align 8
  %0 = load double, ptr %a.addr, align 8
  %1 = load double, ptr %b.addr, align 8
  %cmp = fcmp ogt double %0, %1
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %2 = load double, ptr %a.addr, align 8
  store double %2, ptr %retval, align 8
  br label %return

if.else:                                          ; preds = %entry
  %3 = load double, ptr %b.addr, align 8
  store double %3, ptr %retval, align 8
  br label %return

return:                                           ; preds = %if.else, %if.then
  %4 = load double, ptr %retval, align 8
  ret double %4
}

