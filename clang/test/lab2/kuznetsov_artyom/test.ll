; RUN: opt -load-pass-plugin %llvmshlibdir/InstrumentFunctionsPlugin%pluginext -passes=instr_func -S %s | FileCheck %s

; CHECK-LABEL: @_Z9factoriali
; CHECK: call void @instrument_start()
; CHECK: call void @instrument_end()

define dso_local noundef i32 @_Z9factoriali(i32 noundef %value) {
entry:
  %value.addr = alloca i32, align 4
  store i32 %value, ptr %value.addr, align 4
  %0 = load i32, ptr %value.addr, align 4
  %cmp = icmp sle i32 %0, 1
  br i1 %cmp, label %cond.true, label %cond.false

cond.true:
  br label %cond.end

cond.false:
  %1 = load i32, ptr %value.addr, align 4
  %2 = load i32, ptr %value.addr, align 4
  %sub = sub nsw i32 %2, 1
  %call = call noundef i32 @_Z9factoriali(i32 noundef %sub)
  %mul = mul nsw i32 %1, %call
  br label %cond.end

cond.end:
  %cond = phi i32 [ 1, %cond.true ], [ %mul, %cond.false ]
  ret i32 %cond
}

; CHECK-LABEL: @_Z4funcb
; CHECK: call void @instrument_start()
; CHECK: call void @instrument_end()

define dso_local noundef i32 @_Z4funcb(i1 noundef zeroext %pred) {
entry:
  %retval = alloca i32, align 4
  %pred.addr = alloca i8, align 1
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %frombool = zext i1 %pred to i8
  store i8 %frombool, ptr %pred.addr, align 1
  %0 = load i8, ptr %pred.addr, align 1
  %tobool = trunc i8 %0 to i1
  br i1 %tobool, label %if.then, label %if.end

if.then:
  store i32 10, ptr %a, align 4
  store i32 20, ptr %b, align 4
  %1 = load i32, ptr %a, align 4
  %2 = load i32, ptr %b, align 4
  %add = add nsw i32 %1, %2
  store i32 %add, ptr %retval, align 4
  br label %return

if.end:
  store i32 0, ptr %retval, align 4
  br label %return

return:
  %3 = load i32, ptr %retval, align 4
  ret i32 %3
}

; CHECK-LABEL: @_Z8voidFuncv
; CHECK: call void @instrument_start()
; CHECK: call void @instrument_end()

define dso_local void @_Z8voidFuncv() {
entry:
  ret void
}
