; RUN: opt -load-pass-plugin %llvmshlibdir/InstrumentFunctionVeselovIlya%pluginext\
; RUN: -passes=instrument_function -S %s | FileCheck %s

; CHECK-LABEL: @_Z8sum_funcii
; CHECK: call void @instrument_start()
; CHECK: %0 = load i32, i32* %a.addr, align 4
; CHECK: %1 = load i32, i32* %b.addr, align 4
; CHECK: %add = add nsw i32 %0, %1
; CHECK: store i32 %add, i32* %c, align 4
; CHECK: call void @instrument_end()
; CHECK: %2 = load i32, i32* %c, align 4
; CHECK: ret i32 %2

define dso_local noundef i32 @_Z8sum_funcii(i32 noundef %a, i32 noundef %b) #0 {
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  %c = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  store i32 %b, ptr %b.addr, align 4
  call void @instrument_start()
  %0 = load i32, ptr %a.addr, align 4
  %1 = load i32, ptr %b.addr, align 4
  %add = add nsw i32 %0, %1
  store i32 %add, ptr %c, align 4
  call void @instrument_end()
  %2 = load i32, ptr %c, align 4
  ret i32 %2
}

; CHECK-LABEL: @_Z7func_vvv
; CHECK: call void @instrument_start()
; CHECK: call void @instrument_end()
; CHECK: ret void

define dso_local void @_Z7func_vvv() #0 {
entry:
  call void @instrument_start()
  call void @instrument_end()
  ret void
}

; CHECK-LABEL: @_Z8end_funcii
; CHECK: call void @instrument_start()
; CHECK: %0 = load i32, i32* %a.addr, align 4
; CHECK: %1 = load i32, i32* %b.addr, align 4
; CHECK: %add = add nsw i32 %0, %1
; CHECK: store i32 %add, i32* %c, align 4
; CHECK: call void @instrument_end()
; CHECK: %2 = load i32, i32* %c, align 4
; CHECK: ret i32 %2

define dso_local noundef i32 @_Z8end_funcii(i32 noundef %a, i32 noundef %b) #0 {
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  %c = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  store i32 %b, ptr %b.addr, align 4
  call void @instrument_start()
  %0 = load i32, ptr %a.addr, align 4
  %1 = load i32, ptr %b.addr, align 4
  %add = add nsw i32 %0, %1
  store i32 %add, ptr %c, align 4
  call void @instrument_end()
  %2 = load i32, ptr %c, align 4
  ret i32 %2
}

; CHECK-LABEL: @_Z13sum_cond_funcii
; CHECK: call void @instrument_start()
; CHECK: %0 = load i32, i32* %a.addr, align 4
; CHECK: %1 = load i32, i32* %b.addr, align 4
; CHECK: %cmp = icmp slt i32 %0, %1
; CHECK: br i1 %cmp, label %if.then, label %if.else

; CHECK: if.then:
; CHECK: %2 = load i32, i32* %b.addr, align 4
; CHECK: %3 = load i32, i32* %a.addr, align 4
; CHECK: %sub = sub nsw i32 %2, %3
; CHECK: store i32 %sub, i32* %c, align 4
; CHECK: br label %if.end

; CHECK: if.else:
; CHECK: %4 = load i32, i32* %a.addr, align 4
; CHECK: store i32 %4, i32* %c, align 4
; CHECK: br label %if.end

; CHECK: if.end:
; CHECK: call void @instrument_end()
; CHECK: %5 = load i32, i32* %c, align 4
; CHECK: ret i32 %5

define dso_local noundef i32 @_Z13sum_cond_funcii(i32 noundef %a, i32 noundef %b) #0 {
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  %c = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  store i32 %b, ptr %b.addr, align 4
  call void @instrument_start()
  %0 = load i32, ptr %a.addr, align 4
  %1 = load i32, ptr %b.addr, align 4
  %cmp = icmp slt i32 %0, %1
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %2 = load i32, ptr %b.addr, align 4
  %3 = load i32, ptr %a.addr, align 4
  %sub = sub nsw i32 %2, %3
  store i32 %sub, ptr %c, align 4
  br label %if.end

if.else:                                          ; preds = %entry
  %4 = load i32, ptr %a.addr, align 4
  store i32 %4, ptr %c, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  call void @instrument_end()
  %5 = load i32, ptr %c, align 4
  ret i32 %5
}

; CHECK-LABEL: @_Z3sumii
; CHECK: call void @instrument_start()
; CHECK: %0 = load i32, i32* %a.addr, align 4
; CHECK: %1 = load i32, i32* %b.addr, align 4
; CHECK: %add = add nsw i32 %0, %1
; CHECK: store i32 %add, i32* %result, align 4
; CHECK: call void @instrument_end()
; CHECK: %2 = load i32, i32* %result, align 4
; CHECK: ret i32 %2

define dso_local noundef i32 @_Z3sumii(i32 noundef %a, i32 noundef %b) #0 {
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  %result = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  store i32 %b, ptr %b.addr, align 4
  call void @instrument_start()
  %0 = load i32, ptr %a.addr, align 4
  %1 = load i32, ptr %b.addr, align 4
  %add = add nsw i32 %0, %1
  store i32 %add, ptr %result, align 4
  call void @instrument_end()
  %2 = load i32, ptr %result, align 4
  ret i32 %2
}

; CHECK-LABEL: @_Z8multiplyii
; CHECK: call void @instrument_start()
; CHECK: %0 = load i32, i32* %a.addr, align 4
; CHECK: %1 = load i32, i32* %b.addr, align 4
; CHECK: %mul = mul nsw i32 %0, %1
; CHECK: store i32 %mul, i32* %result, align 4
; CHECK: call void @instrument_end()
; CHECK: %2 = load i32, i32* %result, align 4
; CHECK: ret i32 %2

define dso_local noundef i32 @_Z8multiplyii(i32 noundef %a, i32 noundef %b) #0 {
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  %result = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  store i32 %b, ptr %b.addr, align 4
  call void @instrument_start()
  %0 = load i32, ptr %a.addr, align 4
  %1 = load i32, ptr %b.addr, align 4
  %mul = mul nsw i32 %0, %1
  store i32 %mul, ptr %result, align 4
  call void @instrument_end()
  %2 = load i32, ptr %result, align 4
  ret i32 %2
}

; CHECK-LABEL: @_Z11conditionali
; CHECK: call void @instrument_start()
; CHECK: %0 = load i32, i32* %x.addr, align 4
; CHECK: %cmp = icmp sgt i32 %0, 0
; CHECK: br i1 %cmp, label %if.then, label %if.else

; CHECK: if.then:
; CHECK: store i8 1, i8* %res, align 1
; CHECK: br label %if.end

; CHECK: if.else:
; CHECK: store i8 0, i8* %res, align 1
; CHECK: br label %if.end

; CHECK: if.end:
; CHECK: call void @instrument_end()
; CHECK: %1 = load i8, i8* %res, align 1
; CHECK: %tobool = trunc i8 %1 to i1
; CHECK: ret i1 %tobool

define dso_local noundef zeroext i1 @_Z11conditionali(i32 noundef %x) #0 {
entry:
  %x.addr = alloca i32, align 4
  %res = alloca i8, align 1
  store i32 %x, ptr %x.addr, align 4
  call void @instrument_start()
  %0 = load i32, ptr %x.addr, align 4
  %cmp = icmp sgt i32 %0, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  store i8 1, ptr %res, align 1
  br label %if.end

if.else:                                          ; preds = %entry
  store i8 0, ptr %res, align 1
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  call void @instrument_end()
  %1 = load i8, ptr %res, align 1
  %tobool = trunc i8 %1 to i1
  ret i1 %tobool
}

declare void @instrument_start() #1
declare void @instrument_end() #1