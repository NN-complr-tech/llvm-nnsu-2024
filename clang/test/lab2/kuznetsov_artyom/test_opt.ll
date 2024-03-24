; ModuleID = 'test.ll'
source_filename = "test.ll"

define dso_local noundef i32 @_Z9factoriali(i32 noundef %value) {
entry:
  call void @instrument_start()
  %value.addr = alloca i32, align 4
  store i32 %value, ptr %value.addr, align 4
  %0 = load i32, ptr %value.addr, align 4
  %cmp = icmp sle i32 %0, 1
  br i1 %cmp, label %cond.true, label %cond.false

cond.true:                                        ; preds = %entry
  br label %cond.end

cond.false:                                       ; preds = %entry
  %1 = load i32, ptr %value.addr, align 4
  %2 = load i32, ptr %value.addr, align 4
  %sub = sub nsw i32 %2, 1
  %call = call noundef i32 @_Z9factoriali(i32 noundef %sub)
  %mul = mul nsw i32 %1, %call
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ 1, %cond.true ], [ %mul, %cond.false ]
  call void @instrument_end()
  ret i32 %cond
}

define dso_local noundef i32 @_Z4funcb(i1 noundef zeroext %pred) {
entry:
  call void @instrument_start()
  %retval = alloca i32, align 4
  %pred.addr = alloca i8, align 1
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %frombool = zext i1 %pred to i8
  store i8 %frombool, ptr %pred.addr, align 1
  %0 = load i8, ptr %pred.addr, align 1
  %tobool = trunc i8 %0 to i1
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i32 10, ptr %a, align 4
  store i32 20, ptr %b, align 4
  %1 = load i32, ptr %a, align 4
  %2 = load i32, ptr %b, align 4
  %add = add nsw i32 %1, %2
  store i32 %add, ptr %retval, align 4
  br label %return

if.end:                                           ; preds = %entry
  store i32 0, ptr %retval, align 4
  br label %return

return:                                           ; preds = %if.end, %if.then
  %3 = load i32, ptr %retval, align 4
  call void @instrument_end()
  ret i32 %3
}

define dso_local void @_Z8voidFuncv() {
entry:
  call void @instrument_start()
  call void @instrument_end()
  ret void
}

declare void @instrument_start()

declare void @instrument_end()
