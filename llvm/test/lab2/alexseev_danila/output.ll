; ModuleID = 'test2.ll'
source_filename = "test2.ll"

define dso_local void @whileFooWLoopBorderingCall() {
entry:
  %i = alloca i32, align 4
  store i32 10, ptr %i, align 4
  call void @loop_start()
  br label %while.cond

while.cond:                                       ; preds = %while.body, %entry
  %0 = load i32, ptr %i, align 4
  %cmp = icmp sgt i32 %0, 0
  br i1 %cmp, label %while.body, label %while.end

while.body:                                       ; preds = %while.cond
  %1 = load i32, ptr %i, align 4
  %dec = add nsw i32 %1, -1
  store i32 %dec, ptr %i, align 4
  br label %while.cond

while.end:                                        ; preds = %while.cond
  call void @loop_end()
  ret void
}

declare void @loop_start()

declare void @loop_end()
