; ModuleID = '/home/mortus/NNSU/llvm-nnsu-2024/llvm/test/lab2/polozov_vladislav/test.ll'
source_filename = "/home/mortus/NNSU/llvm-nnsu-2024/llvm/test/lab2/polozov_vladislav/test.ll"

define dso_local i32 @main() {
entry:
  %retval = alloca i32, align 4
  %a = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  store i32 10, ptr %a, align 4
  ret i32 0
}

define dso_local i32 @foo() {
entry:
  %retval = alloca i32, align 4
  %n = alloca i32, align 4
  %t = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  store i32 10, ptr %n, align 4
  store i32 30, ptr %t, align 4
  call void @loop_start()
  br label %while.cond

while.cond:                                       ; preds = %while.body, %entry
  %0 = load i32, ptr %n, align 4
  %cmp = icmp sgt i32 %0, 0
  br i1 %cmp, label %while.body, label %while.end

while.body:                                       ; preds = %while.cond
  %1 = load i32, ptr %t, align 4
  %mul = mul nsw i32 %1, 30
  store i32 %mul, ptr %t, align 4
  %2 = load i32, ptr %t, align 4
  %rem = srem i32 %2, 7
  store i32 %rem, ptr %t, align 4
  %3 = load i32, ptr %n, align 4
  %dec = add nsw i32 %3, -1
  store i32 %dec, ptr %n, align 4
  br label %while.cond

while.end:                                        ; preds = %while.cond
  call void @loop_end()
  ret i32 0
}

define dso_local i32 @bar() {
entry:
  %retval = alloca i32, align 4
  %n = alloca i32, align 4
  %t = alloca i32, align 4
  %i = alloca i32, align 4
  %x = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  store i32 10, ptr %n, align 4
  store i32 30, ptr %t, align 4
  store i32 0, ptr %i, align 4
  call void @loop_start()
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %i, align 4
  %1 = load i32, ptr %n, align 4
  %cmp = icmp slt i32 %0, %1
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %2 = load i32, ptr %t, align 4
  %mul = mul nsw i32 %2, 25
  store i32 %mul, ptr %x, align 4
  %3 = load i32, ptr %t, align 4
  %add = add nsw i32 %3, 3
  store i32 %add, ptr %t, align 4
  %4 = load i32, ptr %t, align 4
  %div = sdiv i32 %4, 5
  store i32 %div, ptr %t, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %5 = load i32, ptr %i, align 4
  %inc = add nsw i32 %5, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  call void @loop_end()
  ret i32 0
}

declare void @loop_start()

declare void @loop_end()
