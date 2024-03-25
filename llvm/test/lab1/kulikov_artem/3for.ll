; RUN: opt -passes=kulikov-wrap-plugin -S %s | FileCheck %s

@j = dso_local global i32 2, align 4

define dso_local void @a() {
entry:
  %c = alloca i32, align 4
  %i = alloca i32, align 4
  %q = alloca i32, align 4
  store i32 10, ptr %c, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond
; CHECK:    call void @loop_start()
; CHECK-NEXT:    br label %for.cond

for.cond:
  %0 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %0, 10
  br i1 %cmp, label %for.body, label %for.end

for.body:
  %1 = load i32, ptr %c, align 4
  %inc = add nsw i32 %1, 1
  store i32 %inc, ptr %c, align 4
  br label %for.inc

for.inc:
  %2 = load i32, ptr %i, align 4
  %inc1 = add nsw i32 %2, 1
  store i32 %inc1, ptr %i, align 4
  br label %for.cond

; CHECK:       for.end:
; CHECK-NEXT:    call void @loop_end()
for.end:
  %3 = load i32, ptr %c, align 4
  %add = add nsw i32 %3, 42
  store i32 %add, ptr %q, align 4
  ret void
}

define dso_local i32 @b(i32 noundef %a) {
entry:
  %a.addr = alloca i32, align 4
  %c = alloca i32, align 4
  %q = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  store i32 10, ptr %c, align 4
  br label %for.cond
; CHECK:    call void @loop_start()
; CHECK-NEXT:    br label %for.cond

for.cond:
  %0 = load i32, ptr @j, align 4
  %1 = load i32, ptr %a.addr, align 4
  %cmp = icmp slt i32 %0, %1
  br i1 %cmp, label %for.body, label %for.end

for.body:
  %2 = load i32, ptr %c, align 4
  %inc = add nsw i32 %2, 1
  store i32 %inc, ptr %c, align 4
  br label %for.inc

for.inc:
  %3 = load i32, ptr @j, align 4
  %inc1 = add nsw i32 %3, 1
  store i32 %inc1, ptr @j, align 4
  br label %for.cond

; CHECK:       for.end:
; CHECK-NEXT:    call void @loop_end()
for.end:
  %4 = load i32, ptr %c, align 4
  %add = add nsw i32 %4, 42
  store i32 %add, ptr %q, align 4
  %5 = load i32, ptr %q, align 4
  ret i32 %5
}

define dso_local i32 @v() {
entry:
  %c = alloca i32, align 4
  %q = alloca i32, align 4
  store i32 10, ptr %c, align 4
  call void @a()
  br label %for.cond
; CHECK:      call void @loop_start()
; CHECK-NEXT:    br label %for.cond

for.cond:
  %0 = load i32, ptr @j, align 4
  %cmp = icmp sgt i32 %0, 5
  br i1 %cmp, label %if.then, label %if.end

; CHECK:       if.then:
; CHECK-NEXT:    call void @loop_end()
if.then:
  br label %for.end

if.end:
  br label %for.inc

for.inc:
  %1 = load i32, ptr @j, align 4
  %inc = add nsw i32 %1, 1
  store i32 %inc, ptr @j, align 4
  br label %for.cond

for.end:
  %2 = load i32, ptr %c, align 4
  %call = call i32 @b(i32 noundef %2)
  %3 = load i32, ptr %c, align 4
  %add = add nsw i32 %3, 42
  store i32 %add, ptr %q, align 4
  %4 = load i32, ptr %q, align 4
  ret i32 %4
}
