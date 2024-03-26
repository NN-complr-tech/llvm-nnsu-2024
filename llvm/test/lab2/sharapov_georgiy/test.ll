; RUN: opt -passes=LoopFramer -S %s | FileCheck %s

@i = dso_local global i32 0, align 4


define dso_local i32 @while_func() #0 {
entry:
  %j = alloca i32, align 4
  store i32 2, ptr %j, align 4
  ; CHECK: call void @loop_start()
  ; CHECK-NEXT: br label %while.cond
  br label %while.cond

while.cond:                                       
  %0 = load i32, ptr %j, align 4
  %cmp = icmp sge i32 %0, 0
  br i1 %cmp, label %while.body, label %while.end

while.body:                                       
  %1 = load i32, ptr %j, align 4
  %dec = add nsw i32 %1, -1
  store i32 %dec, ptr %j, align 4
  br label %while.cond

  ; CHECK: while.end:
  ; CHECK-NEXT: call void @loop_end()                 
while.end:
  %2 = load i32, ptr %j, align 4
  ret i32 %2
}


define dso_local i32 @do_while_func() #0 {
entry:
  ; CHECK: call void @loop_start()
  ; CHECK-NEXT: br label %while.cond
  br label %while.cond

while.cond:                                      
  %0 = load i32, ptr @i, align 4
  %cmp = icmp sle i32 %0, 2
  br i1 %cmp, label %while.body, label %while.end

while.body:                                       
  %1 = load i32, ptr @i, align 4
  %inc = add nsw i32 %1, 1
  store i32 %inc, ptr @i, align 4
  br label %while.cond

  ; CHECK: while.end:
  ; CHECK-NEXT: call void @loop_end()  
while.end:                                       
  %2 = load i32, ptr @i, align 4
  ret i32 %2
}


define dso_local i32 @for_loop() #0 {
entry:
  %k = alloca i32, align 4
  %b = alloca i32, align 4
  store i32 0, ptr %k, align 4
  store i32 1, ptr %b, align 4
  store i32 0, ptr %k, align 4
  ; CHECK: call void @loop_start()
  ; CHECK-NEXT: br label %for.cond
  br label %for.cond

for.cond:                                       
  %0 = load i32, ptr %k, align 4
  %cmp = icmp slt i32 %0, 5
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         
  %1 = load i32, ptr %b, align 4
  %add = add nsw i32 %1, 1
  store i32 %add, ptr %b, align 4
  br label %for.inc

for.inc:                                          
  %2 = load i32, ptr %k, align 4
  %inc = add nsw i32 %2, 1
  store i32 %inc, ptr %k, align 4
  br label %for.cond

  ; CHECK: for.end:
  ; CHECK-NEXT: call void @loop_end()
for.end:                                          
  %3 = load i32, ptr %b, align 4
  ret i32 %3
}

; source code
;
; int while_func() {
;     int j = 2;
;     while (j >= 0) {
;         j--;
;     }
;     return j;
; }
;
; int i = 0;
; int do_while_func() {
;     while (i <= 2) {
;         i++;
;     }
;     return i;
; }
;
; int for_loop() {
;     int k = 0;
;     int b = 1;
;     for (k = 0; k < 5; k++) {
;         b = b + 1;
;     }
;     return b;
; }
