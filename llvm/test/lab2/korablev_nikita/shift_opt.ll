
define dso_local noundef i32 @_Z4testv() {
entry:
  %b = alloca i32, align 4
  %res = alloca i32, align 4
  store i32 3, ptr %b, align 4
  %0 = load i32, ptr %b, align 4
  %shiftInst = shl i32 4, %0
  store i32 %shiftInst, ptr %res, align 4
  %1 = load i32, ptr %res, align 4
  ret i32 %1
}

define dso_local noundef i32 @_Z5test2v() {
entry:
  %b = alloca i32, align 4
  %res = alloca i32, align 4
  store i32 3, ptr %b, align 4
  %0 = load i32, ptr %b, align 4
  %shiftInst = shl i32 4, %0
  store i32 %shiftInst, ptr %res, align 4
  %1 = load i32, ptr %res, align 4
  ret i32 %1
}
