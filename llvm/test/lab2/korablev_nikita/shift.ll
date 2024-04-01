
define dso_local noundef i32 @_Z4testv() #0 {
entry:
  %b = alloca i32, align 4
  %res = alloca i32, align 4
  store i32 3, ptr %b, align 4
  %0 = load i32, ptr %b, align 4
  %mul = mul nsw i32 %0, 4
  store i32 %mul, ptr %res, align 4
  %1 = load i32, ptr %res, align 4
  ret i32 %1
}

define dso_local noundef i32 @_Z5test2v() #0 {
entry:
  %b = alloca i32, align 4
  %res = alloca i32, align 4
  store i32 3, ptr %b, align 4
  %0 = load i32, ptr %b, align 4
  %mul = mul nsw i32 4, %0
  store i32 %mul, ptr %res, align 4
  %1 = load i32, ptr %res, align 4
  ret i32 %1
}