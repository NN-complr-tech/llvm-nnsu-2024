// RUN: split-file %s %t

// RUN: %clang_cc1 -load %llvmshlibdir/identifierRenamer%pluginext\
// RUN: -add-plugin identifierRenamer\
// RUN: -plugin-arg-identifierRenamer formerName=x\
// RUN: -plugin-arg-identifierRenamer renewedName=z %t/rename_var.cpp
// RUN: FileCheck %s < %t/rename_var.cpp --check-prefix=VAR

// VAR: int whoAreMe(int t) {
// VAR-NEXT: int z = 2, y = 3 + t;
// VAR-NEXT: z++;
// VAR-NEXT: y--;
// VAR-NEXT: return z + y;
// VAR-NEXT: }

// RUN: %clang_cc1 -load %llvmshlibdir/identifierRenamer%pluginext\
// RUN: -add-plugin identifierRenamer\
// RUN: -plugin-arg-identifierRenamer formerName=int\
// RUN: -plugin-arg-identifierRenamer renewedName=short %t/rename_type.cpp
// RUN: FileCheck %s < %t/rename_type.cpp --check-prefix=TYPE

// TYPE: short* whoAreMe(short x, short y) {
// TYPE-NEXT: short temp = x - y;
// TYPE-NEXT: short *result = &temp;
// TYPE-NEXT: return result;
// TYPE-NEXT: }

// RUN: %clang_cc1 -load %llvmshlibdir/identifierRenamer%pluginext\
// RUN: -add-plugin identifierRenamer\
// RUN: -plugin-arg-identifierRenamer formerName=whoAreMe\
// RUN: -plugin-arg-identifierRenamer renewedName=whoAreYou %t/rename_func.cpp
// RUN: FileCheck %s < %t/rename_func.cpp --check-prefix=FUNC

// FUNC: bool whoAreYou(bool isCorrect) {
// FUNC-NEXT: return isCorrect == true;
// FUNC-NEXT: }
// FUNC-NEXT: int whoAreMeOther(int id, int bal) {
// FUNC-NEXT: int check = whoAreYou(3) + whoAreYou(5);
// FUNC-NEXT: return check;
// FUNC-NEXT: }

// RUN: %clang_cc1 -load %llvmshlibdir/identifierRenamer%pluginext\
// RUN: -add-plugin identifierRenamer\
// RUN: -plugin-arg-identifierRenamer formerName=Squidward\
// RUN: -plugin-arg-identifierRenamer renewedName=SpongeBob %t/rename_class.cpp
// RUN: FileCheck %s < %t/rename_class.cpp --check-prefix=CLASS

// CLASS: class SpongeBob{
// CLASS-NEXT: private:
// CLASS-NEXT: int h;
// CLASS-NEXT: int w;
// CLASS-NEXT: int d;
// CLASS-NEXT: int s;
// CLASS-NEXT: public:
// CLASS-NEXT: SpongeBob() {}
// CLASS-NEXT: SpongeBob(int h, int w, int d, int s): h(h), w(w), d(d), s(s) {}
// CLASS-NEXT: SpongeBob(SpongeBob &obj) {}
// CLASS-NEXT: SpongeBob returnedClassObj(SpongeBob obj){ return obj; };
// CLASS-NEXT: ~SpongeBob();
// CLASS-NEXT: };
// CLASS: void whoAreMe(SpongeBob obj) {
// CLASS-NEXT: SpongeBob objectOne;
// CLASS-NEXT: SpongeBob *objectTwo = new SpongeBob(5, 10, 1, 20);
// CLASS-NEXT: delete objectTwo;
// CLASS-NEXT: }

// RUN: %clang_cc1 -load %llvmshlibdir/identifierRenamer%pluginext\
// RUN: -add-plugin identifierRenamer\
// RUN: -plugin-arg-identifierRenamer otherFormerName=FormerName\
// RUN: -plugin-arg-identifierRenamer renewedName=SpongeBob \
// RUN: 2>&1 | FileCheck %s --check-prefix=ERROR

// ERROR: Error: incorrect parameters input.

//--- rename_var.cpp
int whoAreMe(int t) {
  int x = 2, y = 3 + t;
  x++;
  y--;
  return x + y;
}
//--- rename_type.cpp
int* whoAreMe(int x, int y) {
  int temp = x - y;
  int *result = &temp;
  return result;
}
//--- rename_func.cpp
bool whoAreMe(bool isCorrect) {
  return isCorrect == true;
}
int whoAreMeOther(int id, int bal) {
  int check = whoAreMe(3) + whoAreMe(5);
  return check;
}
//--- rename_class.cpp
class Squidward{
 private:
  int h;
  int w;
  int d;
  int s;
 public:
  Squidward() {}
  Squidward(int h, int w, int d, int s): h(h), w(w), d(d), s(s) {}
  Squidward(Squidward &obj) {}
  Squidward returnedClassObj(Squidward obj){ return obj; };
  ~Squidward();
};

void whoAreMe(Squidward obj) {
  Squidward objectOne;
  Squidward *objectTwo = new Squidward(5, 10, 1, 20);
  delete objectTwo;
}
