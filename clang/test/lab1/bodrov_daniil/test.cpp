// RUN: %clang_cc1 -load %llvmshlibdir/ClassFieldPrinter%pluginext -plugin class-field-printer %s 1>&1 | FileCheck %s

// CHECK: Empty (class)
class Empty {};

// CHECK: Data (class)
class Data {
public:
  int a; // CHECK-NEXT: |_ a (int|public)
  static int b; // CHECK-NEXT: |_ b (int|public|static)
  void func() {} // CHECK-NEXT: |_ func (void (void)|public|method)
};

// CHECK: Template (class|template)
template <typename T>
class Template {
public:
  T value; // CHECK-NEXT: |_ value (T|public)
};

// CHECK: TestClass (class)
class TestClass {
public:
  int publicInt; // CHECK-NEXT: |_ publicInt (int|public)
  static int publicStaticInt; // CHECK-NEXT: |_ publicStaticInt (int|public|static)
  void publicFunc() {} // CHECK-NEXT: |_ publicFunc (void (void)|public|method)

private:
  int privateInt; // CHECK-NEXT: |_ privateInt (int|private)
  static int privateStaticInt; // CHECK-NEXT: |_ privateStaticInt (int|private|static)
  void privateFunc() {} // CHECK-NEXT: |_ privateFunc (void (void)|private|method)
};

// CHECK: AnotherTestClass (class)
class AnotherTestClass {
public:
  double publicDouble; // CHECK-NEXT: |_ publicDouble (double|public)
  static double publicStaticDouble; // CHECK-NEXT: |_ publicStaticDouble (double|public|static)
  double publicFunc(); // CHECK-NEXT: |_ publicFunc (double (void)|public|method)

  Template<TestClass> t_TestClass; // CHECK-NEXT: |_ t_TestClass (Template<TestClass>|public)

private:
  double privateDouble; // CHECK-NEXT: |_ privateDouble (double|private)
  static double privateStaticDouble; // CHECK-NEXT: |_ privateStaticDouble (double|private|static)
  float privateFunc(const char* str); // CHECK-NEXT: privateFunc (float (const char *)|private|method)
};

// CHECK: DerivedClass (class)
class DerivedClass : public TestClass {
public:
  int derivedPublicInt; // CHECK-NEXT: |_ derivedPublicInt (int|public)
  static int derivedPublicStaticInt; // CHECK-NEXT: |_ derivedPublicStaticInt (int|public|static)
  TestClass derivedPublicFunc(Template<int> data, Template<const char*> data_str); // CHECK-NEXT: |_ derivedPublicFunc (TestClass (Template<int>, Template<const char *>)|public|method)

  AnotherTestClass classField; // CHECK-NEXT: |_ classField (AnotherTestClass|public)

private:
  int derivedPrivateInt; // CHECK-NEXT: |_ derivedPrivateInt (int|private)
  static int derivedPrivateStaticInt; // CHECK-NEXT: |_ derivedPrivateStaticInt (int|private|static)
  void derivedPrivateFunc(); // CHECK-NEXT: |_ derivedPrivateFunc (void (void)|private|method)
  Template<Template<TestClass>> t_TestClass; // CHECK-NEXT: |_ t_TestClass (Template<Template<TestClass> >|private)
};

// CHECK: MyStruct (struct)
struct MyStruct {
  int x; // CHECK-NEXT: |_ x (int|public)
  double y; // CHECK-NEXT: |_ y (double|public)
};