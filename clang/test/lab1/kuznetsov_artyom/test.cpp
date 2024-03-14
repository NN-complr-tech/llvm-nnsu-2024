// RUN: %clang_cc1 -load %llvmshlibdir/PrintClassMembersPlugin%pluginext -plugin
// pcm_plugin %s 1>&1 | FileCheck %s

namespace {
// CHECK: Point3D (struct)
struct Point3D {
  // CHECK-NEXT: |_ m_x (double|public)
  double m_x{};
  // CHECK-NEXT: |_ m_y (double|public)
  double m_y{};
  // CHECK-NEXT: |_ m_z (double|public)
  double m_z{};
};

// CHECK: Pair (struct|template)
template <typename T1, typename T2> struct Pair {
  // CHECK-NEXT: |_ first (T1|public)
  T1 first{};
  // CHECK-NEXT: |_ second (T2|public)
  T2 second{};
};

// CHECK: EmptyClass (class)
class EmptyClass {};

// CHECK: NodeList (struct|template)
template <typename T> struct NodeList {
  // CHECK-NEXT: |_ data (T|public)
  T data{};
  // CHECK-NEXT: |_ next (NodeList<T> *|public)
  NodeList<T> *next{};
};

// CHECK: CheckSpecifiers (class)
class CheckSpecifiers {
public:
  // CHECK-NEXT: |_ publicMember (Point3D|public)
  Point3D publicMember{};

protected:
  // CHECK-NEXT: |_ protectedMember (Pair<int, NodeList<Pair<short, char> >
  // >|protected)
  Pair<int, NodeList<Pair<short, char>>> protectedMember{};

private:
  // CHECK-NEXT: |_ privateMember (EmptyClass|private)
  EmptyClass privateMember{};
};

// CHECK: User (struct)
struct User {
  // CHECK-NEXT: |_ m_id (int|public)
  int m_id{};

  // CHECK: Human (struct)
  struct Human {
    // CHECK-NEXT: |_ m_age (int|public)
    int m_age{};
    // CHECK-NEXT: |_ m_cash (int|public)
    int m_cash{};
  };
};

class CheckStatic {
  static float staticMember;
};

float CheckStatic::staticMember = 0;

class CheckConst {
  const char constMember{};
};

} // namespace