
struct Point3D {
  double m_x{};
  double m_y{};
  double m_z{};
};

template <typename T1, typename T2> struct Pair {
  T1 first{};
  T2 second{};
};

struct User {
  int m_id{};

  struct Human {
    int m_age{};
    int m_cash{};
  };
};

class EmptyClass {};

template <typename T> struct NodeList {
  T data{};
  NodeList<T> *next{};
};

class CheckSpecifiers {
public:
  Point3D publicMember{};

protected:
  User::Human protectedMember{};

private:
  EmptyClass privateMember{};
};
