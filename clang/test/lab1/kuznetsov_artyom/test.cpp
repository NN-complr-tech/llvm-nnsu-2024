

int factorial(int value) {
  return value <= 1 ? 1 : value * factorial(value - 1);
}

struct Point3D {
  double m_x{};
  double m_y{};
  double m_z{};
};