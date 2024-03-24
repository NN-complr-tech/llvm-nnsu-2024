int fact(int value) { return value <= 1 ? 1 : value * fact(value - 1); }

int func(bool pred) {
  if (pred) {
    int a = 10;
    int b = 20;
    return a + b;
  }

  int a = 100;
  int b = 200;
  int c = 300;
  return a + b + c;
}
