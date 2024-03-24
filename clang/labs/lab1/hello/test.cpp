void func1() {}

int func2(int value) { return 0; }

int factorial(int value) {
  return value <= 1 ? 1 : value * factorial(value - 1);
}