#include <iostream>

unsigned long ic;

int func(int a, int b, int c) {
    ic = 0;

    // Function body
    int d = 0;
    for (int i = 0; i < a; i++) {
        if (d < b)
            d += c;
    }
    return d;
}

void empty_func() {}

int main() {
    int a = 10, b = 100, c = 5;
    int res = func(a, b, c);
    std::cout << "Number of machine instructions executed for func("
              << a << ", " << b << ", " << c << ") - " << ic << std::endl;
    return 0;
}