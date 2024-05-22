int func_1(int a, int b, int c) {
    static long ic = 0;

    // Function body
    int d = 0;
    for (int i = 0; i < a; i++) {
        if (d < b)
            d += c;
    }
    return d;
}

int func_2(int a, int b) {
    static long ic = 0;

    // Function body
    int d = 0;
    for (int i = 0; i < a; i++) {
        d += func_1(i, a, b);
    }
    return d;
}