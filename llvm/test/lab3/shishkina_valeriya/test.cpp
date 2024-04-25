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