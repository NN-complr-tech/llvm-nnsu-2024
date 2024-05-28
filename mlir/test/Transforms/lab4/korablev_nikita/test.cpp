int sum(int a, int b) {
    return a + b;
}

int mult(int a, int b) {
    return a*b;
}

int main() {
    int a = 5;

    a = sum(a, 2);
    a = mult(a, 5);
    a = sum(a, a);
    
    return 0;
}
