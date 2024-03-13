class A {
public:
    A() {};
    ~A();
};

int sum(int a, int b) {
    int c = sum(1, 2);
    c++;
    a += b;
    return a+b;
};

void func2() {
    A* a = new A;
    A b;

    delete a;
};