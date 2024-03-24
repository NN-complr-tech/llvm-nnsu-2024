int j = 2;

void a() {
    int c = 10;
    for (int i = 0; i < 10; i++) {
        c++;
    }
    int q = c + 42;
}

int b(int a) {
    int c = 10;
    for (; j < a; j++) {
        c++;
    }
    int q = c + 42;
    return q;
}


int v() {
    int c = 10;
    a();
    for (; ; j++) {
        break;
    }
    b(c);
    int q = c + 42;
    return q;
}

int main() { return 0; }