void foo4() {
  float a = 1.0f;
  if (a < 2.0f)
    a += 1.0f;
}

void bar4() {
  int a = 0;
  if (a < 1)
    foo4();
  a++;
}