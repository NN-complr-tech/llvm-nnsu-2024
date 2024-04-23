#include <immintrin.h>

__m128d muladd_test1(__m128d a, __m128d b, __m128d c) { return a * b + c; } // +

__m128d muladd_test2(__m128d a, __m128d b, __m128d c) { // +
  __m128d tmp = a * b;
  tmp = tmp + c;
  return tmp;
}

__m128d muladd_test3(__m128d a, __m128d b, __m128d c) { // +
  __m128d tmp = a * c + b;
  return tmp * c + b;
}

__m128d muladd_test4(__m128d a, __m128d b, __m128d c) { // -
  __m128d tmp = a * c;
  tmp = tmp / b;
  return tmp + b;
}
__m128d muladd_test5(__m128d a, __m128d b, __m128d c) {
  __m128d tmp = a * b;
  __m128d tmp2 = b + a - c;
  return tmp + c + tmp2;
}
__m128d muladd_test6(__m128d a, __m128d b, __m128d c) {
  __m128d tmp = a * b;
  __m128d tmp2 = tmp + c;
  return tmp * c;
}
