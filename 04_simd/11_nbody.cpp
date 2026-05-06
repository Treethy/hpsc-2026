#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 16;
  alignas(64) float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  for(int i=0; i<N; i++) {
    __m512 xi = _mm512_set1_ps(x[i]);
    __m512 yi = _mm512_set1_ps(y[i]);
    __m512 xj = _mm512_load_ps(x);
    __m512 yj = _mm512_load_ps(y);
    __m512 mj = _mm512_load_ps(m);

    __m512 rx = _mm512_sub_ps(xi, xj);
    __m512 ry = _mm512_sub_ps(yi, yj);

    __m512 r2 = _mm512_add_ps(_mm512_mul_ps(rx, rx), _mm512_mul_ps(ry, ry));

    __mmask16 mask = (__mmask16)(0xffffu ^ (1u << i)); // i != j
    r2 = _mm512_mask_blend_ps(mask, _mm512_set1_ps(1.0f), r2);

    __m512 inv_r = _mm512_rsqrt14_ps(r2);
    __m512 inv_r3 = _mm512_mul_ps(_mm512_mul_ps(inv_r, inv_r), inv_r);
    __m512 coef = _mm512_mul_ps(mj, inv_r3);

    __m512 fx_vec = _mm512_mul_ps(rx, coef);
    __m512 fy_vec = _mm512_mul_ps(ry, coef);
    fx_vec = _mm512_maskz_mov_ps(mask, fx_vec);
    fy_vec = _mm512_maskz_mov_ps(mask, fy_vec);

    fx[i] -= _mm512_reduce_add_ps(fx_vec);
    fy[i] -= _mm512_reduce_add_ps(fy_vec);
    
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
