#include "util.h"

#include <malloc.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>

namespace efanna2e {

void GenRandom(std::mt19937& rng, unsigned* addr, unsigned size, unsigned N) {
  for (unsigned i = 0; i < size; ++i) {
    addr[i] = rng() % (N - size);
  }
  std::sort(addr, addr + size);
  for (unsigned i = 1; i < size; ++i) {
    if (addr[i] <= addr[i - 1]) {
      addr[i] = addr[i - 1] + 1;
    }
  }
  unsigned off = rng() % N;
  for (unsigned i = 0; i < size; ++i) {
    addr[i] = (addr[i] + off) % N;
  }
}

float* data_align(float* data_ori, unsigned point_num, unsigned& dim) {
#ifdef __GNUC__
#ifdef __AVX__
#define DATA_ALIGN_FACTOR 8
#else
#ifdef __SSE2__
#define DATA_ALIGN_FACTOR 4
#else
#define DATA_ALIGN_FACTOR 1
#endif
#endif
#endif
  float* data_new = 0;
  unsigned new_dim =
      (dim + DATA_ALIGN_FACTOR - 1) / DATA_ALIGN_FACTOR * DATA_ALIGN_FACTOR;
#ifdef __APPLE__
  data_new = new float[(size_t)new_dim * (size_t)point_num];
#else
  data_new =
      (float*)memalign(DATA_ALIGN_FACTOR * 4,
                       (size_t)point_num * (size_t)new_dim * sizeof(float));
#endif

  for (size_t i = 0; i < point_num; i++) {
    memcpy(data_new + i * new_dim, data_ori + i * dim, dim * sizeof(float));
    memset(data_new + i * new_dim + dim, 0, (new_dim - dim) * sizeof(float));
  }

  dim = new_dim;

#ifdef __APPLE__
  delete[] data_ori;
#else
  free(data_ori);
#endif

  return data_new;
}

}  // namespace efanna2e
