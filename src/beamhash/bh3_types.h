#pragma once
#include <cstdint>
#if defined(__CUDACC__)
  #define MXBM_HD __host__ __device__
#else
  #define MXBM_HD
#endif
namespace mxbm { namespace bh3 {
constexpr int kWorkWords     = 7;    // 448 bits
constexpr int kNumRounds     = 5;
constexpr int kCollisionBits = 24;
constexpr int kNumIndices    = 32;
constexpr int kSolutionBytes = 104;
constexpr int kIndexPackBits = 25;   // collisionBitSize + 1
struct Elem { uint64_t w[kWorkWords]; };
} } // namespace mxbm::bh3
