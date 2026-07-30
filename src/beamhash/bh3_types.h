#pragma once
// Three compilers see this header: the host C++ compiler, nvcc, and the Metal
// shading-language compiler. They disagree about two things only -- where the
// fixed-width integer types come from, and whether a reference or pointer needs
// an explicit address space -- so both hide behind macros and the primitives in
// bh3_primitives.h stay a single copy shared by all three backends.
//
//   MXBM_HD        function qualifier (__host__ __device__ under nvcc)
//   MXBM_THREAD    address space for thread-local references/pointers ("thread"
//                  under MSL, where it is mandatory rather than optional)
//   MXBM_CONSTEXPR constant declaration (MSL wants the address space here too)
#if defined(__METAL_VERSION__)
  #include <metal_stdlib>
  #define MXBM_HD
  #define MXBM_THREAD thread
  #define MXBM_CONSTEXPR constexpr constant
  typedef ushort uint16_t;
  typedef uint   uint32_t;
  typedef ulong  uint64_t;
  typedef uchar  uint8_t;
#else
  #include <cstdint>
  #define MXBM_THREAD
  #define MXBM_CONSTEXPR constexpr
  #if defined(__CUDACC__)
    #define MXBM_HD __host__ __device__
  #else
    #define MXBM_HD
  #endif
#endif
namespace mxbm { namespace bh3 {
MXBM_CONSTEXPR int kWorkWords     = 7;    // 448 bits
MXBM_CONSTEXPR int kNumRounds     = 5;
MXBM_CONSTEXPR int kCollisionBits = 24;
MXBM_CONSTEXPR int kNumIndices    = 32;
MXBM_CONSTEXPR int kSolutionBytes = 104;
MXBM_CONSTEXPR int kIndexPackBits = 25;   // collisionBitSize + 1
struct Elem { uint64_t w[kWorkWords]; };
} } // namespace mxbm::bh3
