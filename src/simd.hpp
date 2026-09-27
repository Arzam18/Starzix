// clang-format off

#pragma once

#include "utils.hpp"

#if defined(__aarch64__) && defined(__ARM_NEON)
    #include <arm_neon.h>
#else
    #include <immintrin.h>
#endif

// Starzix's NNUE evaluation uses 16-bit hidden neurons and 32-bit
// accumulation.  On x86, one Vec type can represent both because the
// integer operations use the same 256/512-bit register type.  NEON has
// distinct 16-bit and 32-bit vector types, so keep the two types explicit.

#if (defined(__AVX512F__) && defined(__AVX512BW__)) || defined(__AVX2__)

    #if defined(__AVX512F__) && defined(__AVX512BW__)
        using Vec = __m512i;
    #else // AVX2
        using Vec = __m256i;
    #endif

    using Vec32 = Vec;

    constexpr Vec setEpi16(const i16 x)
    {
        #if defined(__AVX512F__) && defined(__AVX512BW__)
            return _mm512_set1_epi16(x);
        #else // AVX2
            return _mm256_set1_epi16(x);
        #endif
    }

    constexpr Vec loadVec(const Vec* vecPtr)
    {
        #if defined(__AVX512F__) && defined(__AVX512BW__)
            return _mm512_load_si512(vecPtr);
        #else // AVX2
            return _mm256_load_si256(vecPtr);
        #endif
    }

    constexpr Vec clampVec(const Vec vec, const Vec minVec, const Vec maxVec)
    {
        #if defined(__AVX512F__) && defined(__AVX512BW__)
            return _mm512_min_epi16(_mm512_max_epi16(vec, minVec), maxVec);
        #else // AVX2
            return _mm256_min_epi16(_mm256_max_epi16(vec, minVec), maxVec);
        #endif
    }

    constexpr Vec mulloEpi16(const Vec a, const Vec b)
    {
        #if defined(__AVX512F__) && defined(__AVX512BW__)
            return _mm512_mullo_epi16(a, b);
        #else // AVX2
            return _mm256_mullo_epi16(a, b);
        #endif
    }

    constexpr Vec32 maddEpi16(const Vec a, const Vec b)
    {
        #if defined(__AVX512F__) && defined(__AVX512BW__)
            return _mm512_madd_epi16(a, b);
        #else // AVX2
            return _mm256_madd_epi16(a, b);
        #endif
    }

    constexpr Vec32 addEpi32(const Vec32 a, const Vec32 b)
    {
        #if defined(__AVX512F__) && defined(__AVX512BW__)
            return _mm512_add_epi32(a, b);
        #else // AVX2
            return _mm256_add_epi32(a, b);
        #endif
    }

    // Adds the i32's in vec, returning an i32
    inline i32 sumVec(const Vec32 vec)
    {
        #if defined(__AVX512F__) && defined(__AVX512BW__)
            return _mm512_reduce_add_epi32(vec);
        #else // AVX2
            // Get the lower and upper half of the register:
            __m128i xmm0 = _mm256_castsi256_si128(vec);
            __m128i xmm1 = _mm256_extracti128_si256(vec, 1);

            // Add the lower and upper half vertically:
            xmm0 = _mm_add_epi32(xmm0, xmm1);

            // Get the upper half of the result:
            xmm1 = _mm_unpackhi_epi64(xmm0, xmm0);

            // Add the lower and upper half vertically:
            xmm0 = _mm_add_epi32(xmm0, xmm1);

            // Shuffle the result so that the lower 32-bits
            // are directly above the second-lower 32-bits:
            xmm1 = _mm_shuffle_epi32(xmm0, _MM_SHUFFLE(2, 3, 0, 1));

            // Add the lower 32-bits to the second-lower 32-bits vertically:
            xmm0 = _mm_add_epi32(xmm0, xmm1);

            // Add the lower 32-bits to the second-lower 32-bits vertically:
            return _mm_cvtsi128_si32(xmm0);
        #endif
    }

#elif defined(__aarch64__) && defined(__ARM_NEON)

    // AArch64 NEON registers are 128-bit.  Eight i16 lanes are used for
    // the NNUE hidden layer, while four i32 lanes are used for the sum.
    using Vec = int16x8_t;
    using Vec32 = int32x4_t;

    inline Vec setEpi16(const i16 x)
    {
        return vdupq_n_s16(x);
    }

    inline Vec loadVec(const Vec* vecPtr)
    {
        return vld1q_s16(reinterpret_cast<const i16*>(vecPtr));
    }

    inline Vec clampVec(const Vec vec, const Vec minVec, const Vec maxVec)
    {
        return vminq_s16(vmaxq_s16(vec, minVec), maxVec);
    }

    inline Vec mulloEpi16(const Vec a, const Vec b)
    {
        return vmulq_s16(a, b);
    }

    // Equivalent to _mm*_madd_epi16:
    // multiply adjacent i16 pairs and add them into i32 lanes.
    inline Vec32 maddEpi16(const Vec a, const Vec b)
    {
        const int32x4_t lo = vmull_s16(vget_low_s16(a), vget_low_s16(b));
        const int32x4_t hi = vmull_s16(vget_high_s16(a), vget_high_s16(b));
        return vpaddq_s32(lo, hi);
    }

    inline Vec32 addEpi32(const Vec32 a, const Vec32 b)
    {
        return vaddq_s32(a, b);
    }

    inline i32 sumVec(const Vec32 vec)
    {
        return vaddvq_s32(vec);
    }

#else

    // Scalar fallback is selected by nnue.hpp.  Vec is retained only so
    // the NNUE data structures remain well-formed on unsupported targets.
    using Vec = __m128i;
    using Vec32 = Vec;

#endif
