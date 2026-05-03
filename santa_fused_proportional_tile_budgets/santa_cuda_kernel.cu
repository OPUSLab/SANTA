// santa_cuda_kernel.cu — SANTA core CUDA kernels.
//
// Implemented:
//   * Pass‑1 (GQA): vec specialized (GH=4, D=128, block_n=256) and WMMA fallback.
//   * m* reduction and on-device tile budgets (largest remainder).
//   * Pass‑2 (tile-local systematic, grouped) with fused atomics into out[H,D].
//   * Optional vrows metric.
//
// u_stash layout: [L, H] (index = n*H + h).

#include <cuda_runtime.h>
#include <math_constants.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <type_traits>

using namespace nvcuda;

// ---- Debug toggle (0 = off)
#ifndef SANTA_DEBUG
#define SANTA_DEBUG 0
#endif
#if SANTA_DEBUG
#define dassert_msg(cond, msg, a,b,c,d, kv,tt,gg,tid)                                  \
  do {                                                                                  \
    if (!(cond)) {                                                                      \
      printf("SANTA_ASSERT: %s | a=%d b=%d c=%d d=%d | kv=%d t=%d g=%d tid=%d\n",       \
             msg, int(a), int(b), int(c), int(d), kv, tt, gg, tid);                     \
    }                                                                                   \
  } while (0)
#else
#define dassert_msg(cond, msg, a,b,c,d, kv,tt,gg,tid) ((void)0)
#endif

// -------------------------
// 16-bit <-> fp32 helpers
// -------------------------
static __device__ __forceinline__ float ld_bf16_as_f32(const uint16_t* base, int idx) {
  uint32_t u = (uint32_t)base[idx] << 16;
  return __uint_as_float(u);
}
static __device__ __forceinline__ float ld_f16_as_f32(const __half* base, int idx) {
  return __half2float(base[idx]);
}
template <typename P, bool IS_BF16>
__device__ __forceinline__ float load16_as_f32(const P* base, int idx) {
  if constexpr (IS_BF16) return ld_bf16_as_f32(reinterpret_cast<const uint16_t*>(base), idx);
  else                    return ld_f16_as_f32 (reinterpret_cast<const __half*   >(base), idx);
}
static __device__ __forceinline__ float2 to_float2_pair(__half2 v)        { return __half22float2(v); }
static __device__ __forceinline__ float2 to_float2_pair(__nv_bfloat162 v) { return __bfloat1622float2(v); }

// -------------------------
// Warp exclusive-scan (no CUB needed)
// -------------------------
template <typename T>
static __device__ __forceinline__ T warp_exclusive_scan_sum(T v, unsigned mask=0xffffffffu) {
  T res = v;
  #pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    T n = __shfl_up_sync(mask, res, offset);
    if ((threadIdx.x & 31) >= offset) res += n;
  }
  return res - v; // exclusive
}

// =======================================================
// Pass‑1 (GQA, WMMA) — grid=(T_c, KVH)
// - Computes per-tile m_t, ell_t and stashes u = exp(s - m_t) in [L,H]
// - Q and K are bf16/f16; Q is scaled by q_scale inside the kernel.
// - K layout is NHD: [L, KVH, D] contiguous.
// =======================================================
template<typename T> struct WmmaIn;
template<> struct WmmaIn<__half>        { using type = __half; };
template<> struct WmmaIn<__nv_bfloat16> { using type = __nv_bfloat16; };

template<typename T> __device__ inline T  to_dtype(float x);
template<> __device__ inline __half        to_dtype<__half>(float x)        { return __float2half_rn(x); }
template<> __device__ inline __nv_bfloat16 to_dtype<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

// Kernel computes one (kv, tile-n) CTA; iterates GH in 16-row chunks.
template<typename TK /* __half or __nv_bfloat16 */>
__global__ void pass1_tiles_kernel_wmma_gqa_x(
    const TK*    __restrict__ Q,      // [H, D]
    const TK*    __restrict__ K,      // [L, KVH, D] (NHD)
    int H, int KVH, int L, int D,
    int block_n, int Tc, float q_scale,
    float*       __restrict__ m_tiles,    // [H, Tc] fp32
    float*       __restrict__ ell_tiles,  // [H, Tc] fp32
    TK*          __restrict__ Ustash      // [L, H] (L-major), may be nullptr to skip stash
) {
    // ---- CTA coordinates ----
    const int t  = blockIdx.x;   // tile along sequence
    const int kv = blockIdx.y;   // KV head
    const int n0 = t * block_n;

    // Per-CTA constants
    const int GH      = H / KVH;       // heads per KV group
    const int NW      = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x & 31;

    // Sanity (block_n must be multiple of 16, D multiple of 16)
    if (block_n <= 0 || (block_n & 15) != 0 || (D & 15) != 0) return;

    // Shared memory layout (dynamic)
    extern __shared__ unsigned char smem_raw[];
    TK*    smemA  = reinterpret_cast<TK*>(smem_raw);                    // [16 x 16]  row-major
    TK*    smemB  = smemA + 16 * 16;                                    // [16 x block_n]  col-major (ld=16)
    float* smemC  = reinterpret_cast<float*>(smemB + 16 * block_n);     // per-warp C tiles, row-major
    float* rowMax = smemC + (NW * 16 * 16);                             // [16 x NW]
    float* rowEll = rowMax + (16 * NW);                                 // [16 x NW]
    float* mRow   = rowEll + (16 * NW);                                 // [16]
    float* ellRow = mRow + 16;                                          // [16]

    auto Ctile_ptr  = [&](int w){ return smemC + w * 16 * 16; };
    auto rowMax_ptr = [&](int r){ return rowMax + r * NW; };
    auto rowEll_ptr = [&](int r){ return rowEll + r * NW; };
    constexpr bool IS_BF16_TK = std::is_same<TK, __nv_bfloat16>::value;

    // Load Q 16x16 panel into smemA (row-major)
    auto load_q_panel = [&](int r0, int k0) {
        for (int idx = threadIdx.x; idx < 16 * 16; idx += blockDim.x) {
            int r = idx / 16;
            int c = idx % 16;
            int hr = r0 + r;
            TK val = to_dtype<TK>(0.0f);
            if (hr < GH && (k0 + c) < D) {
                int h_abs = kv * GH + hr;
                int q_idx = h_abs * D + (k0 + c);
                float q = load16_as_f32<TK, IS_BF16_TK>(Q, q_idx);
                q *= q_scale;
                val = to_dtype<TK>(q);
            }
            smemA[r * 16 + c] = val;
        }
    };

    // Load K 16xblock_n panel into smemB (col-major with ld=16)
    auto load_k_panel = [&](int k0) {
        for (int idx = threadIdx.x; idx < 16 * block_n; idx += blockDim.x) {
            int col = idx / 16;
            int kk  = idx % 16;
            TK val  = to_dtype<TK>(0.0f);
            int n   = n0 + col;
            if (n < L && (k0 + kk) < D) {
                size_t base = ((size_t)n * (size_t)KVH + (size_t)kv) * (size_t)D + (size_t)(k0 + kk);
                val = K[base];
            }
            smemB[col * 16 + kk] = val;
        }
    };

    const int GH_tiles = (GH + 15) / 16;
    for (int tile_m = 0; tile_m < GH_tiles; ++tile_m) {
        const int r0 = tile_m * 16;
        const int valid_rows = min(16, GH - r0);

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        for (int k0 = 0; k0 < D; k0 += 16) {
            load_q_panel(r0, k0);
            load_k_panel(k0);
            __syncthreads();

            const int col_base = warp_id * 16;

            wmma::fragment<wmma::matrix_a, 16, 16, 16, typename WmmaIn<TK>::type, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, typename WmmaIn<TK>::type, wmma::col_major> b_frag;

            wmma::load_matrix_sync(a_frag, smemA, 16);
            wmma::load_matrix_sync(b_frag, smemB + col_base * 16, 16);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

            __syncthreads();
        }

        {
            float* Cw = Ctile_ptr(warp_id);
            wmma::store_matrix_sync(Cw, c_frag, 16, wmma::mem_row_major);
        }
        __syncthreads();

        // Row-wise m_t partials (each warp -> 16 values). Only lanes 0..15 participate.
        if (lane_id < 16) {
            float* Cw = Ctile_ptr(warp_id);
            const int r = lane_id;
            float m_partial = -CUDART_INF_F;
            const int col_base = warp_id * 16;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int n = n0 + col_base + j;
                if (n < L && r < valid_rows) {
                    float val = Cw[r * 16 + j];
                    m_partial = fmaxf(m_partial, val);
                }
            }
            rowMax_ptr(r)[warp_id] = m_partial;
        }
        __syncthreads();

        // Reduce across warps → final m_t for rows.
        if (warp_id == 0) {
            if (lane_id < 16) {
                const int r = lane_id;
                float m_val = -CUDART_INF_F;
                #pragma unroll
                for (int w = 0; w < NW; ++w) {
                    m_val = fmaxf(m_val, rowMax_ptr(r)[w]);
                }
                mRow[r] = (r < valid_rows) ? m_val : -CUDART_INF_F;
            }
        }
        __syncthreads();

        // Compute u = exp(s - m_t), partial ell_t, and stash u[L,H].
        if (lane_id < 16) {
            float* Cw = Ctile_ptr(warp_id);
            const int r = lane_id;
            const int h_abs = kv * GH + (r0 + r);
            float ell_partial = 0.0f;

            if (r < valid_rows && h_abs < H) {
                const float m_r = mRow[r];
                const int col_base = warp_id * 16;

                #pragma unroll
                for (int j = 0; j < 16; ++j) {
                    const int n = n0 + col_base + j;
                    if (n < L) {
                        float s = Cw[r * 16 + j];
                        float u = __expf(s - m_r);
                        ell_partial += u;

                        if (Ustash) {
                            size_t u_idx = (size_t)n * (size_t)H + (size_t)h_abs; // [L,H]
                            Ustash[u_idx] = to_dtype<TK>(u);
                        }
                    }
                }
            }
            rowEll_ptr(r)[warp_id] = ell_partial;
        }
        __syncthreads();

        // Reduce partial ell across warps and write m_t/ell_t.
        if (warp_id == 0 && lane_id < 16) {
            const int r = lane_id;
            const int h_abs = kv * GH + (r0 + r);
            if (r < valid_rows && h_abs < H) {
                float ell = 0.0f;
                #pragma unroll
                for (int w = 0; w < NW; ++w) ell += rowEll_ptr(r)[w];

                m_tiles[(size_t)h_abs * (size_t)Tc + (size_t)t]   = mRow[r];
                ell_tiles[(size_t)h_abs * (size_t)Tc + (size_t)t] = ell;
            }
        }
        __syncthreads();
    }
}

// =======================================================
// Pass‑1 GQA specialized for GH=4, D=128, block_n=256 (no WMMA)
// One CTA per (t, kv). Warp-level dot for one token at a time.
// K layout is NHD: [L, KVH, D] contiguous.
// =======================================================
template<typename TK> struct Vec2;
template<> struct Vec2<__half>        { using type = __half2; };
template<> struct Vec2<__nv_bfloat16> { using type = __nv_bfloat162; };

template<typename TK>
__global__ __launch_bounds__(256, 4)
void pass1_tiles_kernel_vec_gqa_gh4_d128_bn256(
    const TK*    __restrict__ Q,      // [H, 128]
    const TK*    __restrict__ K,      // [L, KVH, 128] (NHD)
    int H, int L, int T_c, float q_scale,
    float* __restrict__ m_tiles,      // [H, T_c] fp32
    float* __restrict__ ell_tiles,    // [H, T_c] fp32
    TK*    __restrict__ Ustash        // [L, H] or nullptr
) {
    constexpr int GH = 4;
    constexpr int D  = 128;
    constexpr int BN = 256;
    constexpr int NW = 8;   // 256 threads -> 8 warps

    const int t  = (int)blockIdx.x;
    const int kv = (int)blockIdx.y;
    const int KVH = (int)gridDim.y;

    const int h0 = kv * GH;
    if (h0 + (GH - 1) >= H) return;

    const int n0 = t * BN;
    int n1 = n0 + BN;
    if (n1 > L) n1 = L;
    const int tile_len = n1 - n0;
    if (tile_len <= 0) return;

    const int tid     = (int)threadIdx.x;
    const int lane    = tid & 31;
    const int warp_id = tid >> 5;

    __shared__ float s_scores[BN * GH];     // 256*4 = 1024 floats
    __shared__ float s_warp_red[NW * GH];   // max and sum reductions
    __shared__ float s_m[GH];

    // Each lane owns 4 dims: d0..d0+3 (32 lanes * 4 = 128 dims)
    const int d0 = lane * 4;

    // Load Q for this kv-group into registers (scaled once)
    float2 q01[GH], q23[GH];
    #pragma unroll
    for (int g = 0; g < GH; ++g) {
        const TK* qptr = Q + (size_t)(h0 + g) * (size_t)D + (size_t)d0;

        using TK2 = typename Vec2<TK>::type;
        const TK2 qa01 = *reinterpret_cast<const TK2*>(qptr);
        const TK2 qa23 = *reinterpret_cast<const TK2*>(qptr + 2);

        float2 a = to_float2_pair(qa01);
        float2 b = to_float2_pair(qa23);

        a.x *= q_scale; a.y *= q_scale;
        b.x *= q_scale; b.y *= q_scale;
        q01[g] = a;
        q23[g] = b;
    }

    // Score compute: stream K rows once; store scores to shared
    float warp_max[GH];
    #pragma unroll
    for (int g = 0; g < GH; ++g) warp_max[g] = -CUDART_INF_F;

    for (int r = warp_id; r < tile_len; r += NW) {
        const int n = n0 + r;
        const TK* kptr = K + ((size_t)n * (size_t)KVH + (size_t)kv) * (size_t)D + (size_t)d0;

        using TK2 = typename Vec2<TK>::type;
        const TK2 k01 = *reinterpret_cast<const TK2*>(kptr);
        const TK2 k23 = *reinterpret_cast<const TK2*>(kptr + 2);
        const float2 k01f = to_float2_pair(k01);
        const float2 k23f = to_float2_pair(k23);

        float acc[GH];
        #pragma unroll
        for (int g = 0; g < GH; ++g) {
            float s = 0.f;
            s = fmaf(q01[g].x, k01f.x, s);
            s = fmaf(q01[g].y, k01f.y, s);
            s = fmaf(q23[g].x, k23f.x, s);
            s = fmaf(q23[g].y, k23f.y, s);
            acc[g] = s;
        }

        // Warp reduce over D
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            #pragma unroll
            for (int g = 0; g < GH; ++g) {
                acc[g] += __shfl_down_sync(0xffffffffu, acc[g], off);
            }
        }

        if (lane == 0) {
            #pragma unroll
            for (int g = 0; g < GH; ++g) {
                const float s = acc[g];
                s_scores[r * GH + g] = s;
                warp_max[g] = fmaxf(warp_max[g], s);
            }
        }
    }

    // Publish per-warp max
    if (lane == 0) {
        #pragma unroll
        for (int g = 0; g < GH; ++g) {
            s_warp_red[warp_id * GH + g] = warp_max[g];
        }
    }
    __syncthreads();

    // Reduce m_t across warps and write m_tiles
    if (warp_id == 0 && lane < GH) {
        float m = -CUDART_INF_F;
        #pragma unroll
        for (int w = 0; w < NW; ++w) {
            m = fmaxf(m, s_warp_red[w * GH + lane]);
        }
        s_m[lane] = m;
        const int h_abs = h0 + lane;
        m_tiles[(size_t)h_abs * (size_t)T_c + (size_t)t] = m;
    }
    __syncthreads();

    // Compute u, ell, stash
    float ell_local[GH] = {0.f, 0.f, 0.f, 0.f};

    const int total = tile_len * GH;
    for (int idx = tid; idx < total; idx += (int)blockDim.x) {
        const int r = idx / GH;
        const int g = idx - r * GH;
        const float s = s_scores[r * GH + g];
        const float u = __expf(s - s_m[g]);
        ell_local[g] += u;

        if (Ustash) {
            const int n = n0 + r;
            const int h_abs = h0 + g;
            const size_t u_idx = (size_t)n * (size_t)H + (size_t)h_abs;
            Ustash[u_idx] = to_dtype<TK>(u);
        }
    }

    // Warp-reduce ell_local, then reduce across warps via shared
    #pragma unroll
    for (int g = 0; g < GH; ++g) {
        float v = ell_local[g];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down_sync(0xffffffffu, v, off);
        }
        if (lane == 0) s_warp_red[warp_id * GH + g] = v;
    }
    __syncthreads();

    if (warp_id == 0 && lane < GH) {
        float ell = 0.f;
        #pragma unroll
        for (int w = 0; w < NW; ++w) ell += s_warp_red[w * GH + lane];
        const int h_abs = h0 + lane;
        ell_tiles[(size_t)h_abs * (size_t)T_c + (size_t)t] = ell;
    }
}

// ------------------------ Pass-1 launchers (match santa_cuda.cpp) ------------------------

extern "C" void launch_pass1_tiles_vec_gqa_bf16(const void* qraw, const void* Kraw,
                                                int H, int KVH, int L, int D,
                                                int block_n, int T_c, float scale,
                                                float* m_tiles, float* ell_tiles,
                                                void* u_stash, cudaStream_t stream) {
    (void)KVH; (void)D; (void)block_n;
    dim3 grid((unsigned)T_c, (unsigned)KVH, 1);
    dim3 block(256, 1, 1);
    pass1_tiles_kernel_vec_gqa_gh4_d128_bn256<__nv_bfloat16>
        <<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(qraw),
            static_cast<const __nv_bfloat16*>(Kraw),
            H, L, T_c, scale,
            m_tiles, ell_tiles,
            static_cast<__nv_bfloat16*>(u_stash));
}

extern "C" void launch_pass1_tiles_vec_gqa_f16(const void* qraw, const void* Kraw,
                                               int H, int KVH, int L, int D,
                                               int block_n, int T_c, float scale,
                                               float* m_tiles, float* ell_tiles,
                                               void* u_stash, cudaStream_t stream) {
    (void)KVH; (void)D; (void)block_n;
    dim3 grid((unsigned)T_c, (unsigned)KVH, 1);
    dim3 block(256, 1, 1);
    pass1_tiles_kernel_vec_gqa_gh4_d128_bn256<__half>
        <<<grid, block, 0, stream>>>(
            static_cast<const __half*>(qraw),
            static_cast<const __half*>(Kraw),
            H, L, T_c, scale,
            m_tiles, ell_tiles,
            static_cast<__half*>(u_stash));
}

extern "C" void launch_pass1_tiles_wmma_gqa_bf16(const void* qraw, const void* Kraw,
                                                 int H, int KVH, int L, int D,
                                                 int block_n, int T_c, float scale,
                                                 float* m_tiles, float* ell_tiles,
                                                 void* u_stash, cudaStream_t stream) {
    dim3 grid((unsigned)T_c, (unsigned)KVH, 1);
    const int NW = block_n / 16;
    const int threads = max(1, NW) * 32;
    size_t smem_bytes = (size_t)(16*16 + 16*block_n) * sizeof(__nv_bfloat16)
                      + (size_t)(NW*16*16 + 2*16*NW + 32) * sizeof(float);

    pass1_tiles_kernel_wmma_gqa_x<__nv_bfloat16>
        <<<grid, threads, smem_bytes, stream>>>(
            static_cast<const __nv_bfloat16*>(qraw),
            static_cast<const __nv_bfloat16*>(Kraw),
            H, KVH, L, D,
            block_n, T_c, scale,
            m_tiles, ell_tiles,
            static_cast<__nv_bfloat16*>(u_stash));
}

extern "C" void launch_pass1_tiles_wmma_gqa_f16 (const void* qraw, const void* Kraw,
                                                 int H, int KVH, int L, int D,
                                                 int block_n, int T_c, float scale,
                                                 float* m_tiles, float* ell_tiles,
                                                 void* u_stash, cudaStream_t stream) {
    dim3 grid((unsigned)T_c, (unsigned)KVH, 1);
    const int NW = block_n / 16;
    const int threads = max(1, NW) * 32;
    size_t smem_bytes = (size_t)(16*16 + 16*block_n) * sizeof(__half)
                      + (size_t)(NW*16*16 + 2*16*NW + 32) * sizeof(float);

    pass1_tiles_kernel_wmma_gqa_x<__half>
        <<<grid, threads, smem_bytes, stream>>>(
            static_cast<const __half*>(qraw),
            static_cast<const __half*>(Kraw),
            H, KVH, L, D,
            block_n, T_c, scale,
            m_tiles, ell_tiles,
            static_cast<__half*>(u_stash));
}

// =======================================================
// Reduce m_star per head (max over tiles)
// =======================================================
__global__ void reduce_mstar_kernel(const float* __restrict__ m_tiles, int H, int T_c,
                                    float* __restrict__ m_star) {
  extern __shared__ unsigned char smem[];
  float* sdata = reinterpret_cast<float*>(smem);
  const int h = blockIdx.x;
  if (h >= H) return;
  const int tid = threadIdx.x, TPB = blockDim.x;

  float local_max = -FLT_MAX;
  for (int t = tid; t < T_c; t += TPB)
    local_max = fmaxf(local_max, m_tiles[h * T_c + t]);
  sdata[tid] = local_max;
  __syncthreads();
  for (int s = TPB >> 1; s > 32; s >>= 1) {
    if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    __syncthreads();
  }
  if (tid < 32) {
    volatile float* v = sdata;
    v[tid] = fmaxf(v[tid], v[tid + 32]);
    v[tid] = fmaxf(v[tid], v[tid + 16]);
    v[tid] = fmaxf(v[tid], v[tid +  8]);
    v[tid] = fmaxf(v[tid], v[tid +  4]);
    v[tid] = fmaxf(v[tid], v[tid +  2]);
    v[tid] = fmaxf(v[tid], v[tid +  1]);
  }
  __syncthreads();
  if (tid == 0) m_star[h] = sdata[0];
}

// =======================================================
// Device RNG (splitmix64) for per-tile offsets
// =======================================================
static __device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  uint64_t z = x;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  z = z ^ (z >> 31);
  return z;
}
static __device__ __forceinline__ float u01_from_seed(uint64_t seed, int h, int t) {
  uint64_t x = seed ^ ((uint64_t(h) << 32) ^ uint64_t(t));
  uint64_t r = splitmix64(x);
  double u = (r >> 11) * (1.0 / 9007199254740992.0);
  return (float)u;
}

// =======================================================
// Tile budgets (largest remainder), 1 CTA per head
// Inputs : m_tiles[h,t], ell_tiles[h,t]
// Output : m_star[h] (WRITE), S_tiles[h,t], invdelta[h,t]=St/ell_t, a0[h,t] ~ U[0,1)
// =======================================================
__global__ void tile_budgets_kernel(
    const float* __restrict__ m_tiles,
    const float* __restrict__ ell_tiles,
    float* __restrict__ m_star,          // [H] OUTPUT (may be nullptr)
    int H, int T_c, int S, uint64_t seed,
    int*   __restrict__ S_tiles,
    float* __restrict__ invdelta_tile,
    float* __restrict__ a0_tile)
{
  const int h   = (int)blockIdx.x;
  if (h >= H) return;

  const int tid = (int)threadIdx.x;
  const int TPB = (int)blockDim.x;

  extern __shared__ unsigned char smem_raw[];
  float* frac = reinterpret_cast<float*>(smem_raw);          // size T_c
  int*   idx  = reinterpret_cast<int*>(frac + T_c);          // size T_c

  // ---- NEW: compute m_star[h] = max_t m_tiles[h,t]
  // Also cache m_tiles[h,t] into frac[t] so we can reuse it when forming W_t.
  __shared__ float s_warp_max[32]; // enough for up to 1024 threads (32 warps)

  float local_max = -FLT_MAX;
  for (int t = tid; t < T_c; t += TPB) {
    const float mt = m_tiles[h * T_c + t];
    frac[t] = mt;                    // cache m_t for the next phase
    local_max = fmaxf(local_max, mt);
  }

  // Warp reduce max
  float wmax = local_max;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    wmax = fmaxf(wmax, __shfl_down_sync(0xffffffffu, wmax, off));
  }
  if ((tid & 31) == 0) s_warp_max[tid >> 5] = wmax;
  __syncthreads();

  // Reduce across warps using warp 0
  if (tid < 32) {
    const int num_warps = (TPB + 31) >> 5;
    float v = (tid < num_warps) ? s_warp_max[tid] : -FLT_MAX;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
    }
    if (tid == 0) s_warp_max[0] = v;
  }
  __syncthreads();

  const float mstar = s_warp_max[0];
  if (tid == 0 && m_star) m_star[h] = mstar;

  // ---- ORIGINAL kernel logic continues, but now uses local `mstar`
  // (1) Accumulate Z = sum_t exp(m_t - m*) * ell_t   (store W_t temporarily in frac[t])
  __shared__ float sZ;
  if (tid == 0) sZ = 0.0f;
  __syncthreads();

  float z_local = 0.0f;
  for (int t = tid; t < T_c; t += TPB) {
    const float mt  = frac[t]; // cached m_tiles[h,t]
    const float ell = ell_tiles[h * T_c + t];
    const float Wt  = __expf(mt - mstar) * ell;
    frac[t] = Wt;              // overwrite cache with W_t (same as old kernel)
    z_local += Wt;
  }
  atomicAdd(&sZ, z_local);
  __syncthreads();

  const float Z = sZ;

  // Trivial case: no samples or zero mass.
  if (S <= 0 || !(Z > 0.0f)) {
    for (int t = tid; t < T_c; t += TPB) {
      S_tiles[h * T_c + t]       = 0;
      invdelta_tile[h * T_c + t] = 0.0f;
      a0_tile[h * T_c + t]       = u01_from_seed(seed, h, t);
    }
    return;
  }

  // (2) Integer bases b_t and fractional parts f_t (with tiny deterministic jitter)
  __shared__ int s_sum_base;
  if (tid == 0) s_sum_base = 0;
  __syncthreads();

  int base_local = 0;
  for (int t = tid; t < T_c; t += TPB) {
    const float Wt = frac[t]; // W_t from phase (1)
    const float q  = (float)S * (Wt / Z);
    const int   b  = (int)floorf(q);
    S_tiles[h * T_c + t] = b;
    base_local          += b;

    float ft     = q - (float)b;
    float jitter = 1.0e-7f * u01_from_seed(seed, h, t);
    frac[t] = ft + jitter;
    idx[t]  = t;
  }
  atomicAdd(&s_sum_base, base_local);
  __syncthreads();

  int rem = S - s_sum_base;
  if (rem < 0)   rem = 0;
  if (rem > T_c) rem = T_c;

  // (3) Largest‑remainder via parallel bitonic sort on {frac, idx} (descending)
  int N = 1; while (N < T_c) N <<= 1;
  for (int k = 2; k <= N; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int i = tid; i < N; i += TPB) {
        int ixj = i ^ j;
        if (ixj > i) {
          const bool up = ((i & k) != 0);  // 'up' flipped => final order is DESC
          float ai = (i    < T_c) ? frac[i] : -CUDART_INF_F;
          float aj = (ixj  < T_c) ? frac[ixj] : -CUDART_INF_F;
          int   ii = (i    < T_c) ? idx[i]  : -1;
          int   ij = (ixj  < T_c) ? idx[ixj]: -1;

          const bool swap = (ai > aj) == up;
          if (swap) { float tfi = ai; ai = aj; aj = tfi; int tii = ii; ii = ij; ij = tii; }

          if (i   < T_c) { frac[i]   = ai; idx[i]   = ii; }
          if (ixj < T_c) { frac[ixj] = aj; idx[ixj] = ij; }
        }
      }
      __syncthreads();
    }
  }

  // Give +1 to the top 'rem' fractional parts
  for (int r = tid; r < rem; r += TPB) {
    const int best_t = (r < T_c) ? idx[r] : -1;
    if (best_t >= 0) {
      S_tiles[h * T_c + best_t] += 1;
    }
  }
  __syncthreads();

  // (4) Finalize invdelta (= St/ell_t) and per‑tile a0 in [0,1)
  const float eps = 1e-20f;
  for (int t = tid; t < T_c; t += TPB) {
    const float ell = ell_tiles[h * T_c + t];
    const int   St  = S_tiles[h * T_c + t];
    invdelta_tile[h * T_c + t] = (St > 0 && ell > eps) ? ((float)St / ell) : 0.0f;
    a0_tile[h * T_c + t]       = u01_from_seed(seed, h, t);
  }
}


// =======================================================
// Pass‑2 (tile-local systematic, grouped; fused atomics per (h,d))
// grid=(KVH, T_c), block=(TPB over D, GH over heads in KV group)
// V layout is NHD: [L, KVH, D] contiguous.
// =======================================================
template <typename VPtr, bool IS_BF16>
__global__ void pass2_grouped_tile_fused_atomic_kernel_x(
    const VPtr*  __restrict__ V,            // [L, KVH, D]
    const VPtr*  __restrict__ Ustash,       // [L, H] — u = exp(s - m_t)
    int H, int KVH, int L, int D,
    int block_n, int T_c,
    const float* __restrict__ invdelta_tile,// [H, T_c] (St/ell_t)
    const float* __restrict__ a0_tile,      // [H, T_c] offset in [0,1)
    const int*   __restrict__ S_tiles,      // [H, T_c] (bookkeeping only)
    float* __restrict__ out,                // [H, D] fp32
    int*   __restrict__ rowmask)            // [KVH, L] (optional)
{
  (void)S_tiles;

  const int kv  = blockIdx.x;
  const int t   = blockIdx.y;
  const int GH  = H / KVH;
  const int g   = threadIdx.y;  // head within KV group
  const int tid = threadIdx.x;
  if (g >= GH || t >= T_c) return;

  const int h   = kv * GH + g;
  const int n0  = t * block_n;
  const int n1  = min(n0 + block_n, L);
  const int tile_len = n1 - n0;
  if (tile_len <= 0) return;

  extern __shared__ unsigned char smem_raw[];
  unsigned char* p = smem_raw;
  float*   acc      = reinterpret_cast<float*>(p);                             p += (size_t)GH * D * sizeof(float);
  int16_t* c_tile   = reinterpret_cast<int16_t*>(p);                           p += (size_t)GH * (size_t)block_n * sizeof(int16_t);
  int*     emit_idx = reinterpret_cast<int*>(p);                               // size block_n
  __shared__ int s_emit_len;

  for (int d = tid; d < D; d += blockDim.x) acc[g * D + d] = 0.f;
  __syncthreads();

  // (1) COUNTS pass: one warp per head computes integerized counts
  const float invdel = invdelta_tile[h * T_c + t];
  const float a0     = a0_tile[h * T_c + t];
  const int lane     = threadIdx.x & 31;
  const int warp_id  = threadIdx.x >> 5;

  if (warp_id == 0) {
    float carry = 0.f;
    for (int base = 0; base < tile_len; base += 32) {
      const int r = base + lane;
      float x = 0.f;
      if (r < tile_len && invdel > 0.f) {
        const int n  = n0 + r;
        const float u = load16_as_f32<VPtr, IS_BF16>(Ustash, n * H + h);
        x = u * invdel;
      }
      float pre = warp_exclusive_scan_sum<float>(x);

      const int remain    = tile_len - base;
      const int last_lane = (remain > 0) ? min(31, remain - 1) : 0;
      float pre_g = carry + pre;
      float cum_g = pre_g + x;

      if (r < tile_len) {
        float f_prev = floorf(a0 + pre_g);
        float f_curr = floorf(a0 + cum_g);
        int c = (int)(f_curr - f_prev);
        c = (c < 0 ? 0 : c);
        c = (c > 32767 ? 32767 : c);
        c_tile[g * block_n + r] = (int16_t)c;
      }

      float chunk_total = __shfl_sync(0xffffffff, pre + x, last_lane);
      if (lane == 0) carry += chunk_total;
      carry = __shfl_sync(0xffffffff, carry, 0);
    }
  }
  __syncthreads();

  // (2) COMPACTION: head 0 builds emit_idx for rows with any emission across GH
  if (g == 0 && warp_id == 0) {
    int emit_carry = 0;
    for (int base = 0; base < tile_len; base += 32) {
      const int r = base + lane;
      int flag = 0;
      if (r < tile_len) {
        int sum = 0;
        for (int gg = 0; gg < GH; ++gg)
          sum += (int)c_tile[gg * block_n + r];
        flag = (sum > 0);
      }
      int pos_local = warp_exclusive_scan_sum<int>(flag);
      const int remain    = tile_len - base;
      const int last_lane = (remain > 0) ? min(31, remain - 1) : 0;
      int chunk_sum = __shfl_sync(0xffffffff, pos_local + flag, last_lane);

      if (r < tile_len && flag) {
        const int n  = n0 + r;
        const int pos = emit_carry + pos_local;
        emit_idx[pos] = n;
        if (rowmask) atomicOr(&rowmask[kv * L + n], 1);
      }
      if (lane == 0) emit_carry += chunk_sum;
      emit_carry = __shfl_sync(0xffffffff, emit_carry, 0);
    }
    if (lane == 0) s_emit_len = emit_carry;
  }
  __syncthreads();

  const int emit_len = s_emit_len;
  if (emit_len == 0) return;

  // (3) V pass: each head loads V directly for rows it uses.
  for (int i = 0; i < emit_len; ++i) {
    const int n = emit_idx[i];
    const int r_local = n - n0;
    dassert_msg(0 <= r_local && r_local < tile_len, "r_local OOB", r_local, tile_len, n, n0, kv,t,g,tid);

    const int c = (int)c_tile[g * block_n + r_local];
    if (c != 0) {
      const float cf = (float)c;
      const int v_off = (n * KVH + kv) * D;
      for (int d = tid; d < D; d += blockDim.x) {
        acc[g * D + d] += cf * load16_as_f32<VPtr, IS_BF16>(V, v_off + d);
      }
    }
  }

  // (4) fused writes
  const int base = h * D;
  for (int d = tid; d < D; d += blockDim.x) {
    atomicAdd(&out[base + d], acc[g * D + d]);
  }
}

// =======================================================
// vrows reduction
// =======================================================
__global__ void reduce_vrows_kernel(const int* __restrict__ rowmask,
                                    int KVH, int L,
                                    int* __restrict__ vrows) {
  extern __shared__ unsigned char smem[];
  int* sdata = reinterpret_cast<int*>(smem);
  const int kv = blockIdx.x;
  if (kv >= KVH) return;
  const int tid = threadIdx.x, TPB = blockDim.x;

  int local_sum = 0;
  for (int n = tid; n < L; n += TPB)
    local_sum += (rowmask[kv * L + n] != 0);
  sdata[tid] = local_sum;
  __syncthreads();

  for (int s = TPB >> 1; s > 32; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid < 32) {
    volatile int* v = sdata;
    v[tid] += v[tid + 32];
    v[tid] += v[tid + 16];
    v[tid] += v[tid +  8];
    v[tid] += v[tid +  4];
    v[tid] += v[tid +  2];
    v[tid] += v[tid +  1];
  }
  __syncthreads();
  if (tid == 0) vrows[kv] = sdata[0];
}

// =======================================================
// Launchers
// =======================================================
extern "C" void launch_tile_budgets(
    const float* m_tiles, const float* ell_tiles, float* m_star,   // m_star is OUTPUT now
    int H, int T_c, int S, uint64_t seed,
    int* S_tiles, float* invdelta_tile, float* a0_tile,
    cudaStream_t stream)
{
  dim3 grid(H), block(128);
  size_t shmem = static_cast<size_t>(T_c) * (sizeof(float) + sizeof(int));
  tile_budgets_kernel<<<grid, block, shmem, stream>>>(
      m_tiles, ell_tiles, m_star, H, T_c, S, seed,
      S_tiles, invdelta_tile, a0_tile);
}

extern "C" void launch_pass2_grouped_fused_atomic_bf16(
    const void* Vraw, const void* Uraw,
    int H, int KVH, int L, int D,
    int block_n, int T_c,
    const float* invdelta_tile, const float* a0_tile, const int* S_tiles,
    float* out, int* rowmask,
    cudaStream_t stream) {
  const int GH = H / KVH;
  const int TPB = 128;
  dim3 block(TPB, GH);
  dim3 grid(KVH, T_c);
  size_t shmem = (size_t)GH * D * sizeof(float)
             + (size_t)GH * (size_t)block_n * sizeof(int16_t)
             + (size_t)block_n * sizeof(int);

  const uint16_t* V = static_cast<const uint16_t*>(Vraw);
  const uint16_t* U = static_cast<const uint16_t*>(Uraw);
  pass2_grouped_tile_fused_atomic_kernel_x<uint16_t, true>
      <<<grid, block, shmem, stream>>>(
          V, U, H, KVH, L, D, block_n, T_c,
          invdelta_tile, a0_tile, S_tiles, out, rowmask);
}

extern "C" void launch_pass2_grouped_fused_atomic_f16(
    const void* Vraw, const void* Uraw,
    int H, int KVH, int L, int D,
    int block_n, int T_c,
    const float* invdelta_tile, const float* a0_tile, const int* S_tiles,
    float* out, int* rowmask,
    cudaStream_t stream) {
  const int GH = H / KVH;
  const int TPB = 128;
  dim3 block(TPB, GH);
  dim3 grid(KVH, T_c);
  size_t shmem = (size_t)GH * D * sizeof(float)
             + (size_t)GH * (size_t)block_n * sizeof(int16_t)
             + (size_t)block_n * sizeof(int);

  const __half* V = static_cast<const __half*>(Vraw);
  const __half* U = static_cast<const __half*>(Uraw);
  pass2_grouped_tile_fused_atomic_kernel_x<__half, false>
      <<<grid, block, shmem, stream>>>(
          V, U, H, KVH, L, D, block_n, T_c,
          invdelta_tile, a0_tile, S_tiles, out, rowmask);
}

extern "C" void launch_reduce_vrows(const int* rowmask,
                                   int KVH, int L,
                                   int* vrows,
                                   cudaStream_t stream) {
  const int TPB = 256;
  dim3 block(TPB), grid(KVH);
  size_t shmem = TPB * sizeof(int);
  reduce_vrows_kernel<<<grid, block, shmem, stream>>>(rowmask, KVH, L, vrows);
}

// =======================================================
// Cast + scale final output: fp32 [H,D] -> bf16/f16 [H,D]
// =======================================================
template <typename OUT_T>
__global__ void cast_scale_out_kernel(const float* __restrict__ in,
                                      OUT_T* __restrict__ out,
                                      int total, float scale) {
  int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < total) {
    float x = in[idx] * scale;
    out[idx] = to_dtype<OUT_T>(x);
  }
}

extern "C" void launch_cast_scale_out_bf16(const float* in, void* out,
                                          int H, int D, float scale,
                                          cudaStream_t stream) {
  const int total = H * D;
  const int TPB = 256;
  const int blocks = (total + TPB - 1) / TPB;
  cast_scale_out_kernel<__nv_bfloat16><<<blocks, TPB, 0, stream>>>(
      in, reinterpret_cast<__nv_bfloat16*>(out), total, scale);
}

extern "C" void launch_cast_scale_out_f16(const float* in, void* out,
                                         int H, int D, float scale,
                                         cudaStream_t stream) {
  const int total = H * D;
  const int TPB = 256;
  const int blocks = (total + TPB - 1) / TPB;
  cast_scale_out_kernel<__half><<<blocks, TPB, 0, stream>>>(
      in, reinterpret_cast<__half*>(out), total, scale);
}
