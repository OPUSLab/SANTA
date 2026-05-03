// santa_cuda_kernel.cu — fused S^2ANTA-flash kernels with BN in {32,64,256}.
//
// Batched contiguous-KV update:
//   - Preserves the existing fused tile / finalize / vrows logic.
//   - Adds a leading batch dimension in-grid (grid.z = B).
//   - Supports a separate L_valid and L_storage so Python can keep a fixed contiguous
//     cache [B, T_storage, KVH, D] and advance valid_len in place during decode.
//
// Supported configurations:
//  - GH == 4
//  - D  == 128
//  - BN == 32 or 64 or 256 (specialized kernels)
//
// Q layout:
//   - single: [H, D]
//   - batch : [B, H, D]
// KV cache layout:
//   - single: [L_storage, KVH, D] contiguous
//   - batch : [B, L_storage, KVH, D] contiguous

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <cmath>
#include <cstdint>
#include <type_traits>

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
  else                   return ld_f16_as_f32(reinterpret_cast<const __half*>(base), idx);
}

static __device__ __forceinline__ float2 to_float2_pair(__half2 v) { return __half22float2(v); }
static __device__ __forceinline__ float2 to_float2_pair(__nv_bfloat162 v) { return __bfloat1622float2(v); }

template <typename T> struct Vec2;
template <> struct Vec2<__half> { using type = __half2; };
template <> struct Vec2<__nv_bfloat16> { using type = __nv_bfloat162; };

template <typename T> __device__ inline T to_dtype(float x);
template <> __device__ inline __half to_dtype<__half>(float x) { return __float2half_rn(x); }
template <> __device__ inline __nv_bfloat16 to_dtype<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

// -------------------------
// Warp exclusive-scan (no CUB needed)
// -------------------------
template <typename T>
static __device__ __forceinline__ T warp_exclusive_scan_sum(T v, unsigned mask = 0xffffffffu) {
  T res = v;
  #pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    T n = __shfl_up_sync(mask, res, offset);
    if ((threadIdx.x & 31) >= offset) res += n;
  }
  return res - v;
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
// Fused kernel template: GH=4, D=128, BN in {32,64,256}
// Variant E1 for V loading.
// grid=(T_c, KVH, B)
// block=(NW*32) where NW=4 for BN=32/64, NW=8 for BN=256
// =======================================================

template <typename TK, int BN, int NW>
__global__ __launch_bounds__(256, 4)
void fused_tiles_vec_gqa_gh4_d128_bn(
    const TK* __restrict__ Q,   // single: [H, D], batch: [B, H, D]
    const TK* __restrict__ K,   // single: [L_storage, KVH, D], batch: [B, L_storage, KVH, D]
    const TK* __restrict__ V,   // single: [L_storage, KVH, D], batch: [B, L_storage, KVH, D]
    int B, int H, int L_valid, int L_storage, int T_c, float q_scale,
    int S_per_tile, uint64_t seed,
    float* __restrict__ m_tiles,    // single: [H, T_c], batch: [B, H, T_c]
    float* __restrict__ ell_tiles,  // single: [H, T_c], batch: [B, H, T_c]
    float* __restrict__ out_tiles,  // single: [H, T_c, 128], batch: [B, H, T_c, 128]
    int* __restrict__ rowmask       // single: [1, KVH, L_valid], batch: [B, KVH, L_valid]
) {
  constexpr int GH = 4;
  constexpr int D = 128;
  constexpr bool IS_BF16_TK = std::is_same<TK, __nv_bfloat16>::value;

  const int t = (int)blockIdx.x;
  const int kv = (int)blockIdx.y;
  const int b = (int)blockIdx.z;
  const int KVH = (int)gridDim.y;
  if (b >= B) return;

  const int h0 = kv * GH;
  if (h0 + (GH - 1) >= H) return;

  const int n0 = t * BN;
  int n1 = n0 + BN;
  if (n1 > L_valid) n1 = L_valid;
  const int tile_len = n1 - n0;

  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;

  __shared__ float s_scores[BN * GH];
  __shared__ TK u_shmem[BN * GH];
  __shared__ int16_t c_tile[GH * BN];
  __shared__ int emit_idx[BN];
  __shared__ int s_emit_len;

  __shared__ float s_warp_red[NW * GH];
  __shared__ float s_m[GH];
  __shared__ float s_ell[GH];

  if (tile_len <= 0) {
    if (warp_id < GH) {
      const int h = h0 + warp_id;
      const int d0 = lane * 4;
      if (d0 + 3 < D) {
        const size_t base = (((size_t)b * (size_t)H + (size_t)h) * (size_t)T_c + (size_t)t) * (size_t)D + (size_t)d0;
        out_tiles[base + 0] = 0.f;
        out_tiles[base + 1] = 0.f;
        out_tiles[base + 2] = 0.f;
        out_tiles[base + 3] = 0.f;
      }
    }
    if (warp_id == 0 && lane < GH) {
      const int h = h0 + lane;
      m_tiles[((size_t)b * (size_t)H + (size_t)h) * (size_t)T_c + (size_t)t] = -CUDART_INF_F;
      ell_tiles[((size_t)b * (size_t)H + (size_t)h) * (size_t)T_c + (size_t)t] = 0.f;
    }
    return;
  }

  const int d0 = lane * 4;

  float2 q01[GH], q23[GH];
  #pragma unroll
  for (int g = 0; g < GH; ++g) {
    const TK* qptr = Q + (((size_t)b * (size_t)H + (size_t)(h0 + g)) * (size_t)D + (size_t)d0);
    using TK2 = typename Vec2<TK>::type;
    const TK2 qa01 = *reinterpret_cast<const TK2*>(qptr);
    const TK2 qa23 = *reinterpret_cast<const TK2*>(qptr + 2);

    float2 a = to_float2_pair(qa01);
    float2 bb = to_float2_pair(qa23);
    a.x *= q_scale; a.y *= q_scale;
    bb.x *= q_scale; bb.y *= q_scale;
    q01[g] = a;
    q23[g] = bb;
  }

  float warp_max[GH];
  #pragma unroll
  for (int g = 0; g < GH; ++g) warp_max[g] = -CUDART_INF_F;

  for (int r = warp_id; r < tile_len; r += NW) {
    const int n = n0 + r;
    const TK* kptr = K + ((((size_t)b * (size_t)L_storage + (size_t)n) * (size_t)KVH + (size_t)kv) * (size_t)D + (size_t)d0);

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

  if (lane == 0) {
    #pragma unroll
    for (int g = 0; g < GH; ++g) {
      s_warp_red[warp_id * GH + g] = warp_max[g];
    }
  }
  __syncthreads();

  if (warp_id == 0 && lane < GH) {
    float m = -CUDART_INF_F;
    #pragma unroll
    for (int w = 0; w < NW; ++w) {
      m = fmaxf(m, s_warp_red[w * GH + lane]);
    }
    s_m[lane] = m;
    const int h = h0 + lane;
    m_tiles[((size_t)b * (size_t)H + (size_t)h) * (size_t)T_c + (size_t)t] = m;
  }
  __syncthreads();

  float ell_local[GH] = {0.f, 0.f, 0.f, 0.f};
  const int total = tile_len * GH;
  for (int idx = tid; idx < total; idx += (int)blockDim.x) {
    const int r = idx / GH;
    const int g = idx - r * GH;
    const float s = s_scores[r * GH + g];
    const float u = __expf(s - s_m[g]);
    ell_local[g] += u;
    u_shmem[r * GH + g] = to_dtype<TK>(u);
  }

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
    s_ell[lane] = ell;
    const int h = h0 + lane;
    ell_tiles[((size_t)b * (size_t)H + (size_t)h) * (size_t)T_c + (size_t)t] = ell;
  }
  __syncthreads();

  if (warp_id < GH) {
    const int g = warp_id;
    const int h = h0 + g;
    const float ell = s_ell[g];
    const float eps = 1e-20f;
    const float invdel = (S_per_tile > 0 && ell > eps) ? ((float)S_per_tile / ell) : 0.0f;
    const float a0 = u01_from_seed(seed, b * H + h, t);

    float carry = 0.f;
    for (int base = 0; base < tile_len; base += 32) {
      const int r = base + lane;
      float x = 0.f;
      if (r < tile_len && invdel > 0.f) {
        const float u = load16_as_f32<TK, IS_BF16_TK>(u_shmem, r * GH + g);
        x = u * invdel;
      }

      float pre = warp_exclusive_scan_sum<float>(x);
      const int remain = tile_len - base;
      const int last_lane = (remain > 0) ? min(31, remain - 1) : 0;

      float pre_g = carry + pre;
      float cum_g = pre_g + x;
      if (r < tile_len) {
        float f_prev = floorf(a0 + pre_g);
        float f_curr = floorf(a0 + cum_g);
        int c = (int)(f_curr - f_prev);
        c = (c < 0 ? 0 : c);
        c = (c > 32767 ? 32767 : c);
        c_tile[(size_t)g * (size_t)BN + (size_t)r] = (int16_t)c;
      }

      float chunk_total = __shfl_sync(0xffffffffu, pre + x, last_lane);
      if (lane == 0) carry += chunk_total;
      carry = __shfl_sync(0xffffffffu, carry, 0);
    }
  }
  __syncthreads();

  if (warp_id == 0) {
    int emit_carry = 0;
    for (int base = 0; base < tile_len; base += 32) {
      const int r = base + lane;
      int flag = 0;
      if (r < tile_len) {
        int sum = 0;
        #pragma unroll
        for (int gg = 0; gg < GH; ++gg) {
          sum += (int)c_tile[(size_t)gg * (size_t)BN + (size_t)r];
        }
        flag = (sum > 0);
      }

      int pos_local = warp_exclusive_scan_sum<int>(flag);
      const int remain = tile_len - base;
      const int last_lane = (remain > 0) ? min(31, remain - 1) : 0;
      int chunk_sum = __shfl_sync(0xffffffffu, pos_local + flag, last_lane);

      if (r < tile_len && flag) {
        const int n = n0 + r;
        const int pos = emit_carry + pos_local;
        emit_idx[pos] = n;
        if (rowmask) rowmask[((size_t)b * (size_t)KVH + (size_t)kv) * (size_t)L_valid + (size_t)n] = 1;
      }

      if (lane == 0) emit_carry += chunk_sum;
      emit_carry = __shfl_sync(0xffffffffu, emit_carry, 0);
    }
    if (lane == 0) s_emit_len = emit_carry;
  }
  __syncthreads();

  if (warp_id < GH) {
    const int g = warp_id;
    const int h = h0 + g;
    (void)h;

    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    const int emit_len = s_emit_len;
    for (int i = 0; i < emit_len; ++i) {
      const int n = emit_idx[i];
      const int r_local = n - n0;
      const int c = (int)c_tile[(size_t)g * (size_t)BN + (size_t)r_local];
      if (c != 0) {
        const float cf = (float)c;
        const TK* vptr = V + ((((size_t)b * (size_t)L_storage + (size_t)n) * (size_t)KVH + (size_t)kv) * (size_t)D + (size_t)d0);
        using TK2 = typename Vec2<TK>::type;
        const TK2 v01 = *reinterpret_cast<const TK2*>(vptr);
        const TK2 v23 = *reinterpret_cast<const TK2*>(vptr + 2);
        const float2 v01f = to_float2_pair(v01);
        const float2 v23f = to_float2_pair(v23);
        acc0 = fmaf(cf, v01f.x, acc0);
        acc1 = fmaf(cf, v01f.y, acc1);
        acc2 = fmaf(cf, v23f.x, acc2);
        acc3 = fmaf(cf, v23f.y, acc3);
      }
    }

    const int h_out = h0 + g;
    const size_t base = ((((size_t)b * (size_t)H + (size_t)h_out) * (size_t)T_c) + (size_t)t) * (size_t)D + (size_t)d0;
    out_tiles[base + 0] = acc0;
    out_tiles[base + 1] = acc1;
    out_tiles[base + 2] = acc2;
    out_tiles[base + 3] = acc3;
  }
}

// =======================================================
// Finalize: Flash-like merge across tiles
// =======================================================

template <typename OUT_T>
__global__ void finalize_weighted_out_kernel(
    const float* __restrict__ m_tiles,     // single: [H,T_c], batch: [B,H,T_c]
    const float* __restrict__ ell_tiles,   // single: [H,T_c], batch: [B,H,T_c]
    const float* __restrict__ out_tiles,   // single: [H,T_c,D], batch: [B,H,T_c,D]
    int B, int H, int T_c, int D,
    int S_per_tile,
    OUT_T* __restrict__ out,               // single: [H,D], batch: [B,H,D]
    float* __restrict__ m_star_out)        // single: [H], batch: [B,H]
{
  const int h = (int)blockIdx.x;
  const int b = (int)blockIdx.y;
  if (h >= H || b >= B) return;

  const int tid = (int)threadIdx.x;
  extern __shared__ unsigned char smem[];
  float* w_shared = reinterpret_cast<float*>(smem);
  float* red = w_shared + T_c;

  float local_max = -CUDART_INF_F;
  for (int t = tid; t < T_c; t += (int)blockDim.x) {
    local_max = fmaxf(local_max, m_tiles[((size_t)b * (size_t)H + (size_t)h) * (size_t)T_c + (size_t)t]);
  }
  red[tid] = local_max;
  __syncthreads();

  for (int s = ((int)blockDim.x) >> 1; s > 32; s >>= 1) {
    if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
    __syncthreads();
  }
  if (tid < 32) {
    volatile float* v = red;
    v[tid] = fmaxf(v[tid], v[tid + 32]);
    v[tid] = fmaxf(v[tid], v[tid + 16]);
    v[tid] = fmaxf(v[tid], v[tid + 8]);
    v[tid] = fmaxf(v[tid], v[tid + 4]);
    v[tid] = fmaxf(v[tid], v[tid + 2]);
    v[tid] = fmaxf(v[tid], v[tid + 1]);
  }
  __syncthreads();
  const float m_star = red[0];
  if (tid == 0 && m_star_out) {
    m_star_out[(size_t)b * (size_t)H + (size_t)h] = m_star;
  }
  __syncthreads();

  float z_local = 0.f;
  for (int t = tid; t < T_c; t += (int)blockDim.x) {
    const float mt = m_tiles[((size_t)b * (size_t)H + (size_t)h) * (size_t)T_c + (size_t)t];
    const float ell = ell_tiles[((size_t)b * (size_t)H + (size_t)h) * (size_t)T_c + (size_t)t];
    const float w = __expf(mt - m_star) * ell;
    w_shared[t] = w;
    z_local += w;
  }
  red[tid] = z_local;
  __syncthreads();

  for (int s = ((int)blockDim.x) >> 1; s > 32; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  if (tid < 32) {
    volatile float* v = red;
    v[tid] += v[tid + 32];
    v[tid] += v[tid + 16];
    v[tid] += v[tid + 8];
    v[tid] += v[tid + 4];
    v[tid] += v[tid + 2];
    v[tid] += v[tid + 1];
  }
  __syncthreads();
  const float Z = red[0];
  const float inv_norm = (S_per_tile > 0 && (Z > 0.0f)) ? (1.0f / (Z * (float)S_per_tile)) : 0.0f;

  for (int d = tid; d < D; d += (int)blockDim.x) {
    float num = 0.f;
    for (int t = 0; t < T_c; ++t) {
      const float w = w_shared[t];
      const size_t idx = ((((size_t)b * (size_t)H + (size_t)h) * (size_t)T_c) + (size_t)t) * (size_t)D + (size_t)d;
      num += w * out_tiles[idx];
    }
    const float y = num * inv_norm;
    out[((size_t)b * (size_t)H + (size_t)h) * (size_t)D + (size_t)d] = to_dtype<OUT_T>(y);
  }
}

// =======================================================
// vrows reduction (optional metric)
// =======================================================
__global__ void reduce_vrows_kernel(const int* __restrict__ rowmask,
                                    int B, int KVH, int L,
                                    int* __restrict__ vrows) {
  extern __shared__ unsigned char smem[];
  int* sdata = reinterpret_cast<int*>(smem);
  const int kv = (int)blockIdx.x;
  const int b = (int)blockIdx.y;
  if (kv >= KVH || b >= B) return;
  const int tid = (int)threadIdx.x;
  const int TPB = (int)blockDim.x;

  int local_sum = 0;
  for (int n = tid; n < L; n += TPB) {
    local_sum += (rowmask[((size_t)b * (size_t)KVH + (size_t)kv) * (size_t)L + (size_t)n] != 0);
  }
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
    v[tid] += v[tid + 8];
    v[tid] += v[tid + 4];
    v[tid] += v[tid + 2];
    v[tid] += v[tid + 1];
  }
  __syncthreads();
  if (tid == 0) vrows[(size_t)b * (size_t)KVH + (size_t)kv] = sdata[0];
}

// =======================================================
// Launchers
// =======================================================

extern "C" void launch_fused_tiles_vec_gqa_gh4_d128_bn32_bf16(
    const void* qraw, const void* Kraw, const void* Vraw,
    int B, int H, int KVH, int L_valid, int L_storage, int T_c, float q_scale,
    int S_per_tile, uint64_t seed,
    float* m_tiles, float* ell_tiles, float* out_tiles,
    int* rowmask, cudaStream_t stream) {
  dim3 grid((unsigned)T_c, (unsigned)KVH, (unsigned)B);
  dim3 block(128, 1, 1);
  fused_tiles_vec_gqa_gh4_d128_bn<__nv_bfloat16, 32, 4>
      <<<grid, block, 0, stream>>>(
          (const __nv_bfloat16*)qraw,
          (const __nv_bfloat16*)Kraw,
          (const __nv_bfloat16*)Vraw,
          B, H, L_valid, L_storage, T_c, q_scale,
          S_per_tile, seed,
          m_tiles, ell_tiles, out_tiles,
          rowmask);
}

extern "C" void launch_fused_tiles_vec_gqa_gh4_d128_bn64_bf16(
    const void* qraw, const void* Kraw, const void* Vraw,
    int B, int H, int KVH, int L_valid, int L_storage, int T_c, float q_scale,
    int S_per_tile, uint64_t seed,
    float* m_tiles, float* ell_tiles, float* out_tiles,
    int* rowmask, cudaStream_t stream) {
  dim3 grid((unsigned)T_c, (unsigned)KVH, (unsigned)B);
  dim3 block(128, 1, 1);
  fused_tiles_vec_gqa_gh4_d128_bn<__nv_bfloat16, 64, 4>
      <<<grid, block, 0, stream>>>(
          (const __nv_bfloat16*)qraw,
          (const __nv_bfloat16*)Kraw,
          (const __nv_bfloat16*)Vraw,
          B, H, L_valid, L_storage, T_c, q_scale,
          S_per_tile, seed,
          m_tiles, ell_tiles, out_tiles,
          rowmask);
}

extern "C" void launch_fused_tiles_vec_gqa_gh4_d128_bn256_bf16(
    const void* qraw, const void* Kraw, const void* Vraw,
    int B, int H, int KVH, int L_valid, int L_storage, int T_c, float q_scale,
    int S_per_tile, uint64_t seed,
    float* m_tiles, float* ell_tiles, float* out_tiles,
    int* rowmask, cudaStream_t stream) {
  dim3 grid((unsigned)T_c, (unsigned)KVH, (unsigned)B);
  dim3 block(256, 1, 1);
  fused_tiles_vec_gqa_gh4_d128_bn<__nv_bfloat16, 256, 8>
      <<<grid, block, 0, stream>>>(
          (const __nv_bfloat16*)qraw,
          (const __nv_bfloat16*)Kraw,
          (const __nv_bfloat16*)Vraw,
          B, H, L_valid, L_storage, T_c, q_scale,
          S_per_tile, seed,
          m_tiles, ell_tiles, out_tiles,
          rowmask);
}

extern "C" void launch_fused_tiles_vec_gqa_gh4_d128_bn32_f16(
    const void* qraw, const void* Kraw, const void* Vraw,
    int B, int H, int KVH, int L_valid, int L_storage, int T_c, float q_scale,
    int S_per_tile, uint64_t seed,
    float* m_tiles, float* ell_tiles, float* out_tiles,
    int* rowmask, cudaStream_t stream) {
  dim3 grid((unsigned)T_c, (unsigned)KVH, (unsigned)B);
  dim3 block(128, 1, 1);
  fused_tiles_vec_gqa_gh4_d128_bn<__half, 32, 4>
      <<<grid, block, 0, stream>>>(
          (const __half*)qraw,
          (const __half*)Kraw,
          (const __half*)Vraw,
          B, H, L_valid, L_storage, T_c, q_scale,
          S_per_tile, seed,
          m_tiles, ell_tiles, out_tiles,
          rowmask);
}

extern "C" void launch_fused_tiles_vec_gqa_gh4_d128_bn64_f16(
    const void* qraw, const void* Kraw, const void* Vraw,
    int B, int H, int KVH, int L_valid, int L_storage, int T_c, float q_scale,
    int S_per_tile, uint64_t seed,
    float* m_tiles, float* ell_tiles, float* out_tiles,
    int* rowmask, cudaStream_t stream) {
  dim3 grid((unsigned)T_c, (unsigned)KVH, (unsigned)B);
  dim3 block(128, 1, 1);
  fused_tiles_vec_gqa_gh4_d128_bn<__half, 64, 4>
      <<<grid, block, 0, stream>>>(
          (const __half*)qraw,
          (const __half*)Kraw,
          (const __half*)Vraw,
          B, H, L_valid, L_storage, T_c, q_scale,
          S_per_tile, seed,
          m_tiles, ell_tiles, out_tiles,
          rowmask);
}

extern "C" void launch_fused_tiles_vec_gqa_gh4_d128_bn256_f16(
    const void* qraw, const void* Kraw, const void* Vraw,
    int B, int H, int KVH, int L_valid, int L_storage, int T_c, float q_scale,
    int S_per_tile, uint64_t seed,
    float* m_tiles, float* ell_tiles, float* out_tiles,
    int* rowmask, cudaStream_t stream) {
  dim3 grid((unsigned)T_c, (unsigned)KVH, (unsigned)B);
  dim3 block(256, 1, 1);
  fused_tiles_vec_gqa_gh4_d128_bn<__half, 256, 8>
      <<<grid, block, 0, stream>>>(
          (const __half*)qraw,
          (const __half*)Kraw,
          (const __half*)Vraw,
          B, H, L_valid, L_storage, T_c, q_scale,
          S_per_tile, seed,
          m_tiles, ell_tiles, out_tiles,
          rowmask);
}

extern "C" void launch_finalize_weighted_bf16(
    const float* m_tiles,
    const float* ell_tiles,
    const float* out_tiles,
    int B, int H, int T_c, int D,
    int S_per_tile,
    void* out,
    float* m_star,
    cudaStream_t stream) {
  const int TPB = 256;
  dim3 block(TPB, 1, 1);
  dim3 grid((unsigned)H, (unsigned)B, 1);
  size_t shmem = (size_t)(T_c + TPB) * sizeof(float);
  finalize_weighted_out_kernel<__nv_bfloat16>
      <<<grid, block, shmem, stream>>>(
          m_tiles, ell_tiles, out_tiles,
          B, H, T_c, D,
          S_per_tile,
          (__nv_bfloat16*)out,
          m_star);
}

extern "C" void launch_finalize_weighted_f16(
    const float* m_tiles,
    const float* ell_tiles,
    const float* out_tiles,
    int B, int H, int T_c, int D,
    int S_per_tile,
    void* out,
    float* m_star,
    cudaStream_t stream) {
  const int TPB = 256;
  dim3 block(TPB, 1, 1);
  dim3 grid((unsigned)H, (unsigned)B, 1);
  size_t shmem = (size_t)(T_c + TPB) * sizeof(float);
  finalize_weighted_out_kernel<__half>
      <<<grid, block, shmem, stream>>>(
          m_tiles, ell_tiles, out_tiles,
          B, H, T_c, D,
          S_per_tile,
          (__half*)out,
          m_star);
}

extern "C" void launch_reduce_vrows(const int* rowmask,
                                     int B, int KVH, int L,
                                     int* vrows,
                                     cudaStream_t stream) {
  const int TPB = 256;
  dim3 block(TPB, 1, 1);
  dim3 grid((unsigned)KVH, (unsigned)B, 1);
  size_t shmem = TPB * sizeof(int);
  reduce_vrows_kernel<<<grid, block, shmem, stream>>>(rowmask, B, KVH, L, vrows);
                                     }
