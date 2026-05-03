// santa_cuda.cpp — PyTorch glue for fused S^2ANTA-flash kernels.
//
// Batched contiguous-KV update:
//   - Preserves the existing fused tile / finalize logic.
//   - Extends the decode path from batch size 1 to a uniform-length batched path.
//   - Supports full-cache tensors with a separate valid_len, so Python can keep
//     a single contiguous cache [B, T_storage, KVH, D] and avoid prefix compaction.
//
// Supported shapes:
//   - D  == 128
//   - GH == H / KVH == 4
//   - Q layout:
//       * single: [H, D]
//       * batch : [B, H, D]
//   - K/V layout:
//       * single: [L_storage, KVH, D]
//       * batch : [B, L_storage, KVH, D]
//
// Python API:
//   decode_systematic_scalar(q_h, K_g, V_g, S, seed=42, block_n=0, want_vrows=false, valid_len=-1)
//   decode_systematic_batched(...)   # alias of the same function

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#include <cmath>
#include <cstdint>
#include <tuple>

#include <cuda_runtime.h>

namespace py = pybind11;

// ---- block_n heuristic (preserved) ----
static inline int64_t _auto_block_n_from_L(int L) {
  if (L <= 1536) return 32;
  if (L <= 3072) return 64;
  return 256;
}
static inline int64_t _normalize_block_n(int L, int64_t block_n) {
  if (block_n > 0) return block_n;
  return _auto_block_n_from_L(L);
}

struct ShapeInfo {
  bool batched;
  int B;
  int H;
  int D;
  int L_storage;
  int KVH;
};

// ==== CUDA launchers (implemented in santa_cuda_kernel.cu) ====
extern "C" void launch_fused_tiles_vec_gqa_gh4_d128_bn32_bf16(
    const void* qraw, const void* Kraw, const void* Vraw,
    int B, int H, int KVH, int L_valid, int L_storage, int T_c, float q_scale,
    int S_per_tile, uint64_t seed,
    float* m_tiles, float* ell_tiles, float* out_tiles,
    int* rowmask, cudaStream_t stream);

extern "C" void launch_fused_tiles_vec_gqa_gh4_d128_bn64_bf16(
    const void* qraw, const void* Kraw, const void* Vraw,
    int B, int H, int KVH, int L_valid, int L_storage, int T_c, float q_scale,
    int S_per_tile, uint64_t seed,
    float* m_tiles, float* ell_tiles, float* out_tiles,
    int* rowmask, cudaStream_t stream);

extern "C" void launch_fused_tiles_vec_gqa_gh4_d128_bn256_bf16(
    const void* qraw, const void* Kraw, const void* Vraw,
    int B, int H, int KVH, int L_valid, int L_storage, int T_c, float q_scale,
    int S_per_tile, uint64_t seed,
    float* m_tiles, float* ell_tiles, float* out_tiles,
    int* rowmask, cudaStream_t stream);

extern "C" void launch_fused_tiles_vec_gqa_gh4_d128_bn32_f16(
    const void* qraw, const void* Kraw, const void* Vraw,
    int B, int H, int KVH, int L_valid, int L_storage, int T_c, float q_scale,
    int S_per_tile, uint64_t seed,
    float* m_tiles, float* ell_tiles, float* out_tiles,
    int* rowmask, cudaStream_t stream);

extern "C" void launch_fused_tiles_vec_gqa_gh4_d128_bn64_f16(
    const void* qraw, const void* Kraw, const void* Vraw,
    int B, int H, int KVH, int L_valid, int L_storage, int T_c, float q_scale,
    int S_per_tile, uint64_t seed,
    float* m_tiles, float* ell_tiles, float* out_tiles,
    int* rowmask, cudaStream_t stream);

extern "C" void launch_fused_tiles_vec_gqa_gh4_d128_bn256_f16(
    const void* qraw, const void* Kraw, const void* Vraw,
    int B, int H, int KVH, int L_valid, int L_storage, int T_c, float q_scale,
    int S_per_tile, uint64_t seed,
    float* m_tiles, float* ell_tiles, float* out_tiles,
    int* rowmask, cudaStream_t stream);

extern "C" void launch_finalize_weighted_bf16(
    const float* m_tiles,
    const float* ell_tiles,
    const float* out_tiles,
    int B, int H, int T_c, int D,
    int S_per_tile,
    void* out,
    float* m_star,
    cudaStream_t stream);

extern "C" void launch_finalize_weighted_f16(
    const float* m_tiles,
    const float* ell_tiles,
    const float* out_tiles,
    int B, int H, int T_c, int D,
    int S_per_tile,
    void* out,
    float* m_star,
    cudaStream_t stream);

extern "C" void launch_reduce_vrows(
    const int* rowmask, int B, int KVH, int L,
    int* vrows, cudaStream_t stream);


static ShapeInfo _check_inputs_q_kv(const at::Tensor& q_h,
                                    const at::Tensor& K_g,
                                    const at::Tensor& V_g) {
  TORCH_CHECK(q_h.is_cuda(), "q_h must be CUDA");
  TORCH_CHECK(K_g.is_cuda() && V_g.is_cuda(), "K_g and V_g must be CUDA");
  TORCH_CHECK((q_h.scalar_type() == at::kBFloat16 || q_h.scalar_type() == at::kHalf),
              "q_h must be bf16 or f16");
  TORCH_CHECK((K_g.scalar_type() == at::kBFloat16 || K_g.scalar_type() == at::kHalf),
              "K_g must be bf16 or f16");
  TORCH_CHECK(K_g.scalar_type() == V_g.scalar_type(), "K_g and V_g must have the same dtype");
  TORCH_CHECK(q_h.scalar_type() == K_g.scalar_type(), "q_h dtype must match K_g/V_g dtype");
  TORCH_CHECK(q_h.device() == K_g.device() && q_h.device() == V_g.device(), "device mismatch");

  TORCH_CHECK(K_g.is_contiguous() && V_g.is_contiguous(),
              "K_g/V_g must be contiguous. Expected [L,KVH,D] for single or [B,L,KVH,D] for batch.");

  ShapeInfo info{};
  if (q_h.dim() == 2) {
    TORCH_CHECK(K_g.dim() == 3 && V_g.dim() == 3,
                "Single-request path expects q_h [H,D], K_g/V_g [L,KVH,D]");
    TORCH_CHECK(K_g.size(0) == V_g.size(0) && K_g.size(1) == V_g.size(1) && K_g.size(2) == V_g.size(2),
                "K_g and V_g must have the same shape");
    TORCH_CHECK(q_h.size(1) == K_g.size(2), "D mismatch (q_h.size(1) vs K_g.size(2))");

    info.batched = false;
    info.B = 1;
    info.H = (int)q_h.size(0);
    info.D = (int)q_h.size(1);
    info.L_storage = (int)K_g.size(0);
    info.KVH = (int)K_g.size(1);
    return info;
  }

  TORCH_CHECK(q_h.dim() == 3 && K_g.dim() == 4 && V_g.dim() == 4,
              "Batched path expects q_h [B,H,D], K_g/V_g [B,L,KVH,D]");
  TORCH_CHECK(K_g.size(0) == V_g.size(0) && K_g.size(1) == V_g.size(1) &&
              K_g.size(2) == V_g.size(2) && K_g.size(3) == V_g.size(3),
              "K_g and V_g must have the same shape");
  TORCH_CHECK(q_h.size(0) == K_g.size(0), "Batch mismatch between q_h and K_g/V_g");
  TORCH_CHECK(q_h.size(2) == K_g.size(3), "D mismatch (q_h.size(2) vs K_g.size(3))");

  info.batched = true;
  info.B = (int)q_h.size(0);
  info.H = (int)q_h.size(1);
  info.D = (int)q_h.size(2);
  info.L_storage = (int)K_g.size(1);
  info.KVH = (int)K_g.size(2);
  return info;
}


std::tuple<at::Tensor, at::Tensor, at::Tensor>
decode_systematic_scalar(at::Tensor q_h, at::Tensor K_g, at::Tensor V_g,
                         int64_t S, uint64_t seed, int64_t block_n,
                         bool want_vrows, int64_t valid_len) {
  ShapeInfo info = _check_inputs_q_kv(q_h, K_g, V_g);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto q = q_h.contiguous();

  const int B = info.B;
  const int H = info.H;
  const int D = info.D;
  const int KVH = info.KVH;
  const int GH = H / KVH;
  const int L_storage = info.L_storage;
  const int L = (valid_len > 0) ? (int)valid_len : L_storage;

  TORCH_CHECK(L > 0, "valid_len must be > 0; got ", L);
  TORCH_CHECK(L <= L_storage, "valid_len exceeds storage length: valid_len=", L, " storage=", L_storage);
  TORCH_CHECK(D == 128, "This build supports only D=128. Got D=", D);
  TORCH_CHECK(H % KVH == 0, "H must be divisible by KVH. Got H=", H, " KVH=", KVH);
  TORCH_CHECK(GH == 4, "This build supports only GH=4 (H/KVH). Got GH=", GH,
                        " (H=", H, " KVH=", KVH, ")");

  block_n = _normalize_block_n(L, block_n);
  TORCH_CHECK(block_n == 32 || block_n == 64 || block_n == 256,
              "block_n must be 0 (auto) or one of {32,64,256}. Got block_n=", block_n);

  const int bn = (int)block_n;
  const int T_c = (L + bn - 1) / bn;

  int S_per_tile = 0;
  if (S > 0) {
    S_per_tile = (int)(S / T_c);
    if (S_per_tile < 1) S_per_tile = 1;
  }

  const float q_scale = 1.0f / std::sqrt((float)D);
  auto optsF = q.options().dtype(at::kFloat);
  auto optsI = q.options().dtype(at::kInt);

  at::Tensor m_tiles;
  at::Tensor ell_tiles;
  at::Tensor out_tiles;
  at::Tensor rowmask;

  if (info.batched) {
    m_tiles = at::empty({B, H, T_c}, optsF);
    ell_tiles = at::empty({B, H, T_c}, optsF);
    out_tiles = at::empty({B, H, T_c, D}, optsF);
    if (want_vrows) rowmask = at::zeros({B, KVH, L}, optsI);
  } else {
    m_tiles = at::empty({H, T_c}, optsF);
    ell_tiles = at::empty({H, T_c}, optsF);
    out_tiles = at::empty({H, T_c, D}, optsF);
    if (want_vrows) rowmask = at::zeros({1, KVH, L}, optsI);
  }

  const void* Qraw = (const void*)q.data_ptr();
  const void* Kraw = (const void*)K_g.data_ptr();
  const void* Vraw = (const void*)V_g.data_ptr();

  if (q.scalar_type() == at::kBFloat16) {
    if (bn == 32) {
      launch_fused_tiles_vec_gqa_gh4_d128_bn32_bf16(
          Qraw, Kraw, Vraw, B, H, KVH, L, L_storage, T_c, q_scale,
          S_per_tile, seed,
          m_tiles.data_ptr<float>(),
          ell_tiles.data_ptr<float>(),
          out_tiles.data_ptr<float>(),
          want_vrows ? rowmask.data_ptr<int>() : nullptr,
          stream.stream());
    } else if (bn == 64) {
      launch_fused_tiles_vec_gqa_gh4_d128_bn64_bf16(
          Qraw, Kraw, Vraw, B, H, KVH, L, L_storage, T_c, q_scale,
          S_per_tile, seed,
          m_tiles.data_ptr<float>(),
          ell_tiles.data_ptr<float>(),
          out_tiles.data_ptr<float>(),
          want_vrows ? rowmask.data_ptr<int>() : nullptr,
          stream.stream());
    } else {
      launch_fused_tiles_vec_gqa_gh4_d128_bn256_bf16(
          Qraw, Kraw, Vraw, B, H, KVH, L, L_storage, T_c, q_scale,
          S_per_tile, seed,
          m_tiles.data_ptr<float>(),
          ell_tiles.data_ptr<float>(),
          out_tiles.data_ptr<float>(),
          want_vrows ? rowmask.data_ptr<int>() : nullptr,
          stream.stream());
    }
  } else {
    if (bn == 32) {
      launch_fused_tiles_vec_gqa_gh4_d128_bn32_f16(
          Qraw, Kraw, Vraw, B, H, KVH, L, L_storage, T_c, q_scale,
          S_per_tile, seed,
          m_tiles.data_ptr<float>(),
          ell_tiles.data_ptr<float>(),
          out_tiles.data_ptr<float>(),
          want_vrows ? rowmask.data_ptr<int>() : nullptr,
          stream.stream());
    } else if (bn == 64) {
      launch_fused_tiles_vec_gqa_gh4_d128_bn64_f16(
          Qraw, Kraw, Vraw, B, H, KVH, L, L_storage, T_c, q_scale,
          S_per_tile, seed,
          m_tiles.data_ptr<float>(),
          ell_tiles.data_ptr<float>(),
          out_tiles.data_ptr<float>(),
          want_vrows ? rowmask.data_ptr<int>() : nullptr,
          stream.stream());
    } else {
      launch_fused_tiles_vec_gqa_gh4_d128_bn256_f16(
          Qraw, Kraw, Vraw, B, H, KVH, L, L_storage, T_c, q_scale,
          S_per_tile, seed,
          m_tiles.data_ptr<float>(),
          ell_tiles.data_ptr<float>(),
          out_tiles.data_ptr<float>(),
          want_vrows ? rowmask.data_ptr<int>() : nullptr,
          stream.stream());
    }
  }
  AT_CUDA_CHECK(cudaGetLastError());

  at::Tensor out;
  at::Tensor m_star;
  if (info.batched) {
    out = at::empty({B, H, D}, q.options());
    m_star = at::empty({B, H}, optsF);
  } else {
    out = at::empty({H, D}, q.options());
    m_star = at::empty({H}, optsF);
  }

  if (q.scalar_type() == at::kBFloat16) {
    launch_finalize_weighted_bf16(
        m_tiles.data_ptr<float>(),
        ell_tiles.data_ptr<float>(),
        out_tiles.data_ptr<float>(),
        B, H, T_c, D,
        S_per_tile,
        (void*)out.data_ptr(),
        m_star.data_ptr<float>(),
        stream.stream());
  } else {
    launch_finalize_weighted_f16(
        m_tiles.data_ptr<float>(),
        ell_tiles.data_ptr<float>(),
        out_tiles.data_ptr<float>(),
        B, H, T_c, D,
        S_per_tile,
        (void*)out.data_ptr(),
        m_star.data_ptr<float>(),
        stream.stream());
  }
  AT_CUDA_CHECK(cudaGetLastError());

  at::Tensor vrows = at::empty({0}, optsI);
  if (want_vrows) {
    if (info.batched) {
      vrows = at::empty({B, KVH}, optsI);
      launch_reduce_vrows(rowmask.data_ptr<int>(), B, KVH, L, vrows.data_ptr<int>(), stream.stream());
    } else {
      at::Tensor vrows_b = at::empty({1, KVH}, optsI);
      launch_reduce_vrows(rowmask.data_ptr<int>(), 1, KVH, L, vrows_b.data_ptr<int>(), stream.stream());
      vrows = vrows_b.squeeze(0);
    }
    AT_CUDA_CHECK(cudaGetLastError());
  }

  return std::make_tuple(out, m_star, vrows);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode_systematic_scalar", &decode_systematic_scalar,
        py::arg("q_h"), py::arg("K_g"), py::arg("V_g"), py::arg("S"),
        py::arg("seed") = 42ULL,
        py::arg("block_n") = 0,
        py::arg("want_vrows") = false,
        py::arg("valid_len") = -1);

  m.def("decode_systematic_batched", &decode_systematic_scalar,
        py::arg("q_h"), py::arg("K_g"), py::arg("V_g"), py::arg("S"),
        py::arg("seed") = 42ULL,
        py::arg("block_n") = 0,
        py::arg("want_vrows") = false,
        py::arg("valid_len") = -1);
}
