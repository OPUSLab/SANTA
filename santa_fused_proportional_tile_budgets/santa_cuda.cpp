// santa_cuda.cpp — minimal C++/PyTorch glue for the slim CUDA kernels.
//
// Key choices:
//   * Pass-1 supports only GQA-aware implementations:
//       - Vec specialized (GH=4, D=128, block_n=256)
//       - WMMA fallback (D % 16 == 0 and block_n % 16 == 0)
//   * Pass-1 selection via env SANTA_PASS1={auto,gqa,vec}.
//       - auto: vec when eligible, else WMMA
//       - gqa : force WMMA GQA
//       - vec : force vec (errors if not eligible)
//   * u_stash layout is [L, H] (index = n*H + h) to match pass-2 reads.
//   * block_n heuristic:
//       - if block_n <= 0 (or omitted from Python), choose based on L.
//       - additionally, if SANTA_PASS1=vec and block_n<=0, we force block_n=256,
//         because the vec kernel is only correct for BN=256.
//
// Build: via PyTorch CUDAExtension (same flags you already use).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#include <tuple>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <cctype>

// --- CUDA version presence ---
#if !defined(CUDA_VERSION)
#error "CUDA headers not found. Build with CUDAExtension (nvcc)."
#endif

// pass1 mode:
//   0 = auto (vec when eligible else WMMA-GQA)
//   1 = gqa  (force WMMA-GQA)
//   2 = vec  (force vec kernel)
static inline int _pass1_mode_from_env() {
  const char* env = std::getenv("SANTA_PASS1");
  if (!env) return 0; // auto
  std::string s(env);
  for (auto& c : s) c = (char)std::tolower((unsigned char)c);

  if (s == "gqa") return 1;
  // Back-compat alias: "perhead" used to exist; now treat it as "gqa".
  if (s == "perhead") return 1;

  if (s == "vec" || s == "gqa_vec") return 2;
  return 0; // auto
}

// Heuristic for auto block_n selection (block_n <= 0).
// L is the total KV length (K_g.size(0)) in NHD layout.
//
// Based on your sweep (S=1024) on RTX 6000 Ada for H=32, KVH=8, D=128.
// Roughly:
//   - small L: 32
//   - mid   L: 64
//   - large L: 256
static inline int64_t _auto_block_n_from_L(int L) {
  if (L <= 1536) return 32;
  if (L <= 3072) return 64;
  return 256;
}

// Normalize block_n so we never divide by 0 anywhere.
// Also respects SANTA_PASS1=vec when block_n<=0 (must use 256).
static inline int64_t _normalize_block_n(int L, int64_t block_n) {
  if (block_n > 0) return block_n;

  // If user forces vec pass1, vec kernel is fixed BN=256.
  const int mode = _pass1_mode_from_env();
  if (mode == 2) return 256;

  return _auto_block_n_from_L(L);
}

// ==== CUDA launchers (implemented in santa_cuda_kernel.cu) ====
// Pass-1 (GQA-aware grid=(T_c, KVH))
extern "C" void launch_pass1_tiles_wmma_gqa_bf16(const void* qraw, const void* Kraw,
                                                 int H, int KVH, int L, int D,
                                                 int block_n, int T_c, float scale,
                                                 float* m_tiles, float* ell_tiles,
                                                 void* u_stash, cudaStream_t stream);
extern "C" void launch_pass1_tiles_wmma_gqa_f16 (const void* qraw, const void* Kraw,
                                                 int H, int KVH, int L, int D,
                                                 int block_n, int T_c, float scale,
                                                 float* m_tiles, float* ell_tiles,
                                                 void* u_stash, cudaStream_t stream);

extern "C" void launch_pass1_tiles_vec_gqa_bf16(const void* qraw, const void* Kraw,
                                                int H, int KVH, int L, int D,
                                                int block_n, int T_c, float scale,
                                                float* m_tiles, float* ell_tiles,
                                                void* u_stash, cudaStream_t stream);
extern "C" void launch_pass1_tiles_vec_gqa_f16 (const void* qraw, const void* Kraw,
                                                int H, int KVH, int L, int D,
                                                int block_n, int T_c, float scale,
                                                float* m_tiles, float* ell_tiles,
                                                void* u_stash, cudaStream_t stream);

// m* reduction
extern "C" void launch_reduce_mstar(const float* m_tiles, int H, int T_c,
                                    float* m_star, cudaStream_t stream);

// Device tile budgets (largest remainder)
extern "C" void launch_tile_budgets(
    const float* m_tiles, const float* ell_tiles, const float* m_star,
    int H, int T_c, int S, uint64_t seed,
    int* S_tiles, float* invdelta_tile, float* a0_tile,
    cudaStream_t stream);

// Pass-2 (tile-local systematic; grouped; fused atomics for (h,d))
extern "C" void launch_pass2_grouped_fused_atomic_bf16(
    const void* Vraw, const void* Uraw,
    int H, int KVH, int L, int D,
    int block_n, int T_c,
    const float* invdelta_tile, const float* a0_tile, const int* S_tiles,
    float* out, int* rowmask, cudaStream_t stream);
extern "C" void launch_pass2_grouped_fused_atomic_f16(
    const void* Vraw, const void* Uraw,
    int H, int KVH, int L, int D,
    int block_n, int T_c,
    const float* invdelta_tile, const float* a0_tile, const int* S_tiles,
    float* out, int* rowmask, cudaStream_t stream);

// Cast+scale (fp32 -> bf16/f16) for final output
extern "C" void launch_cast_scale_out_bf16(const float* in, void* out,
                                           int H, int D, float scale,
                                           cudaStream_t stream);
extern "C" void launch_cast_scale_out_f16 (const float* in, void* out,
                                           int H, int D, float scale,
                                           cudaStream_t stream);

// vrows reduction (optional metric)
extern "C" void launch_reduce_vrows(const int* rowmask, int KVH, int L,
                                    int* vrows, cudaStream_t stream);

// ==== basic checks ====
static void _check_inputs_q_kv(const at::Tensor& q_h,
                               const at::Tensor& K_g,
                               const at::Tensor& V_g) {
  TORCH_CHECK(q_h.is_cuda(), "q_h must be CUDA");
  TORCH_CHECK(K_g.is_cuda() && V_g.is_cuda(), "K_g and V_g must be CUDA");

  TORCH_CHECK((q_h.scalar_type() == at::kBFloat16 || q_h.scalar_type() == at::kHalf),
              "q_h must be bf16 or f16");
  TORCH_CHECK((K_g.scalar_type() == at::kBFloat16 || K_g.scalar_type() == at::kHalf) &&
              K_g.scalar_type() == V_g.scalar_type(),
              "K_g/V_g must be bf16 or f16 and match");
  TORCH_CHECK(q_h.scalar_type() == K_g.scalar_type(),
              "q_h dtype must match K_g/V_g");

  TORCH_CHECK(q_h.dim() == 2 && K_g.dim() == 3 && V_g.dim() == 3, "bad ranks");
  TORCH_CHECK(q_h.device() == K_g.device() && q_h.device() == V_g.device(), "device mismatch");

  // Expected KV cache layout: NHD = [L, KVH, D] (sequence-first)
  TORCH_CHECK(K_g.size(0) == V_g.size(0) && K_g.size(1) == V_g.size(1) && K_g.size(2) == V_g.size(2),
              "K_g and V_g must have the same shape");
  TORCH_CHECK(q_h.size(1) == K_g.size(2), "D mismatch (q_h.size(1) vs K_g.size(2))");
  TORCH_CHECK(q_h.size(0) % K_g.size(1) == 0,
              "H must be divisible by KVH (expected KVH == K_g.size(1) for NHD)");

  TORCH_CHECK(K_g.is_contiguous() && V_g.is_contiguous(),
              "K_g/V_g must be contiguous with layout [L, KVH, D] (NHD)");
}

// ==== Pass-1 (returns m_tiles, ell_tiles, m_star, u_stash[ L, H ]) ====
static std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
pass1_scores_stats(at::Tensor q_h, at::Tensor K_g,
                   int64_t block_n = 0, bool stash = true) {
  TORCH_CHECK(q_h.is_cuda() && K_g.is_cuda(), "CUDA tensors expected");
  TORCH_CHECK((q_h.scalar_type() == at::kBFloat16 || q_h.scalar_type() == at::kHalf),
              "q_h must be bf16/f16");
  TORCH_CHECK(K_g.scalar_type() == at::kBFloat16 || K_g.scalar_type() == at::kHalf,
              "K_g must be bf16/f16");
  TORCH_CHECK(q_h.scalar_type() == K_g.scalar_type(), "q_h dtype must match K_g");

  auto q = q_h.contiguous();
  TORCH_CHECK(K_g.is_contiguous(), "K_g must be contiguous with layout [L, KVH, D] (NHD)");

  const int H   = (int)q.size(0);
  const int D   = (int)q.size(1);
  const int L   = (int)K_g.size(0);   // NHD: [L, KVH, D]
  const int KVH = (int)K_g.size(1);

  // Normalize block_n BEFORE any division.
  block_n = _normalize_block_n(L, block_n);
  TORCH_CHECK(block_n > 0, "internal error: normalized block_n must be > 0");
  const int bn = (int)block_n;

  TORCH_CHECK((int)K_g.size(2) == D, "D mismatch");
  TORCH_CHECK(H % KVH == 0, "H must be divisible by KVH");

  const int GH  = H / KVH;
  const int T_c = (L + bn - 1) / bn;

  auto optsF = q.options().dtype(at::kFloat);
  at::Tensor m_tiles   = at::empty({H, T_c}, optsF);
  at::Tensor ell_tiles = at::empty({H, T_c}, optsF);
  at::Tensor m_star    = at::empty({H},      optsF);

  // stash is [L,H] to match pass-2 read at (n*H + h)
  auto optsKV  = q.options().dtype(K_g.scalar_type());
  at::Tensor u_stash = stash ? at::empty({L, H}, optsKV) : at::Tensor();

  const float scale = 1.0f / std::sqrt((float)D);
  auto stream = at::cuda::getCurrentCUDAStream();

  // WMMA kernel requires 16-aligned head_dim and tile width.
  const bool wmma_ok = ((D & 15) == 0) && ((bn & 15) == 0);

  // Vec path only valid for the target regime.
  const bool can_vec  = (GH == 4) && (D == 128) && (bn == 256);

  const void* Kraw = (const void*)K_g.data_ptr();
  const void* Qraw = (const void*)q.data_ptr();

  const int mode = _pass1_mode_from_env();
  const bool want_vec = (mode == 2);

  if (want_vec) {
    TORCH_CHECK(can_vec,
                "SANTA_PASS1=vec requires GH=4, D=128, block_n=256. Got GH=", GH,
                " D=", D, " block_n=", bn);
  }

  const bool use_vec = (mode == 0 && can_vec) || (mode == 2);

  if (use_vec) {
    if (K_g.scalar_type() == at::kBFloat16)
      launch_pass1_tiles_vec_gqa_bf16(Qraw, Kraw, H, KVH, L, D,
                                      bn, T_c, scale,
                                      m_tiles.data_ptr<float>(), ell_tiles.data_ptr<float>(),
                                      stash ? (void*)u_stash.data_ptr() : nullptr,
                                      stream.stream());
    else
      launch_pass1_tiles_vec_gqa_f16 (Qraw, Kraw, H, KVH, L, D,
                                      bn, T_c, scale,
                                      m_tiles.data_ptr<float>(), ell_tiles.data_ptr<float>(),
                                      stash ? (void*)u_stash.data_ptr() : nullptr,
                                      stream.stream());
  } else {
    TORCH_CHECK(wmma_ok,
                "Pass-1 WMMA-GQA requires D % 16 == 0 and block_n % 16 == 0. Got D=", D,
                " block_n=", bn);
    if (K_g.scalar_type() == at::kBFloat16)
      launch_pass1_tiles_wmma_gqa_bf16(Qraw, Kraw, H, KVH, L, D,
                                       bn, T_c, scale,
                                       m_tiles.data_ptr<float>(), ell_tiles.data_ptr<float>(),
                                       stash ? (void*)u_stash.data_ptr() : nullptr,
                                       stream.stream());
    else
      launch_pass1_tiles_wmma_gqa_f16 (Qraw, Kraw, H, KVH, L, D,
                                       bn, T_c, scale,
                                       m_tiles.data_ptr<float>(), ell_tiles.data_ptr<float>(),
                                       stash ? (void*)u_stash.data_ptr() : nullptr,
                                       stream.stream());
  }

  AT_CUDA_CHECK(cudaGetLastError());
  // m_star is computed later inside launch_tile_budgets (fused reduce + budgets).
  return std::make_tuple(m_tiles, ell_tiles, m_star, u_stash);

}

// ==== decode entrypoint ====
// Returns (out [H,D] in model dtype bf16/f16, m_star [H] fp32, vrows [KVH] int or empty if want_vrows=false)
std::tuple<at::Tensor, at::Tensor, at::Tensor>
decode_systematic_scalar(at::Tensor q_h, at::Tensor K_g, at::Tensor V_g,
                         int64_t S, uint64_t seed, int64_t block_n, bool want_vrows) {
  _check_inputs_q_kv(q_h, K_g, V_g);

  auto stream = at::cuda::getCurrentCUDAStream();
  const int H   = (int)q_h.size(0);
  const int D   = (int)q_h.size(1);

  // NHD cache layout: [L, KVH, D]
  const int L   = (int)K_g.size(0);
  const int KVH = (int)K_g.size(1);

  // Normalize block_n BEFORE any division.
  block_n = _normalize_block_n(L, block_n);
  TORCH_CHECK(block_n > 0,
              "block_n must be > 0, or <= 0 for auto. Got block_n=", block_n);

  const int bn  = (int)block_n;
  const int T_c = (L + bn - 1) / bn;

  // Pass-1
  at::Tensor m_tiles, ell_tiles, m_star, u_stash;
  std::tie(m_tiles, ell_tiles, m_star, u_stash) =
      pass1_scores_stats(q_h, K_g, bn, /*stash=*/true);

  // Budgets per tile (largest remainder) => S_tiles, invdelta_tile = St/ell, a0 in [0,1)
  auto optsI = q_h.options().dtype(at::kInt);
  auto optsF = q_h.options().dtype(at::kFloat);
  at::Tensor S_tiles       = at::empty({H, T_c}, optsI);
  at::Tensor invdelta_tile = at::empty({H, T_c}, optsF);
  at::Tensor a0_tile       = at::empty({H, T_c}, optsF);

  launch_tile_budgets(m_tiles.data_ptr<float>(), ell_tiles.data_ptr<float>(), m_star.data_ptr<float>(),
                      H, T_c, (int)S, seed,
                      S_tiles.data_ptr<int>(), invdelta_tile.data_ptr<float>(), a0_tile.data_ptr<float>(),
                      stream.stream());
  AT_CUDA_CHECK(cudaGetLastError());

  // Pass-2 accumulates into fp32 [H,D]
  at::Tensor out_fp32 = at::zeros({H, D}, optsF);
  at::Tensor rowmask;
  if (want_vrows) rowmask = at::zeros({KVH, L}, optsI);

  const void* Vraw = (const void*)V_g.data_ptr();
  const void* Uraw = (const void*)u_stash.data_ptr();

  if (V_g.scalar_type() == at::kBFloat16)
    launch_pass2_grouped_fused_atomic_bf16(
        Vraw, Uraw, H, KVH, L, D, bn, T_c,
        invdelta_tile.data_ptr<float>(), a0_tile.data_ptr<float>(), S_tiles.data_ptr<int>(),
        out_fp32.data_ptr<float>(), want_vrows ? rowmask.data_ptr<int>() : nullptr, stream.stream());
  else
    launch_pass2_grouped_fused_atomic_f16(
        Vraw, Uraw, H, KVH, L, D, bn, T_c,
        invdelta_tile.data_ptr<float>(), a0_tile.data_ptr<float>(), S_tiles.data_ptr<int>(),
        out_fp32.data_ptr<float>(), want_vrows ? rowmask.data_ptr<int>() : nullptr, stream.stream());

  AT_CUDA_CHECK(cudaGetLastError());

  // Cast+normalize into model dtype (bf16/f16) on GPU.
  // If S==0, output should be zeros.
  at::Tensor out = at::empty({H, D}, q_h.options());
  const float scale_out = (S > 0) ? (1.0f / (float)S) : 0.0f;

  if (q_h.scalar_type() == at::kBFloat16)
    launch_cast_scale_out_bf16(out_fp32.data_ptr<float>(), (void*)out.data_ptr(),
                               H, D, scale_out, stream.stream());
  else
    launch_cast_scale_out_f16 (out_fp32.data_ptr<float>(), (void*)out.data_ptr(),
                               H, D, scale_out, stream.stream());

  AT_CUDA_CHECK(cudaGetLastError());

  at::Tensor vrows = at::empty({0}, optsI);
  if (want_vrows) {
    vrows = at::empty({KVH}, optsI);
    launch_reduce_vrows(rowmask.data_ptr<int>(), KVH, L, vrows.data_ptr<int>(), stream.stream());
    AT_CUDA_CHECK(cudaGetLastError());
  }

  return std::make_tuple(out, m_star, vrows);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // block_n default is 0 => auto heuristic
  m.def("decode_systematic_scalar", &decode_systematic_scalar,
        py::arg("q_h"), py::arg("K_g"), py::arg("V_g"), py::arg("S"),
        py::arg("seed") = 42ULL, py::arg("block_n") = 0, py::arg("want_vrows") = false);
}
