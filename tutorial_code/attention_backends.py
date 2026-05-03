from __future__ import annotations

import importlib
import math
from importlib import metadata as importlib_metadata
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from env_utils import configure_flashinfer_runtime, format_flashinfer_exception


FA2_PAPER_TARGET_VERSION = "2.7.4.post1"
FLASHINFER_PAPER_TARGET_VERSION = "0.6.6"


class BackendError(RuntimeError):
    def __init__(self, message: str, *, details: Optional[str] = None) -> None:
        super().__init__(message)
        self.details = details


class DecodeAttentionBackend:
    name: str = "backend"
    paper_role: str = "auxiliary"
    exact_attention: Optional[bool] = None
    supports_true_batch_decode: Optional[bool] = None

    def decode(self, q_bhd: torch.Tensor, k_nhd: torch.Tensor, v_nhd: torch.Tensor, valid_len: int) -> torch.Tensor:
        raise NotImplementedError

    def info(self) -> Dict[str, Any]:
        return {
            "backend": self.name,
            "paper_role": self.paper_role,
            "exact_attention": self.exact_attention,
            "supports_true_batch_decode": self.supports_true_batch_decode,
        }

    def smoke_test(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        valid_len: int,
    ) -> Dict[str, Any]:
        batch_size = 2
        q = torch.randn(batch_size, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, valid_len, num_kv_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, valid_len, num_kv_heads, head_dim, device=device, dtype=dtype)
        out = self.decode(q, k, v, valid_len)
        expected_shape = (batch_size, num_heads, head_dim)
        if out.shape != expected_shape:
            raise BackendError(
                f"Smoke test returned wrong shape for {self.name}: got {tuple(out.shape)}, expected {expected_shape}"
            )
        return {
            "backend": self.name,
            "batch_size": batch_size,
            "valid_len": int(valid_len),
            "output_shape": list(out.shape),
            **self.info(),
        }


class SDPABackend(DecodeAttentionBackend):
    name = "sdpa"
    paper_role = "debug_exact_reference"
    exact_attention = True
    supports_true_batch_decode = True

    def decode(self, q_bhd: torch.Tensor, k_nhd: torch.Tensor, v_nhd: torch.Tensor, valid_len: int) -> torch.Tensor:
        q = q_bhd.unsqueeze(2)  # [B,H,1,D]
        k = k_nhd[:, :valid_len].permute(0, 2, 1, 3)
        v = v_nhd[:, :valid_len].permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=(q.shape[1] != k.shape[1]),
        )
        return out[:, :, 0, :].contiguous()


class FA2Backend(DecodeAttentionBackend):
    name = "fa2"
    paper_role = "main_exact_dense_batched_baseline"
    exact_attention = True
    supports_true_batch_decode = True

    def __init__(
        self,
        *,
        expected_version: str = FA2_PAPER_TARGET_VERSION,
        version_policy: str = "warn",
    ) -> None:
        self.expected_version = str(expected_version)
        self.version_policy = str(version_policy).lower()
        self.actual_mode = "kvcache_batched"
        self.module_name: Optional[str] = None
        self.module: Optional[Any] = None
        self.decode_fn = None
        self.version_warning: Optional[str] = None
        self._cache_seqlens_buffers: Dict[Tuple[str, int], torch.Tensor] = {}

        import_errors: List[str] = []
        for candidate in ("flash_attn", "flash_attn.flash_attn_interface"):
            try:
                module = importlib.import_module(candidate)
            except Exception as exc:
                import_errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
                continue
            fn = getattr(module, "flash_attn_with_kvcache", None)
            if fn is None:
                import_errors.append(f"{candidate}: missing flash_attn_with_kvcache")
                continue
            self.module_name = candidate
            self.module = module
            self.decode_fn = fn
            break

        if self.decode_fn is None:
            raise BackendError(
                "Failed to import FlashAttention-2 decode KV-cache API.",
                details=(
                    "Expected flash_attn_with_kvcache in flash_attn or flash_attn.flash_attn_interface. "
                    f"Tried: {' | '.join(import_errors)}"
                ),
            )

        self.version = self._resolve_version(self.module)
        if self.expected_version and self.version != self.expected_version:
            self.version_warning = (
                f"flash-attn version mismatch: expected {self.expected_version}, got {self.version}. "
                f"This can change API or performance behavior."
            )
            if self.version_policy == "error":
                raise BackendError(self.version_warning)
            if self.version_policy == "warn":
                print(f"[fa2 warning] {self.version_warning}")

    @staticmethod
    def _resolve_version(module: Any) -> str:
        for dist_name in ("flash-attn", "flash_attn"):
            try:
                return str(importlib_metadata.version(dist_name))
            except Exception:
                pass
        version = getattr(module, "__version__", None)
        return str(version) if version is not None else "unknown"

    @staticmethod
    def _unwrap_tensor_out(x: Any) -> torch.Tensor:
        if isinstance(x, (tuple, list)):
            return x[0]
        return x

    def _cache_seqlens(self, batch_size: int, device: torch.device, valid_len: int) -> torch.Tensor:
        key = (str(device), int(batch_size))
        buf = self._cache_seqlens_buffers.get(key)
        if buf is None or buf.numel() != batch_size or buf.device != device:
            buf = torch.empty((batch_size,), device=device, dtype=torch.int32)
            self._cache_seqlens_buffers[key] = buf
        buf.fill_(int(valid_len))
        return buf

    def _call_decode(
        self,
        q_b1hd: torch.Tensor,
        k_nhd: torch.Tensor,
        v_nhd: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        sm_scale = 1.0 / math.sqrt(float(q_b1hd.shape[-1]))
        attempts = [
            lambda: self.decode_fn(
                q=q_b1hd,
                k_cache=k_nhd,
                v_cache=v_nhd,
                cache_seqlens=cache_seqlens,
                softmax_scale=sm_scale,
                causal=True,
            ),
            lambda: self.decode_fn(
                q_b1hd,
                k_nhd,
                v_nhd,
                cache_seqlens=cache_seqlens,
                softmax_scale=sm_scale,
                causal=True,
            ),
            lambda: self.decode_fn(
                q_b1hd,
                k_nhd,
                v_nhd,
                cache_seqlens=cache_seqlens,
                causal=True,
            ),
        ]
        last_type_error: Optional[TypeError] = None
        for attempt in attempts:
            try:
                return attempt()
            except TypeError as exc:
                last_type_error = exc
                continue
            except Exception as exc:
                raise BackendError(
                    f"FlashAttention-2 decode call failed: {type(exc).__name__}: {exc}",
                    details=(
                        f"module={self.module_name} version={self.version} "
                        f"q_shape={tuple(q_b1hd.shape)} k_cache_shape={tuple(k_nhd.shape)} "
                        f"v_cache_shape={tuple(v_nhd.shape)} valid_len={int(cache_seqlens[0].item())}"
                    ),
                ) from exc
        raise BackendError(
            f"FlashAttention-2 decode call failed: {type(last_type_error).__name__}: {last_type_error}",
            details=(
                f"module={self.module_name} version={self.version} "
                f"q_shape={tuple(q_b1hd.shape)} k_cache_shape={tuple(k_nhd.shape)} "
                f"v_cache_shape={tuple(v_nhd.shape)} valid_len={int(cache_seqlens[0].item())}"
            ),
        )

    def decode(self, q_bhd: torch.Tensor, k_nhd: torch.Tensor, v_nhd: torch.Tensor, valid_len: int) -> torch.Tensor:
        if q_bhd.device.type != "cuda":
            raise BackendError("FA2 backend requires CUDA tensors.")
        if q_bhd.dim() != 3:
            raise BackendError(f"Expected q_bhd rank 3, got shape {tuple(q_bhd.shape)}")

        batch_size = int(q_bhd.shape[0])
        q_b1hd = q_bhd.unsqueeze(1).contiguous()  # [B,1,H,D]
        cache_seqlens = self._cache_seqlens(batch_size, q_bhd.device, valid_len)
        out = self._unwrap_tensor_out(self._call_decode(q_b1hd, k_nhd, v_nhd, cache_seqlens))
        if out.dim() != 4 or out.shape[1] != 1:
            raise BackendError(
                f"Unexpected FA2 decode output shape: got {tuple(out.shape)}, expected [B,1,H,D]"
            )
        return out[:, 0, :, :].contiguous()

    def info(self) -> Dict[str, Any]:
        return {
            **super().info(),
            "module": self.module_name,
            "version": self.version,
            "expected_version": self.expected_version,
            "version_warning": self.version_warning,
            "decode_kernel": "flash_attn_with_kvcache",
            "actual_mode": self.actual_mode,
            "cache_layout": "NHD_contiguous",
            "cache_seqlens_mode": "uniform_int32_tensor",
        }


class FlashInferBackend(DecodeAttentionBackend):
    name = "flashinfer"
    paper_role = "legacy_reference_not_main_batched_baseline"
    exact_attention = True

    def __init__(
        self,
        *,
        mode: str = "single_loop",
        use_tensor_cores: bool = True,
        expected_version: str = FLASHINFER_PAPER_TARGET_VERSION,
        version_policy: str = "warn",
        jit_mode: str = "auto",
        preload_libstdcpp: str = "auto",
    ) -> None:
        self.mode = str(mode)
        self.use_tensor_cores = bool(use_tensor_cores)
        self.actual_mode = self.mode
        self.expected_version = str(expected_version)
        self.version_policy = str(version_policy).lower()
        self.runtime = configure_flashinfer_runtime(jit_mode=jit_mode, preload_libstdcpp=preload_libstdcpp)
        self.supports_true_batch_decode = self.mode == "batch_compact"

        try:
            self.flashinfer = importlib.import_module("flashinfer")
        except Exception as exc:
            raise BackendError(
                "Failed to import flashinfer.",
                details=format_flashinfer_exception(exc, self.runtime),
            ) from exc

        self.version = getattr(self.flashinfer, "__version__", "unknown")
        self.version_warning: Optional[str] = None
        if self.expected_version and self.version != self.expected_version:
            self.version_warning = (
                f"flashinfer version mismatch: expected {self.expected_version}, got {self.version}. "
                f"This can change API and performance behavior."
            )
            if self.version_policy == "error":
                raise BackendError(self.version_warning)
            if self.version_policy == "warn":
                print(f"[flashinfer warning] {self.version_warning}")

        self.single_fn = getattr(self.flashinfer, "single_decode_with_kv_cache", None)
        self.batch_fn = getattr(self.flashinfer, "batch_decode_with_padded_kv_cache", None)

        decode_mod = getattr(self.flashinfer, "decode", None)
        if self.single_fn is None and decode_mod is not None:
            self.single_fn = getattr(decode_mod, "single_decode_with_kv_cache", None)
        if self.batch_fn is None and decode_mod is not None:
            self.batch_fn = getattr(decode_mod, "batch_decode_with_padded_kv_cache", None)

        if self.single_fn is None:
            raise BackendError(
                "Could not find flashinfer single decode API.",
                details=(
                    "Expected flashinfer.single_decode_with_kv_cache or "
                    "flashinfer.decode.single_decode_with_kv_cache to exist."
                ),
            )

    def _call_single(self, q_h_d: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        sm_scale = 1.0 / math.sqrt(float(q_h_d.shape[-1]))
        attempts = [
            lambda: self.single_fn(
                q_h_d,
                k,
                v,
                kv_layout="NHD",
                use_tensor_cores=self.use_tensor_cores,
                sm_scale=sm_scale,
            ),
            lambda: self.single_fn(q_h_d, k, v, kv_layout="NHD", sm_scale=sm_scale),
            lambda: self.single_fn(q_h_d, k, v),
        ]
        last_type_error: Optional[TypeError] = None
        for attempt in attempts:
            try:
                return attempt()
            except TypeError as exc:
                last_type_error = exc
                continue
            except Exception as exc:
                raise BackendError(
                    f"FlashInfer single decode call failed: {type(exc).__name__}: {exc}",
                    details=format_flashinfer_exception(exc, self.runtime),
                ) from exc
        raise BackendError(
            f"FlashInfer single decode call failed: {type(last_type_error).__name__}: {last_type_error}",
            details=format_flashinfer_exception(last_type_error, self.runtime) if last_type_error is not None else None,
        )

    def _call_batch_compact(self, q_bhd: torch.Tensor, k_nhd: torch.Tensor, v_nhd: torch.Tensor, valid_len: int) -> torch.Tensor:
        if self.batch_fn is None:
            raise BackendError(
                "FlashInfer batch padded decode function was requested, but the installed flashinfer package does not expose it."
            )
        sm_scale = 1.0 / math.sqrt(float(q_bhd.shape[-1]))
        k_compact = k_nhd[:, :valid_len].contiguous()
        v_compact = v_nhd[:, :valid_len].contiguous()
        q_compact = q_bhd.contiguous()
        attempts = [
            lambda: self.batch_fn(
                q_compact,
                k_compact,
                v_compact,
                kv_layout="NHD",
                use_tensor_cores=self.use_tensor_cores,
                sm_scale=sm_scale,
            ),
            lambda: self.batch_fn(q_compact, k_compact, v_compact, kv_layout="NHD", sm_scale=sm_scale),
            lambda: self.batch_fn(q_compact, k_compact, v_compact),
        ]
        last_type_error: Optional[TypeError] = None
        for attempt in attempts:
            try:
                return attempt()
            except TypeError as exc:
                last_type_error = exc
                continue
            except Exception as exc:
                raise BackendError(
                    f"FlashInfer batch decode call failed: {type(exc).__name__}: {exc}",
                    details=format_flashinfer_exception(exc, self.runtime),
                ) from exc
        raise BackendError(
            f"FlashInfer batch decode call failed: {type(last_type_error).__name__}: {last_type_error}",
            details=format_flashinfer_exception(last_type_error, self.runtime) if last_type_error is not None else None,
        )

    def decode(self, q_bhd: torch.Tensor, k_nhd: torch.Tensor, v_nhd: torch.Tensor, valid_len: int) -> torch.Tensor:
        if self.mode == "batch_compact":
            self.actual_mode = "batch_compact"
            return self._call_batch_compact(q_bhd, k_nhd, v_nhd, valid_len).contiguous()

        self.actual_mode = "single_loop"
        outputs = []
        batch_size = int(q_bhd.shape[0])
        for b in range(batch_size):
            k_b = k_nhd[b, :valid_len]
            v_b = v_nhd[b, :valid_len]
            out_b = self._call_single(q_bhd[b].contiguous(), k_b, v_b)
            outputs.append(out_b)
        return torch.stack(outputs, dim=0).contiguous()

    def info(self) -> Dict[str, Any]:
        return {
            **super().info(),
            "version": self.version,
            "expected_version": self.expected_version,
            "version_warning": self.version_warning,
            "requested_mode": self.mode,
            "actual_mode": self.actual_mode,
            "has_batch_padded_api": self.batch_fn is not None,
            "use_tensor_cores": self.use_tensor_cores,
        }


class _SantaBaseBackend(DecodeAttentionBackend):
    module_candidates: Sequence[str] = ()
    name = "santa"
    paper_role = "main_batched_sparse_baseline"
    exact_attention = False

    def __init__(self, *, S: int, seed: int, block_n: Optional[int]) -> None:
        self.S = int(S)
        self.seed = int(seed)
        self.block_n = None if block_n is None else int(block_n)
        self.actual_mode = "unknown"
        self.module = None
        self.module_name = None
        self.supports_true_batch_decode = False
        import_errors: List[str] = []
        for candidate in self.module_candidates:
            try:
                self.module = importlib.import_module(candidate)
                self.module_name = candidate
                break
            except Exception as exc:
                import_errors.append(f"{candidate}: {exc}")
        if self.module is None:
            raise BackendError(
                f"Could not import any {self.name} extension module. Tried: {', '.join(self.module_candidates)}.",
                details=" | ".join(import_errors),
            )
        self.supports_true_batch_decode = hasattr(self.module, "decode_systematic_batched")

    @staticmethod
    def _unwrap_tensor_out(x: Any) -> torch.Tensor:
        if isinstance(x, (tuple, list)):
            return x[0]
        return x

    def _get_decode_fn(self):
        if hasattr(self.module, "decode_systematic_batched"):
            return getattr(self.module, "decode_systematic_batched")
        if hasattr(self.module, "decode_systematic_scalar"):
            return getattr(self.module, "decode_systematic_scalar")
        raise BackendError(f"Module {self.module_name} does not expose a decode entry point.")

    def decode(self, q_bhd: torch.Tensor, k_nhd: torch.Tensor, v_nhd: torch.Tensor, valid_len: int) -> torch.Tensor:
        fn = self._get_decode_fn()
        kwargs = {
            "seed": self.seed,
            "want_vrows": False,
            "valid_len": int(valid_len),
        }
        if self.block_n is not None:
            kwargs["block_n"] = int(self.block_n)

        try:
            out = fn(q_bhd.contiguous(), k_nhd, v_nhd, self.S, **kwargs)
            self.actual_mode = "batch"
            return self._unwrap_tensor_out(out).contiguous()
        except Exception as exc:
            msg = str(exc)
            should_fallback = (
                q_bhd.dim() == 3
                and (
                    "Expected q_h [H,D]" in msg
                    or "Expected q_h [H, D]" in msg
                    or "valid_len" in msg
                    or "rank" in msg
                    or "shape" in msg
                )
            )
            if not should_fallback:
                raise

        kwargs.pop("valid_len", None)
        self.actual_mode = "single_loop_fallback"
        outputs = []
        batch_size = int(q_bhd.shape[0])
        for b in range(batch_size):
            k_b = k_nhd[b, :valid_len].contiguous()
            v_b = v_nhd[b, :valid_len].contiguous()
            out_b = fn(q_bhd[b].contiguous(), k_b, v_b, self.S, **kwargs)
            outputs.append(self._unwrap_tensor_out(out_b))
        return torch.stack(outputs, dim=0).contiguous()

    def info(self) -> Dict[str, Any]:
        return {
            **super().info(),
            "module": self.module_name,
            "requested_S": self.S,
            "seed": self.seed,
            "block_n": self.block_n,
            "actual_mode": self.actual_mode,
        }


class SantaFlashBackend(_SantaBaseBackend):
    name = "santa_flash"
    module_candidates = ("santa_flash_batch_cuda", "santa_cuda")


class SantaPropBackend(_SantaBaseBackend):
    name = "santa_prop"
    module_candidates = ("santa_prop_batch_cuda",)


def _make_one_backend(
    raw_name: str,
    *,
    fa2_expected_version: str,
    fa2_version_policy: str,
    santa_s: int,
    santa_seed: int,
    santa_block_n: Optional[int],
    flashinfer_mode: str,
    flashinfer_use_tensor_cores: bool,
    flashinfer_expected_version: str,
    flashinfer_version_policy: str,
    flashinfer_jit_mode: str,
    flashinfer_preload_libstdcpp: str,
) -> DecodeAttentionBackend:
    name = raw_name.lower().strip()
    if name == "sdpa":
        return SDPABackend()
    if name in ("fa2", "flashattn2", "flash_attention_2", "flash_attn"):
        return FA2Backend(
            expected_version=fa2_expected_version,
            version_policy=fa2_version_policy,
        )
    if name == "flashinfer":
        return FlashInferBackend(
            mode=flashinfer_mode,
            use_tensor_cores=flashinfer_use_tensor_cores,
            expected_version=flashinfer_expected_version,
            version_policy=flashinfer_version_policy,
            jit_mode=flashinfer_jit_mode,
            preload_libstdcpp=flashinfer_preload_libstdcpp,
        )
    if name in ("santa", "santa_flash"):
        return SantaFlashBackend(
            S=santa_s,
            seed=santa_seed,
            block_n=santa_block_n,
        )
    if name == "santa_prop":
        return SantaPropBackend(
            S=santa_s,
            seed=santa_seed,
            block_n=santa_block_n,
        )
    raise ValueError(f"Unsupported backend name: {raw_name}")


def build_backends(
    backend_names: Iterable[str],
    *,
    fa2_expected_version: str = FA2_PAPER_TARGET_VERSION,
    fa2_version_policy: str = "warn",
    santa_s: int,
    santa_seed: int,
    santa_block_n: Optional[int],
    flashinfer_mode: str,
    flashinfer_use_tensor_cores: bool,
    flashinfer_expected_version: str = FLASHINFER_PAPER_TARGET_VERSION,
    flashinfer_version_policy: str = "warn",
    flashinfer_jit_mode: str = "auto",
    flashinfer_preload_libstdcpp: str = "auto",
    skip_init_failures: Optional[bool] = None,
) -> Union[List[DecodeAttentionBackend], Tuple[List[DecodeAttentionBackend], List[Dict[str, Any]]]]:
    backends: List[DecodeAttentionBackend] = []
    init_errors: List[Dict[str, Any]] = []

    for raw_name in backend_names:
        try:
            backend = _make_one_backend(
                raw_name,
                fa2_expected_version=fa2_expected_version,
                fa2_version_policy=fa2_version_policy,
                santa_s=santa_s,
                santa_seed=santa_seed,
                santa_block_n=santa_block_n,
                flashinfer_mode=flashinfer_mode,
                flashinfer_use_tensor_cores=flashinfer_use_tensor_cores,
                flashinfer_expected_version=flashinfer_expected_version,
                flashinfer_version_policy=flashinfer_version_policy,
                flashinfer_jit_mode=flashinfer_jit_mode,
                flashinfer_preload_libstdcpp=flashinfer_preload_libstdcpp,
            )
            backends.append(backend)
        except Exception as exc:
            if skip_init_failures:
                init_errors.append(
                    {
                        "backend": str(raw_name),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "details": getattr(exc, "details", None),
                    }
                )
                continue
            raise

    if skip_init_failures is None:
        return backends
    return backends, init_errors
