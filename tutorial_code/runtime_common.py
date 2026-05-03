from __future__ import annotations

import gc
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


# ============================================================
# Small data containers
# ============================================================


@dataclass
class PreparedExample:
    index: int
    input_text: str
    outputs: List[str]
    answer_prefix: str
    original_prompt_len: int
    used_prompt_len: int
    prompt_token_ids: List[int]
    raw_record: Dict[str, Any]


@dataclass
class PreparedBatch:
    batch_id: int
    prompt_len: int
    examples: List[PreparedExample]

    @property
    def batch_size(self) -> int:
        return len(self.examples)

    @property
    def example_indices(self) -> List[int]:
        return [ex.index for ex in self.examples]

    def to_tensor(self, device: torch.device) -> torch.Tensor:
        ids = [ex.prompt_token_ids for ex in self.examples]
        return torch.tensor(ids, dtype=torch.long, device=device)


# ============================================================
# Files / serialization
# ============================================================


def ensure_dir(path: str | os.PathLike[str]) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)



def write_json(path: str | os.PathLike[str], payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)



def write_jsonl(path: str | os.PathLike[str], rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ============================================================
# Dtypes / formatting
# ============================================================


def dtype_from_str(s: str) -> torch.dtype:
    s = s.lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    raise ValueError(f"Unsupported dtype string: {s}")



def dtype_to_name(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float16:
        return "float16"
    return str(dtype)



def format_optional_int(x: Optional[int]) -> str:
    if x is None:
        return "unset(default)"
    return str(int(x))


def load_model_and_tokenizer(
    model_name: str,
    dtype: torch.dtype,
    device: torch.device,
    *,
    attn_implementation: str = "sdpa",
) -> Tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    def _load_with_kwargs(load_kwargs: Dict[str, Any]) -> Any:
        if attn_implementation:
            try:
                return AutoModelForCausalLM.from_pretrained(
                    model_name,
                    attn_implementation=attn_implementation,
                    **load_kwargs,
                )
            except TypeError as exc:
                msg = str(exc)
                if "attn_implementation" not in msg and "unexpected keyword argument" not in msg:
                    raise
        return AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    try:
        model = _load_with_kwargs({"dtype": dtype, "low_cpu_mem_usage": True})
    except TypeError as exc:
        msg = str(exc)
        if "dtype" not in msg and "unexpected keyword argument" not in msg:
            raise
        model = _load_with_kwargs({"torch_dtype": dtype, "low_cpu_mem_usage": True})

    model.to(device)
    model.eval()

    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None:
        if getattr(gen_cfg, "pad_token_id", None) is None and getattr(tokenizer, "pad_token_id", None) is not None:
            gen_cfg.pad_token_id = int(tokenizer.pad_token_id)
        if getattr(gen_cfg, "eos_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
            gen_cfg.eos_token_id = int(tokenizer.eos_token_id)
    if getattr(model.config, "pad_token_id", None) is None and getattr(tokenizer, "pad_token_id", None) is not None:
        model.config.pad_token_id = int(tokenizer.pad_token_id)
    if getattr(model.config, "eos_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        model.config.eos_token_id = int(tokenizer.eos_token_id)

    return model, tokenizer


def maybe_cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ============================================================
# RoPE helpers (kept close to the original tutorial)
# ============================================================


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)



def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed



def get_rotary(model: Any, attn: Any) -> Any:
    rotary = getattr(attn, "rotary_emb", None)
    if rotary is None:
        rotary = getattr(getattr(model, "model", None), "rotary_emb", None)
    if rotary is None:
        rotary = getattr(model, "rotary_emb", None)
    if rotary is None:
        raise AttributeError(
            "Could not find rotary embedding module. Tried attn.rotary_emb, model.model.rotary_emb, model.rotary_emb."
        )
    return rotary



def rotary_cos_sin(rotary: Any, v: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    try:
        return rotary(v, position_ids)
    except TypeError:
        return rotary(v, position_ids=position_ids)


# ============================================================
# Model / cache helpers
# ============================================================


def get_lm_head(model: Any) -> Any:
    if hasattr(model, "lm_head"):
        return model.lm_head
    return model.get_output_embeddings()



def get_input_embeddings(model: Any) -> Any:
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens
    return model.get_input_embeddings()



def unpack_past_kv(past_key_values: Any) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    if hasattr(past_key_values, "to_legacy_cache"):
        past_key_values = past_key_values.to_legacy_cache()

    if isinstance(past_key_values, (tuple, list)):
        out: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer_past in past_key_values:
            if not (isinstance(layer_past, (tuple, list)) and len(layer_past) >= 2):
                raise TypeError(f"Unexpected layer past type: {type(layer_past)}")
            out.append((layer_past[0], layer_past[1]))
        return out

    raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)}")



def infer_attention_dims(model: Any) -> Dict[str, int]:
    first_layer = model.model.layers[0]
    attn = first_layer.self_attn

    n_heads = (
        getattr(attn, "num_heads", None)
        or getattr(attn, "num_attention_heads", None)
        or getattr(model.config, "num_attention_heads", None)
    )
    n_kv = getattr(attn, "num_key_value_heads", None) or getattr(model.config, "num_key_value_heads", None)
    if n_kv is None:
        n_kv = n_heads

    head_dim = getattr(attn, "head_dim", None)
    if head_dim is None:
        hidden_size = getattr(model.config, "hidden_size", None)
        if hidden_size is None or n_heads is None:
            raise RuntimeError("Could not infer hidden_size/head_dim from model config.")
        head_dim = hidden_size // n_heads

    return {
        "num_heads": int(n_heads),
        "num_kv_heads": int(n_kv),
        "head_dim": int(head_dim),
        "num_layers": int(len(model.model.layers)),
        "hidden_size": int(getattr(model.config, "hidden_size", head_dim * n_heads)),
    }


@torch.inference_mode()
def prefill_in_chunks(
    model: Any,
    input_ids: torch.Tensor,
    *,
    prefill_chunk_size: int,
) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Chunked prefill to avoid materializing full-sequence logits for very long prompts.
    Returns:
      - last-token logits: [B, V]
      - legacy-style past_kv list with per-layer tensors [B, KVH, L, D]
    """
    if prefill_chunk_size <= 0:
        raise ValueError(f"prefill_chunk_size must be > 0, got {prefill_chunk_size}")

    prompt_len = int(input_ids.shape[1])
    lm_head = get_lm_head(model)
    base_model = getattr(model, "model", None)

    past_key_values = None
    last_logits: Optional[torch.Tensor] = None

    for start in range(0, prompt_len, prefill_chunk_size):
        end = min(prompt_len, start + prefill_chunk_size)
        chunk = input_ids[:, start:end]

        if base_model is not None:
            try:
                out = base_model(
                    input_ids=chunk,
                    use_cache=True,
                    past_key_values=past_key_values,
                    return_dict=True,
                )
                past_key_values = out.past_key_values
                if end == prompt_len:
                    last_hidden = out.last_hidden_state[:, -1:, :].contiguous()
                    last_logits = lm_head(last_hidden)[:, 0, :].contiguous()
                del out
                continue
            except Exception:
                # Fall back to the full CausalLM forward path. This keeps the runtime usable
                # across transformer releases whose base-model cache APIs differ slightly.
                pass

        out = model(
            input_ids=chunk,
            use_cache=True,
            past_key_values=past_key_values,
            return_dict=True,
        )
        past_key_values = out.past_key_values
        if end == prompt_len:
            last_logits = out.logits[:, -1, :].contiguous()
        del out

    if last_logits is None:
        raise RuntimeError("Chunked prefill failed to produce final logits.")

    past_list = unpack_past_kv(past_key_values)
    del past_key_values
    return last_logits, past_list



def alloc_nhd_caches_from_prefill(
    past_list: List[Optional[Tuple[torch.Tensor, torch.Tensor]]],
    *,
    prompt_len: int,
    total_len: int,
    dtype: torch.dtype,
    device: torch.device,
    consume_past: bool = True,
) -> List[Dict[str, torch.Tensor]]:
    """
    Create batched contiguous caches per layer:
      k_nhd: [B, total_len, KVH, D]
      v_nhd: [B, total_len, KVH, D]
      k_hnd: [B, KVH, total_len, D] (view)
      v_hnd: [B, KVH, total_len, D] (view)

    The source HF prefill cache is expected to be [B, KVH, L, D].
    """
    caches: List[Dict[str, torch.Tensor]] = []
    for layer_idx, kv in enumerate(past_list):
        if kv is None:
            raise RuntimeError(f"Layer {layer_idx} past KV was already consumed.")
        k_pref, v_pref = kv

        if k_pref.dim() != 4 or v_pref.dim() != 4:
            raise RuntimeError(
                f"Expected prefill K/V tensors with rank 4, got {tuple(k_pref.shape)} and {tuple(v_pref.shape)}"
            )

        bsz, kvh, L, d = k_pref.shape
        if L != prompt_len:
            raise RuntimeError(f"Prefill prompt length mismatch: expected {prompt_len}, got {L}")
        if tuple(v_pref.shape) != tuple(k_pref.shape):
            raise RuntimeError(f"K/V shape mismatch at layer {layer_idx}: {tuple(k_pref.shape)} vs {tuple(v_pref.shape)}")

        k_nhd = torch.empty((bsz, total_len, kvh, d), device=device, dtype=dtype)
        v_nhd = torch.empty((bsz, total_len, kvh, d), device=device, dtype=dtype)

        k_nhd[:, :prompt_len].copy_(k_pref.to(dtype).transpose(1, 2).contiguous())
        v_nhd[:, :prompt_len].copy_(v_pref.to(dtype).transpose(1, 2).contiguous())

        caches.append(
            {
                "k_nhd": k_nhd,
                "v_nhd": v_nhd,
                "k_hnd": k_nhd.permute(0, 2, 1, 3),
                "v_hnd": v_nhd.permute(0, 2, 1, 3),
            }
        )

        if consume_past:
            past_list[layer_idx] = None
            del k_pref, v_pref

    if consume_past:
        gc.collect()

    return caches


# ============================================================
# Stop tokens / normalization / exact match
# ============================================================


def get_stop_token_ids(tokenizer: Any, extra_stop_strings: Sequence[str]) -> List[int]:
    stop_ids: List[int] = []
    if getattr(tokenizer, "eos_token_id", None) is not None:
        stop_ids.append(int(tokenizer.eos_token_id))

    unk_id = getattr(tokenizer, "unk_token_id", None)
    for s in extra_stop_strings:
        try:
            token_id = tokenizer.convert_tokens_to_ids(s)
        except Exception:
            token_id = None
        if token_id is None or not isinstance(token_id, int) or token_id < 0:
            continue
        if unk_id is not None and int(token_id) == int(unk_id):
            continue
        stop_ids.append(int(token_id))

    return sorted(set(stop_ids))



def normalize_prediction_text(text: str, answer_prefix: str = "") -> str:
    out = text.strip()
    prefix = (answer_prefix or "").strip()
    if prefix:
        out_cmp = out.lower()
        prefix_cmp = prefix.lower()
        if out_cmp.startswith(prefix_cmp):
            out = out[len(prefix) :].strip()
    return out



def exact_match_any(prediction_text: str, acceptable_outputs: Sequence[str], answer_prefix: str = "") -> bool:
    pred = normalize_prediction_text(prediction_text, answer_prefix=answer_prefix)
    gold = {str(x).strip() for x in acceptable_outputs}
    return pred in gold


# ============================================================
# Metrics helpers
# ============================================================


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(v) for v in values)
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac



def summarize_numeric(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    if not vals:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "ci95_lo": float("nan"),
            "ci95_hi": float("nan"),
        }

    mean = statistics.fmean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    half_width = 1.96 * std / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
    return {
        "count": len(vals),
        "mean": mean,
        "std": std,
        "median": _quantile(vals, 0.5),
        "p90": _quantile(vals, 0.9),
        "min": min(vals),
        "max": max(vals),
        "ci95_lo": mean - half_width,
        "ci95_hi": mean + half_width,
    }


# ============================================================
# Batched one-token manual decode and lockstep generation
# ============================================================


@torch.inference_mode()
def forward_one_token_manual_batched(
    model: Any,
    *,
    token_id_t: torch.Tensor,      # [B,1] long
    position_ids: torch.Tensor,    # [B,1] long
    caches: List[Dict[str, torch.Tensor]],
    pos: int,
    attention_backend: Any,
) -> torch.Tensor:
    if not (hasattr(model, "model") and hasattr(model.model, "layers")):
        raise RuntimeError("Expected a HF Llama-style model with model.model.layers")

    embed = get_input_embeddings(model)
    lm_head = get_lm_head(model)
    hidden_states = embed(token_id_t)  # [B,1,hidden]
    batch_size = int(token_id_t.shape[0])

    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn

        n_heads = (
            getattr(attn, "num_heads", None)
            or getattr(attn, "num_attention_heads", None)
            or getattr(model.config, "num_attention_heads", None)
        )
        n_kv = getattr(attn, "num_key_value_heads", None) or getattr(model.config, "num_key_value_heads", None)
        if n_kv is None:
            n_kv = n_heads

        head_dim = getattr(attn, "head_dim", None)
        if head_dim is None:
            head_dim = getattr(model.config, "hidden_size", None) // n_heads

        residual = hidden_states
        x = layer.input_layernorm(hidden_states)

        q = attn.q_proj(x)
        k = attn.k_proj(x)
        v = attn.v_proj(x)

        _, q_len, _ = q.shape
        q = q.view(batch_size, q_len, n_heads, head_dim).transpose(1, 2).contiguous()  # [B,H,1,D]
        k = k.view(batch_size, q_len, n_kv, head_dim).transpose(1, 2).contiguous()      # [B,KVH,1,D]
        v = v.view(batch_size, q_len, n_kv, head_dim).transpose(1, 2).contiguous()      # [B,KVH,1,D]

        rotary = get_rotary(model, attn)
        cos, sin = rotary_cos_sin(rotary, v, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

        kc = caches[layer_idx]["k_nhd"]
        vc = caches[layer_idx]["v_nhd"]
        kc[:, pos].copy_(k[:, :, 0, :])
        vc[:, pos].copy_(v[:, :, 0, :])

        q_bhd = q[:, :, 0, :].contiguous()
        attn_out_bhd = attention_backend.decode(q_bhd, kc, vc, valid_len=pos + 1)
        attn_out = attn_out_bhd.unsqueeze(2)  # [B,H,1,D]

        attn_out = attn_out.transpose(1, 2).reshape(batch_size, q_len, n_heads * head_dim).contiguous()
        attn_out = attn.o_proj(attn_out)
        hidden_states = residual + attn_out

        residual = hidden_states
        x = layer.post_attention_layernorm(hidden_states)
        x = layer.mlp(x)
        hidden_states = residual + x

    hidden_states = model.model.norm(hidden_states)
    logits = lm_head(hidden_states)
    return logits  # [B,1,V]


@torch.inference_mode()
def generate_lockstep_batch(
    model: Any,
    tokenizer: Any,
    *,
    prompt_ids: torch.Tensor,            # [B, L]
    prefill_logits_last: torch.Tensor,   # [B, V]
    caches: List[Dict[str, torch.Tensor]],
    attention_backend: Any,
    stop_token_ids: Sequence[int],
    max_new_tokens: int,
    lockstep_stop_mode: str,
    skip_special_tokens: bool,
    answer_prefixes: Sequence[str],
    acceptable_outputs: Sequence[Sequence[str]],
) -> Dict[str, Any]:
    device = prompt_ids.device
    batch_size, prompt_len = prompt_ids.shape
    stop_set = {int(x) for x in stop_token_ids}
    stop_fill_id = int(next(iter(stop_set))) if stop_set else int(getattr(tokenizer, "eos_token_id", 0) or 0)

    generated_all: List[List[int]] = [[] for _ in range(batch_size)]
    first_stop_step: List[Optional[int]] = [None for _ in range(batch_size)]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    cur_logits = prefill_logits_last.contiguous()

    maybe_cuda_sync(device)
    t0 = time.perf_counter()

    pos = int(prompt_len)
    steps_sampled = 0
    for step in range(int(max_new_tokens)):
        next_ids = torch.argmax(cur_logits, dim=-1)
        if finished.any():
            next_ids = torch.where(finished, torch.full_like(next_ids, stop_fill_id), next_ids)

        next_ids_list = [int(x) for x in next_ids.tolist()]
        for b, token_id in enumerate(next_ids_list):
            generated_all[b].append(token_id)
            if first_stop_step[b] is None and token_id in stop_set:
                first_stop_step[b] = step
                finished[b] = True

        steps_sampled = step + 1

        should_break = False
        if lockstep_stop_mode == "all_finished" and bool(finished.all().item()):
            should_break = True
        if step == int(max_new_tokens) - 1:
            should_break = True
        if should_break:
            break

        token_id_t = next_ids.view(batch_size, 1)
        position_ids = torch.full((batch_size, 1), pos, dtype=torch.long, device=device)
        cur_logits = forward_one_token_manual_batched(
            model,
            token_id_t=token_id_t,
            position_ids=position_ids,
            caches=caches,
            pos=pos,
            attention_backend=attention_backend,
        )[:, 0, :].contiguous()
        pos += 1

    maybe_cuda_sync(device)
    t1 = time.perf_counter()

    records: List[Dict[str, Any]] = []
    visible_generated_tokens = 0
    for b in range(batch_size):
        stop_step = first_stop_step[b]
        full_ids = list(generated_all[b])
        if stop_step is None:
            visible_ids = list(full_ids)
        else:
            visible_ids = list(full_ids[:stop_step])

        visible_text = tokenizer.decode(visible_ids, skip_special_tokens=skip_special_tokens)
        normalized_text = normalize_prediction_text(visible_text, answer_prefix=answer_prefixes[b])
        is_exact_match = exact_match_any(visible_text, acceptable_outputs[b], answer_prefix=answer_prefixes[b])
        visible_generated_tokens += len(visible_ids)

        records.append(
            {
                "generated_token_ids_all": full_ids,
                "generated_token_ids_visible": visible_ids,
                "generated_text": visible_text,
                "generated_text_normalized": normalized_text,
                "stop_step": stop_step,
                "exact_match": bool(is_exact_match),
            }
        )

    timed_generated_tokens = int(batch_size * steps_sampled)
    decode_time_s = float(t1 - t0)
    return {
        "decode_time_s": decode_time_s,
        "steps_sampled": int(steps_sampled),
        "timed_generated_tokens": timed_generated_tokens,
        "visible_generated_tokens": int(visible_generated_tokens),
        "examples": records,
    }
