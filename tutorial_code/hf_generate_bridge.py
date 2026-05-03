from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from runtime_common import (
    alloc_nhd_caches_from_prefill,
    exact_match_any,
    forward_one_token_manual_batched,
    maybe_cuda_sync,
    normalize_prediction_text,
    prefill_in_chunks,
)


@dataclass
class HFGenerateTrace:
    prefill_time_s: float = 0.0
    cache_setup_time_s: float = 0.0
    decode_time_s: float = 0.0
    inner_wall_time_s: float = 0.0
    generate_api_wall_time_s: float = 0.0
    steps_sampled: int = 0
    timed_generated_tokens: int = 0
    visible_generated_tokens: int = 0
    stop_fill_id: Optional[int] = None
    first_stop_step: List[Optional[int]] = field(default_factory=list)
    generated_token_ids_all: List[List[int]] = field(default_factory=list)

    @property
    def end_to_end_wall_time_s(self) -> float:
        if self.generate_api_wall_time_s > 0.0:
            return self.generate_api_wall_time_s
        return self.inner_wall_time_s


def ensure_hf_custom_generate_available(model: Any) -> None:
    try:
        sig = inspect.signature(model.generate)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Could not introspect model.generate to verify HF custom_generate support.") from exc
    if "custom_generate" not in sig.parameters:
        raise RuntimeError(
            "This transformers build does not expose generate(custom_generate=...). "
            "Please upgrade transformers to a version that supports the official custom_generate hook."
        )


def _as_list_of_ints(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, (list, tuple, set)):
        return [int(x) for x in value]
    return []


def resolve_stop_token_ids(
    model: Any,
    generation_config: Any,
    extra_stop_token_ids: Sequence[int],
) -> List[int]:
    stop_ids = list(int(x) for x in extra_stop_token_ids)
    stop_ids.extend(_as_list_of_ints(getattr(generation_config, "eos_token_id", None)))
    stop_ids.extend(_as_list_of_ints(getattr(getattr(model, "generation_config", None), "eos_token_id", None)))
    stop_ids.extend(_as_list_of_ints(getattr(getattr(model, "config", None), "eos_token_id", None)))
    return sorted(set(int(x) for x in stop_ids if int(x) >= 0))


def resolve_pad_token_id(model: Any, generation_config: Any, fallback_stop_ids: Sequence[int]) -> int:
    for candidate in (
        getattr(generation_config, "pad_token_id", None),
        getattr(getattr(model, "generation_config", None), "pad_token_id", None),
        getattr(getattr(model, "config", None), "pad_token_id", None),
    ):
        if candidate is not None:
            return int(candidate)
    if fallback_stop_ids:
        return int(fallback_stop_ids[0])
    for candidate in (
        getattr(generation_config, "eos_token_id", None),
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
        getattr(getattr(model, "config", None), "eos_token_id", None),
    ):
        ids = _as_list_of_ints(candidate)
        if ids:
            return int(ids[0])
    return 0


def choose_surrogate_pad_token_id_for_hf_generate(
    *,
    model: Any,
    tokenizer: Any,
    prompt_ids: torch.Tensor,
    preferred_pad_token_id: Optional[int],
    stop_token_ids: Sequence[int],
) -> int:
    """Pick a pad_token_id that lets HF infer an all-ones attention_mask without warnings.

    For this benchmark path, prompts are uniform-length and unpadded. Some Transformers
    versions warn when attention_mask is omitted and pad_token_id == eos_token_id. To avoid
    that warning without passing attention_mask into the compatibility-sensitive custom_generate path, we
    choose a surrogate pad token that:
      - is not an EOS/stop token, and
      - does not appear anywhere in the prompt batch.

    Because no actual padding is present, this yields the same inferred all-ones mask.
    """
    eos_like = set(int(x) for x in stop_token_ids if int(x) >= 0)

    if preferred_pad_token_id is not None:
        pad_id = int(preferred_pad_token_id)
        if pad_id not in eos_like:
            return pad_id

    vocab_size = None
    for candidate in (
        getattr(getattr(model, "config", None), "vocab_size", None),
        getattr(tokenizer, "vocab_size", None),
    ):
        if candidate is not None:
            vocab_size = int(candidate)
            break
    if vocab_size is None:
        try:
            vocab_size = int(len(tokenizer))
        except Exception:
            vocab_size = None

    if vocab_size is None or vocab_size <= 0:
        if preferred_pad_token_id is not None:
            return int(preferred_pad_token_id)
        if eos_like:
            # Fall back to EOS if no surrogate pad token is available.
            return int(sorted(eos_like)[0])
        return 0

    used_prompt_ids = set(int(x) for x in torch.unique(prompt_ids).tolist())
    forbidden = used_prompt_ids | eos_like

    # Scan from the top of the vocab down; prompt batches use only a tiny fraction of ids.
    for token_id in range(vocab_size - 1, -1, -1):
        if token_id not in forbidden:
            return int(token_id)

    if preferred_pad_token_id is not None:
        return int(preferred_pad_token_id)
    if eos_like:
        return int(sorted(eos_like)[0])
    return 0


def resolve_max_new_tokens(
    input_ids: torch.Tensor,
    generation_config: Any,
    stopping_criteria: Optional[Any],
) -> int:
    max_new_tokens = getattr(generation_config, "max_new_tokens", None)
    if max_new_tokens is not None:
        return int(max_new_tokens)

    if stopping_criteria is not None:
        for criterion in stopping_criteria:
            max_length = getattr(criterion, "max_length", None)
            if max_length is not None:
                return max(0, int(max_length) - int(input_ids.shape[1]))

    max_length = getattr(generation_config, "max_length", None)
    if max_length is not None:
        return max(0, int(max_length) - int(input_ids.shape[1]))

    raise ValueError("Could not infer max_new_tokens from generation_config or stopping_criteria.")


def build_hf_custom_generate_loop(
    *,
    attention_backend: Any,
    prefill_chunk_size: int,
    extra_stop_token_ids: Sequence[int],
    lockstep_stop_mode: str,
    runtime_trace: Optional[HFGenerateTrace] = None,
):
    if lockstep_stop_mode not in {"fixed", "all_finished"}:
        raise ValueError(f"Unsupported lockstep_stop_mode: {lockstep_stop_mode}")

    @torch.inference_mode()
    def custom_loop(
        model: Any,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        logits_processor: Optional[Any] = None,
        stopping_criteria: Optional[Any] = None,
        generation_config: Optional[Any] = None,
        streamer: Optional[Any] = None,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        attention_mask = attention_mask if attention_mask is not None else model_kwargs.pop("attention_mask", None)
        model_kwargs.pop("decoder_attention_mask", None)
        # Current benchmark path uses uniform-length unpadded prompts, so we do not consume
        # the HF-prepared mask further. We still normalize it here to tolerate differences
        # across Transformers versions in how custom_generate forwards kwargs.
        del attention_mask, model_kwargs

        if generation_config is None:
            generation_config = getattr(model, "generation_config", None)
        if generation_config is None:
            raise ValueError("generation_config is required for the custom HF generate loop.")

        if input_ids.dim() != 2:
            raise ValueError(f"Expected input_ids rank 2, got shape {tuple(input_ids.shape)}")

        batch_size, prompt_len = input_ids.shape
        max_new_tokens = resolve_max_new_tokens(input_ids, generation_config, stopping_criteria)
        stop_ids = resolve_stop_token_ids(model, generation_config, extra_stop_token_ids)
        pad_token_id = resolve_pad_token_id(model, generation_config, stop_ids)
        stop_fill_id = int(stop_ids[0]) if stop_ids else int(pad_token_id)

        if streamer is not None:
            # We keep this simple and deterministic for the benchmark path.
            # Streamers are not needed for the paper benchmark scripts.
            raise NotImplementedError("Streamer support is not implemented for this custom HF benchmark loop.")

        if runtime_trace is not None:
            runtime_trace.prefill_time_s = 0.0
            runtime_trace.cache_setup_time_s = 0.0
            runtime_trace.decode_time_s = 0.0
            runtime_trace.inner_wall_time_s = 0.0
            runtime_trace.steps_sampled = 0
            runtime_trace.timed_generated_tokens = 0
            runtime_trace.visible_generated_tokens = 0
            runtime_trace.stop_fill_id = int(stop_fill_id)
            runtime_trace.first_stop_step = [None for _ in range(int(batch_size))]
            runtime_trace.generated_token_ids_all = [[] for _ in range(int(batch_size))]

        stop_set = {int(x) for x in stop_ids}
        use_logits_processor = logits_processor is not None and len(logits_processor) > 0

        # Keep a running copy of sequences only when logits processors need the full prefix.
        current_ids = input_ids.contiguous() if use_logits_processor else None

        maybe_cuda_sync(input_ids.device)
        t_inner0 = time.perf_counter()

        do_sample = bool(getattr(generation_config, "do_sample", False))
        num_beams = int(getattr(generation_config, "num_beams", 1) or 1)
        if do_sample or num_beams != 1:
            raise NotImplementedError(
                "This custom HF benchmark loop currently supports greedy decoding only (do_sample=False, num_beams=1)."
            )

        maybe_cuda_sync(input_ids.device)
        t_prefill0 = time.perf_counter()
        prefill_logits_last, past_list = prefill_in_chunks(
            model,
            input_ids,
            prefill_chunk_size=int(prefill_chunk_size),
        )
        maybe_cuda_sync(input_ids.device)
        t_prefill1 = time.perf_counter()

        maybe_cuda_sync(input_ids.device)
        t_cache0 = time.perf_counter()
        model_dtype = getattr(model, "dtype", None)
        if model_dtype is None:
            try:
                model_dtype = next(model.parameters()).dtype
            except StopIteration:
                raise RuntimeError("Could not infer model dtype for contiguous cache allocation.")
        caches = alloc_nhd_caches_from_prefill(
            past_list,
            prompt_len=int(prompt_len),
            total_len=int(prompt_len + max_new_tokens),
            dtype=model_dtype,
            device=input_ids.device,
            consume_past=True,
        )
        maybe_cuda_sync(input_ids.device)
        t_cache1 = time.perf_counter()

        cur_logits = prefill_logits_last.contiguous()
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        first_stop_step: List[Optional[int]] = [None for _ in range(int(batch_size))]
        generated_all: List[List[int]] = [[] for _ in range(int(batch_size))]

        maybe_cuda_sync(input_ids.device)
        t_decode0 = time.perf_counter()

        pos = int(prompt_len)
        steps_sampled = 0
        for step in range(int(max_new_tokens)):
            processed_logits = cur_logits
            if use_logits_processor:
                processed_logits = logits_processor(current_ids, processed_logits)

            next_ids = torch.argmax(processed_logits, dim=-1)
            if finished.any():
                next_ids = torch.where(finished, torch.full_like(next_ids, stop_fill_id), next_ids)

            next_ids_list = [int(x) for x in next_ids.tolist()]
            for b_idx, token_id in enumerate(next_ids_list):
                generated_all[b_idx].append(token_id)
                if first_stop_step[b_idx] is None and token_id in stop_set:
                    first_stop_step[b_idx] = step
                    finished[b_idx] = True

            next_ids_col = next_ids.view(batch_size, 1)
            if current_ids is not None:
                current_ids = torch.cat((current_ids, next_ids_col), dim=1)

            steps_sampled = step + 1

            should_break = False
            if lockstep_stop_mode == "all_finished" and bool(finished.all().item()):
                should_break = True
            if step == int(max_new_tokens) - 1:
                should_break = True
            if should_break:
                break

            position_ids = torch.full((batch_size, 1), pos, dtype=torch.long, device=input_ids.device)
            cur_logits = forward_one_token_manual_batched(
                model,
                token_id_t=next_ids_col,
                position_ids=position_ids,
                caches=caches,
                pos=pos,
                attention_backend=attention_backend,
            )[:, 0, :].contiguous()
            pos += 1

        maybe_cuda_sync(input_ids.device)
        t_decode1 = time.perf_counter()
        maybe_cuda_sync(input_ids.device)
        t_inner1 = time.perf_counter()

        generated_tensor = input_ids.new_tensor(generated_all, dtype=torch.long)
        sequences = torch.cat((input_ids, generated_tensor), dim=1)

        visible_generated_tokens = 0
        for ids, stop_step in zip(generated_all, first_stop_step):
            if stop_step is None:
                visible_generated_tokens += len(ids)
            else:
                visible_generated_tokens += int(stop_step)

        if runtime_trace is not None:
            runtime_trace.prefill_time_s = float(t_prefill1 - t_prefill0)
            runtime_trace.cache_setup_time_s = float(t_cache1 - t_cache0)
            runtime_trace.decode_time_s = float(t_decode1 - t_decode0)
            runtime_trace.inner_wall_time_s = float(t_inner1 - t_inner0)
            runtime_trace.steps_sampled = int(steps_sampled)
            runtime_trace.timed_generated_tokens = int(batch_size * steps_sampled)
            runtime_trace.visible_generated_tokens = int(visible_generated_tokens)
            runtime_trace.stop_fill_id = int(stop_fill_id)
            runtime_trace.first_stop_step = list(first_stop_step)
            runtime_trace.generated_token_ids_all = [list(x) for x in generated_all]

        return sequences

    return custom_loop


def run_generate_with_hf_custom_loop(
    *,
    model: Any,
    tokenizer: Any,
    prompt_ids: torch.Tensor,
    attention_backend: Any,
    prefill_chunk_size: int,
    stop_token_ids: Sequence[int],
    max_new_tokens: int,
    lockstep_stop_mode: str,
    runtime_trace: Optional[HFGenerateTrace] = None,
) -> torch.Tensor:
    ensure_hf_custom_generate_available(model)
    trace = runtime_trace if runtime_trace is not None else HFGenerateTrace()
    custom_loop = build_hf_custom_generate_loop(
        attention_backend=attention_backend,
        prefill_chunk_size=int(prefill_chunk_size),
        extra_stop_token_ids=list(stop_token_ids),
        lockstep_stop_mode=str(lockstep_stop_mode),
        runtime_trace=trace,
    )

    maybe_cuda_sync(prompt_ids.device)
    t0 = time.perf_counter()

    tokenizer_pad = getattr(tokenizer, "pad_token_id", None)
    safe_pad_token_id = choose_surrogate_pad_token_id_for_hf_generate(
        model=model,
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        preferred_pad_token_id=(None if tokenizer_pad is None else int(tokenizer_pad)),
        stop_token_ids=stop_token_ids,
    )

    generate_kwargs = dict(
        input_ids=prompt_ids,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        max_new_tokens=int(max_new_tokens),
        pad_token_id=int(safe_pad_token_id),
        return_dict_in_generate=False,
        custom_generate=custom_loop,
    )

    sequences = model.generate(**generate_kwargs)

    maybe_cuda_sync(prompt_ids.device)
    t1 = time.perf_counter()
    trace.generate_api_wall_time_s = float(t1 - t0)
    return sequences


def extract_generation_rows_from_sequences(
    *,
    tokenizer: Any,
    prompt_len: int,
    sequences: torch.Tensor,
    stop_token_ids: Sequence[int],
    skip_special_tokens: bool,
    answer_prefixes: Sequence[str],
    acceptable_outputs: Sequence[Sequence[str]],
    trace: Optional[HFGenerateTrace] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    if sequences.dim() != 2:
        raise ValueError(f"Expected sequences rank 2, got shape {tuple(sequences.shape)}")

    stop_set = {int(x) for x in stop_token_ids}
    generated_all_tensor = sequences[:, int(prompt_len) :]
    generated_all = [[int(x) for x in row] for row in generated_all_tensor.tolist()]

    first_stop_steps = list(trace.first_stop_step) if trace is not None and trace.first_stop_step else [None] * len(generated_all)
    records: List[Dict[str, Any]] = []
    visible_total = 0

    for b_idx, full_ids in enumerate(generated_all):
        stop_step = first_stop_steps[b_idx]
        if stop_step is None:
            for idx, tok in enumerate(full_ids):
                if tok in stop_set:
                    stop_step = idx
                    break

        if stop_step is None:
            visible_ids = list(full_ids)
        else:
            visible_ids = list(full_ids[: int(stop_step)])

        visible_text = tokenizer.decode(visible_ids, skip_special_tokens=skip_special_tokens)
        normalized_text = normalize_prediction_text(visible_text, answer_prefix=answer_prefixes[b_idx])
        is_exact_match = exact_match_any(
            visible_text,
            acceptable_outputs[b_idx],
            answer_prefix=answer_prefixes[b_idx],
        )
        visible_total += len(visible_ids)

        records.append(
            {
                "generated_token_ids_all": list(full_ids),
                "generated_token_ids_visible": list(visible_ids),
                "generated_text": visible_text,
                "generated_text_normalized": normalized_text,
                "stop_step": stop_step,
                "exact_match": bool(is_exact_match),
            }
        )

    return records, int(visible_total)
