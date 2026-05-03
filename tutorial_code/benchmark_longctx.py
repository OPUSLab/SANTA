#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from attention_backends import FA2_PAPER_TARGET_VERSION, FLASHINFER_PAPER_TARGET_VERSION, build_backends
from hf_generate_bridge import HFGenerateTrace, extract_generation_rows_from_sequences, run_generate_with_hf_custom_loop
from runtime_common import (
    PreparedBatch,
    PreparedExample,
    alloc_nhd_caches_from_prefill,
    dtype_from_str,
    dtype_to_name,
    ensure_dir,
    format_optional_int,
    generate_lockstep_batch,
    get_stop_token_ids,
    infer_attention_dims,
    load_model_and_tokenizer,
    maybe_cuda_sync,
    prefill_in_chunks,
    summarize_numeric,
    write_json,
    write_jsonl,
)


MAIN_PAPER_BACKENDS = ["fa2", "santa_flash", "santa_prop"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batched contiguous-KV long-context benchmark through HF generate(custom_generate=...): FA2 exact dense decode vs S^2ANTA-Flash and S^2ANTA-Prop"
    )
    parser.add_argument("--model-name", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset", required=True, help="Path to benchmark JSONL; see data/README.md for schema")
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--batch-sizes", nargs="*", type=int, default=None)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timed-runs", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--target-prompt-token-length", type=int, default=32768)
    parser.add_argument("--prompt-length-mode", choices=["truncate", "pad", "exact"], default="truncate")
    parser.add_argument("--truncation-side", choices=["left", "right"], default="left")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "bfloat16", "fp16", "float16"])
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--prefill-chunk-size", type=int, default=1024)
    parser.add_argument("--backends", nargs="+", default=list(MAIN_PAPER_BACKENDS))
    parser.add_argument("--fa2-expected-version", default=FA2_PAPER_TARGET_VERSION)
    parser.add_argument("--fa2-version-policy", choices=["error", "warn", "ignore"], default="warn")
    parser.add_argument(
        "--flashinfer-mode",
        choices=["single_loop", "batch_compact"],
        default="single_loop",
        help="Legacy FlashInfer reference only; not the main paper-fair batched baseline.",
    )
    parser.add_argument("--flashinfer-expected-version", default=FLASHINFER_PAPER_TARGET_VERSION)
    parser.add_argument("--flashinfer-version-policy", choices=["error", "warn", "ignore"], default="warn")
    parser.add_argument("--flashinfer-jit", choices=["auto", "allow", "disable"], default="auto")
    parser.add_argument("--flashinfer-preload-libstdcpp", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--no-flashinfer-tensor-cores", action="store_true")
    parser.add_argument("--santa-s", type=int, default=2048)
    parser.add_argument("--santa-seed", type=int, default=1690)
    parser.add_argument("--santa-block-n", type=int, default=None)
    parser.add_argument("--lockstep-stop-mode", choices=["fixed", "all_finished"], default="fixed")
    parser.add_argument(
        "--generation-surface",
        choices=["hf_generate", "manual"],
        default="hf_generate",
        help="Default uses the official HF generate(custom_generate=...) hook while preserving the existing decode hot path.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_outputs"),
    )
    parser.add_argument("--quick-mode", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.set_defaults(skip_special_tokens=True)
    parser.add_argument("--no-skip-special-tokens", dest="skip_special_tokens", action="store_false")
    parser.add_argument(
        "--extra-stop-token-strings",
        nargs="*",
        default=["<|eot_id|>", "<|end_of_text|>"],
    )
    return parser.parse_args()



def apply_quick_mode(args: argparse.Namespace) -> None:
    if not args.quick_mode:
        return
    args.warmup_runs = 0
    args.timed_runs = 1
    args.max_new_tokens = min(int(args.max_new_tokens), 32)
    if args.num_examples is None:
        bs = args.batch_sizes[0] if args.batch_sizes else args.batch_size
        args.num_examples = max(int(bs), min(8, int(bs) * 2))



def _canonical_outputs_field(outputs_value: Any) -> List[str]:
    if outputs_value is None:
        return []
    if isinstance(outputs_value, list):
        return [str(x) for x in outputs_value]
    return [str(outputs_value)]



def prepare_examples(
    dataset_path: str,
    tokenizer: Any,
    *,
    num_examples: Optional[int],
    target_prompt_token_length: int,
    prompt_length_mode: str,
    truncation_side: str,
) -> Tuple[List[PreparedExample], Dict[str, Any]]:
    if prompt_length_mode == "pad":
        raise NotImplementedError(
            "pad mode is intentionally not implemented for this paper benchmark runtime. "
            "The batched contiguous-KV path is optimized for true uniform-length prompts, so use --prompt-length-mode truncate or exact."
        )

    if target_prompt_token_length <= 0:
        raise ValueError(f"target_prompt_token_length must be > 0, got {target_prompt_token_length}")

    examples: List[PreparedExample] = []
    total_rows = 0
    accepted = 0
    skipped_short = 0
    skipped_length_mismatch = 0

    with open(dataset_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            total_rows += 1
            row = json.loads(line)

            input_text = str(row.get("input", ""))
            if not input_text:
                continue

            token_ids = tokenizer(input_text, add_special_tokens=False)["input_ids"]
            original_len = int(len(token_ids))

            if prompt_length_mode == "truncate":
                if original_len < target_prompt_token_length:
                    skipped_short += 1
                    continue
                if truncation_side == "left":
                    token_ids = token_ids[-target_prompt_token_length:]
                else:
                    token_ids = token_ids[:target_prompt_token_length]
            elif prompt_length_mode == "exact":
                if original_len != target_prompt_token_length:
                    skipped_length_mismatch += 1
                    continue
            else:
                raise ValueError(f"Unsupported prompt_length_mode: {prompt_length_mode}")

            used_len = int(len(token_ids))
            outputs = _canonical_outputs_field(row.get("outputs", []))
            answer_prefix = str(row.get("answer_prefix", ""))
            index = int(row.get("index", len(examples)))

            examples.append(
                PreparedExample(
                    index=index,
                    input_text=input_text,
                    outputs=outputs,
                    answer_prefix=answer_prefix,
                    original_prompt_len=original_len,
                    used_prompt_len=used_len,
                    prompt_token_ids=[int(x) for x in token_ids],
                    raw_record=row,
                )
            )
            accepted += 1

            if num_examples is not None and accepted >= int(num_examples):
                break

    summary = {
        "dataset_path": dataset_path,
        "total_rows_seen": total_rows,
        "accepted_examples": accepted,
        "skipped_short_for_target": skipped_short,
        "skipped_length_mismatch": skipped_length_mismatch,
        "target_prompt_token_length": int(target_prompt_token_length),
        "prompt_length_mode": prompt_length_mode,
        "truncation_side": truncation_side,
    }
    return examples, summary



def make_batches(examples: Sequence[PreparedExample], batch_size: int) -> Tuple[List[PreparedBatch], int]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    usable = (len(examples) // batch_size) * batch_size
    dropped = len(examples) - usable
    batches: List[PreparedBatch] = []
    for batch_id, start in enumerate(range(0, usable, batch_size)):
        chunk = list(examples[start : start + batch_size])
        if not chunk:
            continue
        prompt_len = int(chunk[0].used_prompt_len)
        for ex in chunk:
            if int(ex.used_prompt_len) != prompt_len:
                raise RuntimeError("All examples in a batch must have the same used prompt length.")
        batches.append(PreparedBatch(batch_id=batch_id, prompt_len=prompt_len, examples=chunk))
    return batches, dropped



def serialize_example_for_manifest(ex: PreparedExample) -> Dict[str, Any]:
    row = dict(ex.raw_record)
    row["benchmark_original_prompt_len"] = int(ex.original_prompt_len)
    row["benchmark_used_prompt_len"] = int(ex.used_prompt_len)
    return row



def flatten_backend_info(prefix: str, info: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in info.items():
        out[f"{prefix}{k}"] = v
    return out



def run_backend_on_batch(
    *,
    model: Any,
    tokenizer: Any,
    batch: PreparedBatch,
    backend: Any,
    dtype: torch.dtype,
    device: torch.device,
    max_new_tokens: int,
    prefill_chunk_size: int,
    stop_token_ids: Sequence[int],
    lockstep_stop_mode: str,
    skip_special_tokens: bool,
    generation_surface: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    prompt_ids = batch.to_tensor(device)
    total_len = int(batch.prompt_len + max_new_tokens)

    if generation_surface == "manual":
        maybe_cuda_sync(device)
        t_prefill0 = time.perf_counter()
        prefill_logits_last, past_list = prefill_in_chunks(
            model,
            prompt_ids,
            prefill_chunk_size=prefill_chunk_size,
        )
        maybe_cuda_sync(device)
        t_prefill1 = time.perf_counter()

        maybe_cuda_sync(device)
        t_cache0 = time.perf_counter()
        caches = alloc_nhd_caches_from_prefill(
            past_list,
            prompt_len=batch.prompt_len,
            total_len=total_len,
            dtype=dtype,
            device=device,
            consume_past=True,
        )
        del past_list
        maybe_cuda_sync(device)
        t_cache1 = time.perf_counter()

        decode_result = generate_lockstep_batch(
            model,
            tokenizer,
            prompt_ids=prompt_ids,
            prefill_logits_last=prefill_logits_last,
            caches=caches,
            attention_backend=backend,
            stop_token_ids=stop_token_ids,
            max_new_tokens=max_new_tokens,
            lockstep_stop_mode=lockstep_stop_mode,
            skip_special_tokens=skip_special_tokens,
            answer_prefixes=[ex.answer_prefix for ex in batch.examples],
            acceptable_outputs=[ex.outputs for ex in batch.examples],
        )
        del prefill_logits_last, caches

        prefill_time_s = float(t_prefill1 - t_prefill0)
        cache_setup_time_s = float(t_cache1 - t_cache0)
        decode_time_s = float(decode_result["decode_time_s"])
        generate_api_wall_time_s = float(prefill_time_s + cache_setup_time_s + decode_time_s)
        wall_time_s = generate_api_wall_time_s
        timed_generated_tokens = int(decode_result["timed_generated_tokens"])
        visible_generated_tokens = int(decode_result["visible_generated_tokens"])
        generation_core_rows = list(decode_result["examples"])
    elif generation_surface == "hf_generate":
        trace = HFGenerateTrace()
        sequences = run_generate_with_hf_custom_loop(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            attention_backend=backend,
            prefill_chunk_size=prefill_chunk_size,
            stop_token_ids=stop_token_ids,
            max_new_tokens=max_new_tokens,
            lockstep_stop_mode=lockstep_stop_mode,
            runtime_trace=trace,
        )
        generation_core_rows, visible_generated_tokens = extract_generation_rows_from_sequences(
            tokenizer=tokenizer,
            prompt_len=int(batch.prompt_len),
            sequences=sequences,
            stop_token_ids=stop_token_ids,
            skip_special_tokens=skip_special_tokens,
            answer_prefixes=[ex.answer_prefix for ex in batch.examples],
            acceptable_outputs=[ex.outputs for ex in batch.examples],
            trace=trace,
        )
        prefill_time_s = float(trace.prefill_time_s)
        cache_setup_time_s = float(trace.cache_setup_time_s)
        decode_time_s = float(trace.decode_time_s)
        generate_api_wall_time_s = float(trace.generate_api_wall_time_s)
        wall_time_s = float(trace.end_to_end_wall_time_s)
        timed_generated_tokens = int(trace.timed_generated_tokens)
        del sequences
    else:
        raise ValueError(f"Unsupported generation_surface: {generation_surface}")

    del prompt_ids

    batch_size = int(batch.batch_size)
    wrapper_overhead_s = float(wall_time_s - (prefill_time_s + cache_setup_time_s + decode_time_s))

    batch_metric = {
        "batch_id": int(batch.batch_id),
        "batch_size": batch_size,
        "example_indices": list(batch.example_indices),
        "prompt_len": int(batch.prompt_len),
        "prefill_time_s": prefill_time_s,
        "cache_setup_time_s": cache_setup_time_s,
        "decode_time_s": decode_time_s,
        "generate_api_wall_time_s": float(generate_api_wall_time_s),
        "hf_generate_wrapper_overhead_s": wrapper_overhead_s,
        "end_to_end_wall_time_s": wall_time_s,
        "timed_generated_tokens": int(timed_generated_tokens),
        "visible_generated_tokens": int(visible_generated_tokens),
        "steps_sampled": int(timed_generated_tokens // batch_size) if batch_size > 0 else 0,
        "tpot_ms": (1000.0 * decode_time_s / timed_generated_tokens) if timed_generated_tokens > 0 else float("nan"),
        "output_tokens_per_s": (timed_generated_tokens / decode_time_s) if decode_time_s > 0 else float("nan"),
        "visible_output_tokens_per_s": (visible_generated_tokens / decode_time_s) if decode_time_s > 0 else float("nan"),
        "requests_per_s": (batch_size / wall_time_s) if wall_time_s > 0 else float("nan"),
        "prefill_tokens_per_s": (batch_size * batch.prompt_len / prefill_time_s) if prefill_time_s > 0 else float("nan"),
    }

    generation_rows: List[Dict[str, Any]] = []
    for batch_slot, (ex, gen) in enumerate(zip(batch.examples, generation_core_rows)):
        generation_rows.append(
            {
                "batch_id": int(batch.batch_id),
                "batch_slot": int(batch_slot),
                "batch_size": batch_size,
                "example_index": int(ex.index),
                "prompt_len": int(ex.used_prompt_len),
                "original_prompt_len": int(ex.original_prompt_len),
                "outputs": list(ex.outputs),
                "answer_prefix": ex.answer_prefix,
                "generated_token_ids_all": gen["generated_token_ids_all"],
                "generated_token_ids_visible": gen["generated_token_ids_visible"],
                "generated_text": gen["generated_text"],
                "generated_text_normalized": gen["generated_text_normalized"],
                "stop_step": gen["stop_step"],
                "exact_match": bool(gen["exact_match"]),
            }
        )

    return batch_metric, generation_rows



def aggregate_run_rows(batch_rows: Sequence[Dict[str, Any]], generation_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["num_batches"] = int(len(batch_rows))
    out["num_requests"] = int(sum(int(r["batch_size"]) for r in batch_rows))
    out["prefill_time_s"] = float(sum(float(r["prefill_time_s"]) for r in batch_rows))
    out["cache_setup_time_s"] = float(sum(float(r["cache_setup_time_s"]) for r in batch_rows))
    out["decode_time_s"] = float(sum(float(r["decode_time_s"]) for r in batch_rows))
    out["end_to_end_wall_time_s"] = float(sum(float(r["end_to_end_wall_time_s"]) for r in batch_rows))
    out["timed_generated_tokens"] = int(sum(int(r["timed_generated_tokens"]) for r in batch_rows))
    out["visible_generated_tokens"] = int(sum(int(r["visible_generated_tokens"]) for r in batch_rows))
    out["prompt_tokens_total"] = int(sum(int(r["batch_size"]) * int(r["prompt_len"]) for r in batch_rows))

    decode_time = out["decode_time_s"]
    wall_time = out["end_to_end_wall_time_s"]
    prefill_time = out["prefill_time_s"]
    timed_tokens = out["timed_generated_tokens"]
    visible_tokens = out["visible_generated_tokens"]
    num_requests = out["num_requests"]

    out["tpot_ms"] = (1000.0 * decode_time / timed_tokens) if timed_tokens > 0 else float("nan")
    out["output_tokens_per_s"] = (timed_tokens / decode_time) if decode_time > 0 else float("nan")
    out["visible_output_tokens_per_s"] = (visible_tokens / decode_time) if decode_time > 0 else float("nan")
    out["requests_per_s"] = (num_requests / wall_time) if wall_time > 0 else float("nan")
    out["prefill_tokens_per_s"] = (out["prompt_tokens_total"] / prefill_time) if prefill_time > 0 else float("nan")
    out["exact_match_rate"] = (
        sum(1 for row in generation_rows if bool(row["exact_match"])) / len(generation_rows)
        if generation_rows
        else float("nan")
    )
    return out



def compute_pairwise_agreement(generation_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
    backends = sorted({str(r["backend"]) for r in generation_rows})
    for row in generation_rows:
        key = (int(row["run_idx"]), int(row["example_index"]), str(row["backend"]))
        by_key[key] = row

    pair_rows: List[Dict[str, Any]] = []
    for i in range(len(backends)):
        for j in range(i + 1, len(backends)):
            a = backends[i]
            b = backends[j]
            comparable = 0
            norm_match = 0
            token_match = 0
            both_em = 0
            a_only_em = 0
            b_only_em = 0
            run_ids = sorted({int(r["run_idx"]) for r in generation_rows})
            example_ids = sorted({int(r["example_index"]) for r in generation_rows})
            for run_idx in run_ids:
                for example_index in example_ids:
                    row_a = by_key.get((run_idx, example_index, a))
                    row_b = by_key.get((run_idx, example_index, b))
                    if row_a is None or row_b is None:
                        continue
                    comparable += 1
                    if row_a["generated_text_normalized"] == row_b["generated_text_normalized"]:
                        norm_match += 1
                    if row_a["generated_token_ids_visible"] == row_b["generated_token_ids_visible"]:
                        token_match += 1
                    a_em = bool(row_a["exact_match"])
                    b_em = bool(row_b["exact_match"])
                    if a_em and b_em:
                        both_em += 1
                    elif a_em and not b_em:
                        a_only_em += 1
                    elif b_em and not a_em:
                        b_only_em += 1
            if comparable == 0:
                continue
            pair_rows.append(
                {
                    "backend_a": a,
                    "backend_b": b,
                    "comparable_examples": comparable,
                    "normalized_text_match_rate": norm_match / comparable,
                    "visible_token_id_match_rate": token_match / comparable,
                    "both_exact_match_rate": both_em / comparable,
                    "a_only_exact_match_rate": a_only_em / comparable,
                    "b_only_exact_match_rate": b_only_em / comparable,
                }
            )
    return pair_rows



def save_aggregate_csv(path: str, aggregate_payload: Dict[str, Any]) -> None:
    fieldnames = [
        "backend",
        "runs",
        "prefill_time_s_mean",
        "decode_time_s_mean",
        "end_to_end_wall_time_s_mean",
        "tpot_ms_mean",
        "output_tokens_per_s_mean",
        "requests_per_s_mean",
        "exact_match_rate_mean",
        "prefill_time_s_median",
        "decode_time_s_median",
        "end_to_end_wall_time_s_median",
        "tpot_ms_median",
        "output_tokens_per_s_median",
        "requests_per_s_median",
        "exact_match_rate_median",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for backend, payload in aggregate_payload.items():
            metrics = payload["metrics"]
            writer.writerow(
                {
                    "backend": backend,
                    "runs": payload["num_runs"],
                    "prefill_time_s_mean": metrics["prefill_time_s"]["mean"],
                    "decode_time_s_mean": metrics["decode_time_s"]["mean"],
                    "end_to_end_wall_time_s_mean": metrics["end_to_end_wall_time_s"]["mean"],
                    "tpot_ms_mean": metrics["tpot_ms"]["mean"],
                    "output_tokens_per_s_mean": metrics["output_tokens_per_s"]["mean"],
                    "requests_per_s_mean": metrics["requests_per_s"]["mean"],
                    "exact_match_rate_mean": metrics["exact_match_rate"]["mean"],
                    "prefill_time_s_median": metrics["prefill_time_s"]["median"],
                    "decode_time_s_median": metrics["decode_time_s"]["median"],
                    "end_to_end_wall_time_s_median": metrics["end_to_end_wall_time_s"]["median"],
                    "tpot_ms_median": metrics["tpot_ms"]["median"],
                    "output_tokens_per_s_median": metrics["output_tokens_per_s"]["median"],
                    "requests_per_s_median": metrics["requests_per_s"]["median"],
                    "exact_match_rate_median": metrics["exact_match_rate"]["median"],
                }
            )



def benchmark_for_batch_size(
    *,
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> None:
    out_dir = ensure_dir(os.path.join(args.output_dir, f"bs{batch_size}"))

    examples, dataset_summary = prepare_examples(
        args.dataset,
        tokenizer,
        num_examples=args.num_examples,
        target_prompt_token_length=args.target_prompt_token_length,
        prompt_length_mode=args.prompt_length_mode,
        truncation_side=args.truncation_side,
    )
    batches, dropped_tail = make_batches(examples, batch_size)
    if not batches:
        raise RuntimeError(
            f"No full batches available for batch_size={batch_size}. "
            f"Accepted examples={len(examples)} after filtering."
        )

    write_json(
        os.path.join(out_dir, "dataset_summary.json"),
        {**dataset_summary, "dropped_tail_examples": int(dropped_tail)},
    )
    write_jsonl(
        os.path.join(out_dir, "examples_used.jsonl"),
        [serialize_example_for_manifest(ex) for ex in examples[: len(batches) * batch_size]],
    )

    backends = build_backends(
        args.backends,
        fa2_expected_version=args.fa2_expected_version,
        fa2_version_policy=args.fa2_version_policy,
        santa_s=args.santa_s,
        santa_seed=args.santa_seed,
        santa_block_n=args.santa_block_n,
        flashinfer_mode=args.flashinfer_mode,
        flashinfer_use_tensor_cores=(not args.no_flashinfer_tensor_cores),
        flashinfer_expected_version=args.flashinfer_expected_version,
        flashinfer_version_policy=args.flashinfer_version_policy,
        flashinfer_jit_mode=args.flashinfer_jit,
        flashinfer_preload_libstdcpp=args.flashinfer_preload_libstdcpp,
    )

    stop_token_ids = get_stop_token_ids(tokenizer, args.extra_stop_token_strings)
    batch_metrics_rows: List[Dict[str, Any]] = []
    generation_rows: List[Dict[str, Any]] = []
    run_metrics_rows: List[Dict[str, Any]] = []

    def phase_runs(phase: str, num_runs: int) -> None:
        for run_idx in range(num_runs):
            for backend in backends:
                per_backend_batch_rows: List[Dict[str, Any]] = []
                per_backend_generation_rows: List[Dict[str, Any]] = []
                for batch in batches:
                    batch_metric, batch_generation_rows = run_backend_on_batch(
                        model=model,
                        tokenizer=tokenizer,
                        batch=batch,
                        backend=backend,
                        dtype=dtype,
                        device=device,
                        max_new_tokens=args.max_new_tokens,
                        prefill_chunk_size=args.prefill_chunk_size,
                        stop_token_ids=stop_token_ids,
                        lockstep_stop_mode=args.lockstep_stop_mode,
                        skip_special_tokens=args.skip_special_tokens,
                        generation_surface=args.generation_surface,
                    )
                    backend_info = backend.info()
                    batch_row = {
                        "phase": phase,
                        "run_idx": int(run_idx),
                        "backend": backend.name,
                        **flatten_backend_info("backend_", backend_info),
                        **batch_metric,
                    }
                    per_backend_batch_rows.append(batch_row)
                    batch_metrics_rows.append(batch_row)

                    for row in batch_generation_rows:
                        full_row = {
                            "phase": phase,
                            "run_idx": int(run_idx),
                            "backend": backend.name,
                            **flatten_backend_info("backend_", backend_info),
                            **row,
                        }
                        per_backend_generation_rows.append(full_row)
                        generation_rows.append(full_row)

                    print(
                        f"[{phase} run {run_idx}] backend={backend.name:<12} batch={batch.batch_id:03d} "
                        f"prefill={batch_row['prefill_time_s']:.3f}s cache={batch_row['cache_setup_time_s']:.3f}s "
                        f"decode={batch_row['decode_time_s']:.3f}s wall={batch_row['end_to_end_wall_time_s']:.3f}s "
                        f"TPOT={batch_row['tpot_ms']:.3f} ms"
                    )

                run_summary = aggregate_run_rows(per_backend_batch_rows, per_backend_generation_rows)
                run_summary.update(
                    {
                        "phase": phase,
                        "run_idx": int(run_idx),
                        "backend": backend.name,
                        **flatten_backend_info("backend_", backend.info()),
                    }
                )
                run_metrics_rows.append(run_summary)
                print(
                    f"[{phase} run {run_idx}] backend={backend.name:<12} total_wall={run_summary['end_to_end_wall_time_s']:.3f}s "
                    f"decode={run_summary['decode_time_s']:.3f}s output_tok/s={run_summary['output_tokens_per_s']:.2f} "
                    f"req/s={run_summary['requests_per_s']:.2f} EM={run_summary['exact_match_rate']:.4f}"
                )

    phase_runs("warmup", args.warmup_runs)
    phase_runs("timed", args.timed_runs)

    write_jsonl(os.path.join(out_dir, "batch_metrics.jsonl"), batch_metrics_rows)
    write_jsonl(os.path.join(out_dir, "run_metrics.jsonl"), run_metrics_rows)
    write_jsonl(os.path.join(out_dir, "generations.jsonl"), generation_rows)

    timed_run_rows = [row for row in run_metrics_rows if row["phase"] == "timed"]
    aggregate_payload: Dict[str, Any] = {}
    metrics_to_summarize = [
        "prefill_time_s",
        "cache_setup_time_s",
        "decode_time_s",
        "end_to_end_wall_time_s",
        "tpot_ms",
        "output_tokens_per_s",
        "visible_output_tokens_per_s",
        "requests_per_s",
        "prefill_tokens_per_s",
        "exact_match_rate",
    ]
    by_backend: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in timed_run_rows:
        by_backend[str(row["backend"])].append(row)

    for backend, rows in by_backend.items():
        aggregate_payload[backend] = {
            "num_runs": len(rows),
            "metrics": {metric: summarize_numeric([float(r[metric]) for r in rows]) for metric in metrics_to_summarize},
            "last_backend_info": {k: v for k, v in rows[-1].items() if k.startswith("backend_")},
        }

    write_json(os.path.join(out_dir, "aggregates.json"), aggregate_payload)
    save_aggregate_csv(os.path.join(out_dir, "aggregates.csv"), aggregate_payload)

    pairwise_agreement = compute_pairwise_agreement([row for row in generation_rows if row["phase"] == "timed"])
    write_json(os.path.join(out_dir, "backend_agreement.json"), pairwise_agreement)

    batch_config = {
        "model_name": args.model_name,
        "dataset": args.dataset,
        "num_examples_requested": args.num_examples,
        "batch_size": batch_size,
        "warmup_runs": args.warmup_runs,
        "timed_runs": args.timed_runs,
        "max_new_tokens": args.max_new_tokens,
        "target_prompt_token_length": args.target_prompt_token_length,
        "prompt_length_mode": args.prompt_length_mode,
        "truncation_side": args.truncation_side,
        "dtype": dtype_to_name(dtype),
        "device": str(device),
        "prefill_chunk_size": args.prefill_chunk_size,
        "backends": list(args.backends),
        "paper_main_backends": list(MAIN_PAPER_BACKENDS),
        "fa2_expected_version": args.fa2_expected_version,
        "fa2_version_policy": args.fa2_version_policy,
        "flashinfer_mode": args.flashinfer_mode,
        "flashinfer_use_tensor_cores": (not args.no_flashinfer_tensor_cores),
        "flashinfer_expected_version": args.flashinfer_expected_version,
        "flashinfer_version_policy": args.flashinfer_version_policy,
        "santa_s": args.santa_s,
        "santa_seed": args.santa_seed,
        "santa_block_n": args.santa_block_n,
        "lockstep_stop_mode": args.lockstep_stop_mode,
        "generation_surface": args.generation_surface,
        "extra_stop_token_strings": list(args.extra_stop_token_strings),
        "stop_token_ids": stop_token_ids,
        "attention_dims": infer_attention_dims(model),
    }
    write_json(os.path.join(out_dir, "config.json"), batch_config)



def main() -> None:
    args = parse_args()
    apply_quick_mode(args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device(f"cuda:{args.device_index}" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_str(args.dtype)

    ensure_dir(args.output_dir)
    write_json(
        os.path.join(args.output_dir, "invocation.json"),
        {
            "argv": vars(args),
            "device": str(device),
            "dtype": dtype_to_name(dtype),
            "santa_block_n": format_optional_int(args.santa_block_n),
            "paper_main_backends": list(MAIN_PAPER_BACKENDS),
            "timestamp_unix": time.time(),
        },
    )

    if any(str(name).lower() == "flashinfer" for name in args.backends):
        print("[note] backend=flashinfer is kept only as a legacy reference. The main paper-fair batched baseline is backend=fa2.")

    model, tokenizer = load_model_and_tokenizer(args.model_name, dtype, device)
    batch_sizes = args.batch_sizes if args.batch_sizes else [args.batch_size]
    for batch_size in batch_sizes:
        benchmark_for_batch_size(
            args=args,
            model=model,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            batch_size=int(batch_size),
        )


if __name__ == "__main__":
    main()
