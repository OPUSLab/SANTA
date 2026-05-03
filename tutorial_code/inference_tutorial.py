#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Tuple

import torch

from attention_backends import FA2_PAPER_TARGET_VERSION, FLASHINFER_PAPER_TARGET_VERSION, build_backends
from hf_generate_bridge import HFGenerateTrace, extract_generation_rows_from_sequences, run_generate_with_hf_custom_loop
from runtime_common import (
    alloc_nhd_caches_from_prefill,
    dtype_from_str,
    dtype_to_name,
    generate_lockstep_batch,
    get_stop_token_ids,
    infer_attention_dims,
    load_model_and_tokenizer,
    maybe_cuda_sync,
    prefill_in_chunks,
)


DEFAULT_BACKENDS = ["fa2", "santa_flash", "santa_prop"]


def parse_args() -> argparse.Namespace:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Single-prompt runner for FA2, S^2ANTA-Flash, and S^2ANTA-Prop on the shared batched contiguous-KV scaffold."
    )
    parser.add_argument("--model-name", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--prompt-file", default=os.path.join(here, "prompt.txt"))
    parser.add_argument("--output-file", default=os.path.join(here, "inference_tutorial_output.json"))
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "bfloat16", "fp16", "float16"])
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--prefill-chunk-size", type=int, default=1024)
    parser.add_argument("--backends", nargs="+", default=list(DEFAULT_BACKENDS))
    parser.add_argument("--fa2-expected-version", default=FA2_PAPER_TARGET_VERSION)
    parser.add_argument("--fa2-version-policy", choices=["error", "warn", "ignore"], default="warn")
    parser.add_argument(
        "--flashinfer-mode",
        choices=["single_loop", "batch_compact"],
        default="single_loop",
        help="Legacy FlashInfer reference only; not the main batched paper baseline.",
    )
    parser.add_argument("--flashinfer-expected-version", default=FLASHINFER_PAPER_TARGET_VERSION)
    parser.add_argument("--flashinfer-version-policy", choices=["error", "warn", "ignore"], default="warn")
    parser.add_argument("--flashinfer-jit", choices=["auto", "allow", "disable"], default="auto")
    parser.add_argument("--flashinfer-preload-libstdcpp", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--no-flashinfer-tensor-cores", action="store_true")
    parser.add_argument("--backend-smoke-test-len", type=int, default=16)
    parser.set_defaults(continue_on_backend_error=True)
    parser.add_argument("--continue-on-backend-error", dest="continue_on_backend_error", action="store_true")
    parser.add_argument("--fail-on-backend-error", dest="continue_on_backend_error", action="store_false")
    parser.add_argument("--santa-s", type=int, default=2048)
    parser.add_argument("--santa-seed", type=int, default=1690)
    parser.add_argument("--santa-block-n", type=int, default=None)
    parser.add_argument("--lockstep-stop-mode", choices=["fixed", "all_finished"], default="all_finished")
    parser.add_argument(
        "--generation-surface",
        choices=["hf_generate", "manual"],
        default="hf_generate",
        help="Default uses the official HF generate(custom_generate=...) hook while preserving the existing decode hot path.",
    )
    parser.add_argument("--extra-stop-token-strings", nargs="*", default=["<|eot_id|>", "<|end_of_text|>"])
    parser.set_defaults(skip_special_tokens=True)
    parser.add_argument("--no-skip-special-tokens", dest="skip_special_tokens", action="store_false")
    return parser.parse_args()



def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()



def smoke_test_backends(
    backends: List[Any],
    *,
    model: Any,
    device: torch.device,
    dtype: torch.dtype,
    valid_len: int,
    continue_on_error: bool,
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    dims = infer_attention_dims(model)
    active: List[Any] = []
    failures: List[Dict[str, Any]] = []
    for backend in backends:
        try:
            record = backend.smoke_test(
                device=device,
                dtype=dtype,
                num_heads=int(dims["num_heads"]),
                num_kv_heads=int(dims["num_kv_heads"]),
                head_dim=int(dims["head_dim"]),
                valid_len=valid_len,
            )
            print(
                f"[backend ready] {backend.name}: batch_size={record['batch_size']} valid_len={record['valid_len']} mode={record.get('actual_mode', 'n/a')}"
            )
            active.append(backend)
        except Exception as exc:
            failure = {
                "backend": backend.name,
                "stage": "smoke_test",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "details": getattr(exc, "details", None),
                **{f"backend_{k}": v for k, v in backend.info().items()},
            }
            failures.append(failure)
            print(f"[backend failed] {backend.name}: {type(exc).__name__}: {exc}")
            details = getattr(exc, "details", None)
            if details:
                print(details)
            if not continue_on_error:
                raise
    return active, failures



def run_one_backend(
    *,
    backend: Any,
    model: Any,
    tokenizer: Any,
    prompt_ids: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
    stop_token_ids: List[int],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if args.generation_surface == "manual":
        maybe_cuda_sync(device)
        t0 = time.perf_counter()
        prefill_logits_last, past_list = prefill_in_chunks(model, prompt_ids, prefill_chunk_size=args.prefill_chunk_size)
        maybe_cuda_sync(device)
        t1 = time.perf_counter()

        maybe_cuda_sync(device)
        t2 = time.perf_counter()
        caches = alloc_nhd_caches_from_prefill(
            past_list,
            prompt_len=int(prompt_ids.shape[1]),
            total_len=int(prompt_ids.shape[1] + args.max_new_tokens),
            dtype=dtype,
            device=device,
            consume_past=True,
        )
        del past_list
        maybe_cuda_sync(device)
        t3 = time.perf_counter()

        decode_result = generate_lockstep_batch(
            model,
            tokenizer,
            prompt_ids=prompt_ids,
            prefill_logits_last=prefill_logits_last,
            caches=caches,
            attention_backend=backend,
            stop_token_ids=stop_token_ids,
            max_new_tokens=args.max_new_tokens,
            lockstep_stop_mode=args.lockstep_stop_mode,
            skip_special_tokens=args.skip_special_tokens,
            answer_prefixes=[""],
            acceptable_outputs=[[]],
        )
        maybe_cuda_sync(device)

        example = decode_result["examples"][0]
        prefill_time_s = float(t1 - t0)
        cache_setup_time_s = float(t3 - t2)
        decode_time_s = float(decode_result["decode_time_s"])
        generate_api_wall_time_s = float(prefill_time_s + cache_setup_time_s + decode_time_s)
        wall_time_s = generate_api_wall_time_s
        wrapper_overhead_s = 0.0
    elif args.generation_surface == "hf_generate":
        trace = HFGenerateTrace()
        sequences = run_generate_with_hf_custom_loop(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            attention_backend=backend,
            prefill_chunk_size=args.prefill_chunk_size,
            stop_token_ids=stop_token_ids,
            max_new_tokens=args.max_new_tokens,
            lockstep_stop_mode=args.lockstep_stop_mode,
            runtime_trace=trace,
        )
        example_rows, _ = extract_generation_rows_from_sequences(
            tokenizer=tokenizer,
            prompt_len=int(prompt_ids.shape[1]),
            sequences=sequences,
            stop_token_ids=stop_token_ids,
            skip_special_tokens=args.skip_special_tokens,
            answer_prefixes=[""],
            acceptable_outputs=[[]],
            trace=trace,
        )
        example = example_rows[0]
        prefill_time_s = float(trace.prefill_time_s)
        cache_setup_time_s = float(trace.cache_setup_time_s)
        decode_time_s = float(trace.decode_time_s)
        generate_api_wall_time_s = float(trace.generate_api_wall_time_s)
        wall_time_s = float(trace.end_to_end_wall_time_s)
        wrapper_overhead_s = float(wall_time_s - (prefill_time_s + cache_setup_time_s + decode_time_s))
    else:
        raise ValueError(f"Unsupported generation_surface: {args.generation_surface}")

    return {
        "backend": backend.name,
        "status": "ok",
        **{f"backend_{k}": v for k, v in backend.info().items()},
        "prompt_len": int(prompt_ids.shape[1]),
        "dtype": dtype_to_name(dtype),
        "device": str(device),
        "generation_surface": args.generation_surface,
        "prefill_time_s": prefill_time_s,
        "cache_setup_time_s": cache_setup_time_s,
        "decode_time_s": decode_time_s,
        "generate_api_wall_time_s": generate_api_wall_time_s,
        "hf_generate_wrapper_overhead_s": wrapper_overhead_s,
        "end_to_end_wall_time_s": wall_time_s,
        "generated_token_ids_all": example["generated_token_ids_all"],
        "generated_token_ids_visible": example["generated_token_ids_visible"],
        "generated_text": example["generated_text"],
    }



def main() -> None:
    args = parse_args()
    device = torch.device(f"cuda:{args.device_index}" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_str(args.dtype)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if any(str(name).lower() == "flashinfer" for name in args.backends):
        print("[note] backend=flashinfer is kept only as a legacy reference. The main paper-fair batched baseline is backend=fa2.")

    model, tokenizer = load_model_and_tokenizer(args.model_name, dtype, device)
    prompt_text = read_text(args.prompt_file)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    stop_token_ids = get_stop_token_ids(tokenizer, args.extra_stop_token_strings)

    backends, init_errors = build_backends(
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
        skip_init_failures=args.continue_on_backend_error,
    )

    results: List[Dict[str, Any]] = []
    for row in init_errors:
        print(f"[init failed] backend={row['backend']}: {row['error_type']}: {row['error']}")
        details = row.get("details")
        if details:
            print(details)
        results.append({"status": "init_error", **row})

    backends, smoke_failures = smoke_test_backends(
        backends,
        model=model,
        device=device,
        dtype=dtype,
        valid_len=args.backend_smoke_test_len,
        continue_on_error=args.continue_on_backend_error,
    )
    results.extend({"status": "smoke_test_error", **row} for row in smoke_failures)

    if not backends:
        raise RuntimeError("No backends passed initialization + smoke test.")

    for backend in backends:
        try:
            result = run_one_backend(
                backend=backend,
                model=model,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                dtype=dtype,
                device=device,
                stop_token_ids=stop_token_ids,
                args=args,
            )
            print(f"\n=== {backend.name} ===")
            print(result["generated_text"])
            print(
                f"prefill={result['prefill_time_s']:.3f}s cache={result['cache_setup_time_s']:.3f}s "
                f"decode={result['decode_time_s']:.3f}s wall={result['end_to_end_wall_time_s']:.3f}s"
            )
            results.append(result)
        except Exception as exc:
            failure = {
                "backend": backend.name,
                "status": "run_error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "details": getattr(exc, "details", None),
                **{f"backend_{k}": v for k, v in backend.info().items()},
            }
            print(f"[run failed] backend={backend.name}: {type(exc).__name__}: {exc}")
            details = getattr(exc, "details", None)
            if details:
                print(details)
            results.append(failure)
            if not args.continue_on_backend_error:
                raise

    payload = {
        "model_name": args.model_name,
        "prompt_file": args.prompt_file,
        "prompt_len": int(prompt_ids.shape[1]),
        "dtype": dtype_to_name(dtype),
        "device": str(device),
        "backends_requested": list(args.backends),
        "paper_main_backends": list(DEFAULT_BACKENDS),
        "fa2_expected_version": args.fa2_expected_version,
        "fa2_version_policy": args.fa2_version_policy,
        "flashinfer_expected_version": args.flashinfer_expected_version,
        "flashinfer_version_policy": args.flashinfer_version_policy,
        "santa_s": args.santa_s,
        "santa_seed": args.santa_seed,
        "santa_block_n": args.santa_block_n,
        "max_new_tokens": args.max_new_tokens,
        "prefill_chunk_size": args.prefill_chunk_size,
        "lockstep_stop_mode": args.lockstep_stop_mode,
        "generation_surface": args.generation_surface,
        "results": results,
    }
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {args.output_file}")


if __name__ == "__main__":
    main()
