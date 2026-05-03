#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare saved generations across two backends.")
    p.add_argument("--generations", required=True, help="Path to generations.jsonl from benchmark_longctx.py")
    p.add_argument("--backend-a", required=True)
    p.add_argument("--backend-b", required=True)
    p.add_argument("--phase", choices=["warmup", "timed"], default=None)
    p.add_argument("--run-idx", type=int, default=None)
    return p.parse_args()



def normalize_text(x: Any) -> str:
    return str(x or "").strip()



def load_rows(path: str, *, phase: Optional[str], run_idx: Optional[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if phase is not None and row.get("phase") != phase:
                continue
            row_run_idx = row.get("run_idx", row.get("run_index"))
            if run_idx is not None and row_run_idx != run_idx:
                continue
            rows.append(row)
    return rows



def key_for_row(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        row.get("phase", row.get("run_kind")),
        row.get("run_idx", row.get("run_index")),
        row.get("batch_id"),
        row.get("example_index"),
    )



def main() -> None:
    args = parse_args()
    rows = load_rows(args.generations, phase=args.phase, run_idx=args.run_idx)

    by_backend: Dict[str, Dict[Tuple[Any, ...], Dict[str, Any]]] = {
        args.backend_a: {},
        args.backend_b: {},
    }
    for row in rows:
        backend = str(row.get("backend"))
        if backend in by_backend:
            by_backend[backend][key_for_row(row)] = row

    keys = sorted(set(by_backend[args.backend_a].keys()) & set(by_backend[args.backend_b].keys()))
    if not keys:
        raise RuntimeError(
            f"No overlapping rows found between backend_a={args.backend_a} and backend_b={args.backend_b}."
        )

    total = 0
    exact_text_match = 0
    exact_visible_id_match = 0
    exact_all_id_match = 0
    em_a = 0
    em_b = 0
    differing_examples: List[Dict[str, Any]] = []

    for key in keys:
        ra = by_backend[args.backend_a][key]
        rb = by_backend[args.backend_b][key]
        total += 1

        ta = normalize_text(ra.get("generated_text"))
        tb = normalize_text(rb.get("generated_text"))
        if ta == tb:
            exact_text_match += 1

        ida = list(ra.get("generated_token_ids_visible", []))
        idb = list(rb.get("generated_token_ids_visible", []))
        if ida == idb:
            exact_visible_id_match += 1

        idaa = list(ra.get("generated_token_ids_all", []))
        idba = list(rb.get("generated_token_ids_all", []))
        if idaa == idba:
            exact_all_id_match += 1

        if bool(ra.get("exact_match")):
            em_a += 1
        if bool(rb.get("exact_match")):
            em_b += 1

        if ta != tb:
            differing_examples.append(
                {
                    "key": list(key),
                    "example_index": ra.get("example_index"),
                    "batch_id": ra.get("batch_id"),
                    "text_a": ta,
                    "text_b": tb,
                    "exact_match_a": bool(ra.get("exact_match")),
                    "exact_match_b": bool(rb.get("exact_match")),
                }
            )

    out = {
        "backend_a": args.backend_a,
        "backend_b": args.backend_b,
        "phase": args.phase,
        "run_idx": args.run_idx,
        "overlapping_examples": total,
        "text_match_count": exact_text_match,
        "text_match_rate": exact_text_match / total,
        "visible_token_id_match_count": exact_visible_id_match,
        "visible_token_id_match_rate": exact_visible_id_match / total,
        "all_token_id_match_count": exact_all_id_match,
        "all_token_id_match_rate": exact_all_id_match / total,
        "backend_a_exact_match_count": em_a,
        "backend_a_exact_match_rate": em_a / total,
        "backend_b_exact_match_count": em_b,
        "backend_b_exact_match_rate": em_b / total,
        "num_differing_text_examples": len(differing_examples),
        "sample_differences": differing_examples[:20],
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
