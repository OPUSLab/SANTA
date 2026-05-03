from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _normalize_mode(value: str, *, default: str = "auto") -> str:
    v = str(value or default).strip().lower()
    if v not in {"auto", "allow", "disable", "on", "off"}:
        return default
    return v


def _find_conda_runtime() -> Dict[str, Optional[str]]:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return {
            "conda_prefix": None,
            "conda_lib_dir": None,
            "libstdcxx": None,
            "libgcc_s": None,
        }
    lib_dir = Path(conda_prefix) / "lib"
    libstdcxx = lib_dir / "libstdc++.so.6"
    libgcc_s = lib_dir / "libgcc_s.so.1"
    return {
        "conda_prefix": str(conda_prefix),
        "conda_lib_dir": str(lib_dir) if lib_dir.exists() else None,
        "libstdcxx": str(libstdcxx) if libstdcxx.exists() else None,
        "libgcc_s": str(libgcc_s) if libgcc_s.exists() else None,
    }


def _split_env_list(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [x for x in value.split(":") if x]


def _prepend_env_list(value: Optional[str], *entries: Optional[str]) -> str:
    existing = _split_env_list(value)
    new_items = [x for x in entries if x]
    out: list[str] = []
    for item in new_items + existing:
        if item and item not in out:
            out.append(item)
    return ":".join(out)


def _needs_reexec(preload_libstdcpp: str, runtime: Dict[str, Any]) -> bool:
    mode = str(preload_libstdcpp or "auto").strip().lower()
    if mode == "off":
        return False
    libstdcxx = runtime.get("libstdcxx")
    conda_lib_dir = runtime.get("conda_lib_dir")
    if not libstdcxx or not conda_lib_dir:
        return False
    if os.environ.get("_SANTA_FLASHINFER_ENV_REEXEC_DONE") == "1":
        return False

    ld_library_path = _split_env_list(os.environ.get("LD_LIBRARY_PATH"))
    ld_preload = _split_env_list(os.environ.get("LD_PRELOAD"))
    need_ld_path = conda_lib_dir not in ld_library_path
    need_preload = libstdcxx not in ld_preload
    if mode == "on":
        return need_ld_path or need_preload

    return need_ld_path or need_preload


def _maybe_reexec_for_conda_runtime(preload_libstdcpp: str, runtime: Dict[str, Any]) -> None:
    if not _needs_reexec(preload_libstdcpp, runtime):
        return

    env = dict(os.environ)
    conda_lib_dir = runtime.get("conda_lib_dir")
    libstdcxx = runtime.get("libstdcxx")
    libgcc_s = runtime.get("libgcc_s")
    env["LD_LIBRARY_PATH"] = _prepend_env_list(env.get("LD_LIBRARY_PATH"), conda_lib_dir)
    env["LD_PRELOAD"] = _prepend_env_list(env.get("LD_PRELOAD"), libstdcxx, libgcc_s)
    env["_SANTA_FLASHINFER_ENV_REEXEC_DONE"] = "1"
    os.execvpe(sys.executable, [sys.executable] + sys.argv, env)


def configure_flashinfer_runtime(
    *,
    jit_mode: str = "auto",
    preload_libstdcpp: str = "auto",
) -> Dict[str, Any]:
    runtime = _find_conda_runtime()
    runtime.update(
        {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "jit_mode": _normalize_mode(jit_mode, default="auto"),
            "preload_libstdcpp": _normalize_mode(preload_libstdcpp, default="auto"),
            "flashinfer_disable_jit_before": os.environ.get("FLASHINFER_DISABLE_JIT"),
            "ld_library_path_before": os.environ.get("LD_LIBRARY_PATH", ""),
            "ld_preload_before": os.environ.get("LD_PRELOAD", ""),
            "reexec_done": os.environ.get("_SANTA_FLASHINFER_ENV_REEXEC_DONE") == "1",
        }
    )

    if runtime["jit_mode"] == "disable":
        os.environ["FLASHINFER_DISABLE_JIT"] = "1"
    elif runtime["jit_mode"] in {"allow", "auto"}:
        if os.environ.get("FLASHINFER_DISABLE_JIT") == "1":
            os.environ.pop("FLASHINFER_DISABLE_JIT", None)

    runtime["flashinfer_disable_jit_after"] = os.environ.get("FLASHINFER_DISABLE_JIT")
    _maybe_reexec_for_conda_runtime(str(runtime["preload_libstdcpp"]), runtime)

    runtime["ld_library_path_after"] = os.environ.get("LD_LIBRARY_PATH", "")
    runtime["ld_preload_after"] = os.environ.get("LD_PRELOAD", "")
    runtime["reexec_done_after"] = os.environ.get("_SANTA_FLASHINFER_ENV_REEXEC_DONE") == "1"
    return runtime


def compact_runtime_report(runtime: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "python",
        "platform",
        "conda_prefix",
        "conda_lib_dir",
        "libstdcxx",
        "libgcc_s",
        "jit_mode",
        "preload_libstdcpp",
        "flashinfer_disable_jit_after",
        "reexec_done_after",
    ]
    return {k: runtime.get(k) for k in keys}


def format_flashinfer_exception(exc: BaseException, runtime: Dict[str, Any]) -> str:
    msg = str(exc)
    lines = [
        "FlashInfer runtime diagnostics:",
        f"  exception_type: {type(exc).__name__}",
        f"  exception: {msg}",
        "",
        "Runtime summary:",
    ]
    for k, v in compact_runtime_report(runtime).items():
        lines.append(f"  {k}: {v}")

    lower_msg = msg.lower()
    if "glibcxx_" in lower_msg or "libstdc++.so.6" in lower_msg:
        lines.extend(
            [
                "",
                "Detected a libstdc++ / GLIBCXX loader issue while FlashInfer tried to load a JIT-built module.",
                "This usually means the process is picking up an older system libstdc++.so.6 instead of the conda env copy.",
                "",
                "Suggested fixes:",
                "  1) Re-run with the conda runtime first on the loader path:",
                '       export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"',
                '       export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"',
                "",
                "  2) Or rely on the auto-reexec path in this repo by leaving",
                "       --flashinfer-preload-libstdcpp auto",
                "",
                "  3) If the env still disagrees with the pinned paper stack, use a clean dedicated env",
                "     for this benchmark instead of a vLLM-oriented env.",
            ]
        )

    return "\n".join(lines)
