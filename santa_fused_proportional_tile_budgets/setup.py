from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import re

DEFAULT_ARCH_LIST = "8.6;8.9"
ALLOWED = {"8.6", "8.9", "8.6+PTX", "8.9+PTX"}
MODULE_NAME = "santa_prop_batch_cuda"


def _sanitize_arch_list(raw: str) -> str:
    toks = re.split(r"[;\s]+", (raw or "").strip())
    toks = [t for t in toks if t]
    kept = [t for t in toks if t in ALLOWED]
    if not kept:
        return DEFAULT_ARCH_LIST
    order = ["8.6", "8.6+PTX", "8.9", "8.9+PTX"]
    kept_sorted = [t for t in order if t in kept]
    out = []
    for t in kept_sorted:
        if t not in out:
            out.append(t)
    return ";".join(out)


raw_arch = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
os.environ["TORCH_CUDA_ARCH_LIST"] = _sanitize_arch_list(raw_arch)
os.environ.pop("CUDAARCHS", None)

print(f"[setup.py] building {MODULE_NAME}")
print(f"[setup.py] TORCH_CUDA_ARCH_LIST={os.environ['TORCH_CUDA_ARCH_LIST']}")

setup(
    name=MODULE_NAME,
    ext_modules=[
        CUDAExtension(
            name=MODULE_NAME,
            sources=["santa_cuda.cpp", "santa_cuda_kernel.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
