from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="santa_flash_batch_cuda",
    ext_modules=[
        CUDAExtension(
            name="santa_flash_batch_cuda",
            sources=[
                "santa_cuda.cpp",
                "santa_cuda_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-lineinfo",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
