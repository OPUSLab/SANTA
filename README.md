# S²ANTA CUDA decoding kernels

This repository contains the CUDA/PyTorch extension kernels and single-GPU benchmark harness for the S²ANTA long-context decoding experiments.

The repository includes two S²ANTA decode variants:

- `santa_flash`: S²ANTA-Flash, a fixed tile-budget decode kernel.
- `santa_prop`: S²ANTA-Prop, a proportional tile-budget decode kernel.

The benchmark harness compares these kernels against `fa2`, an exact dense FlashAttention-2 decode backend using `flash_attn_with_kvcache`. The included JSONL files are synthetic examples for demonstrating the expected data format and running quick functional checks; they are not the paper evaluation data.

## Repository layout

```text
flash_style_fused_santa_fixed_tile_budgets/
  setup.py
  santa_cuda.cpp
  santa_cuda_kernel.cu

santa_fused_proportional_tile_budgets/
  setup.py
  santa_cuda.cpp
  santa_cuda_kernel.cu

tutorial_code/
  attention_backends.py       # backend adapters: fa2, santa_flash, santa_prop, sdpa, optional flashinfer
  benchmark_longctx.py        # batched long-context benchmark
  compare_outputs.py          # pairwise comparison of saved generations
  env_utils.py                # optional FlashInfer runtime helpers
  hf_generate_bridge.py       # Hugging Face generate(custom_generate=...) bridge
  inference_tutorial.py       # single-prompt generation example
  prompt.txt                  # example prompt for the generation script
  runtime_common.py           # shared model/cache/decode runtime

data/
  README.md
  sample_qa2.jsonl            # small synthetic benchmark example
  sample_qa2_32k.jsonl        # optional synthetic long-context example
```

## Requirements

The reported experiments and example commands target NVIDIA RTX 6000 Ada GPUs, i.e. SM 8.9, with Python 3.10, CUDA-enabled PyTorch, FlashAttention-2, and the two S²ANTA CUDA extension modules.

Tested software stack:

```text
GPU: NVIDIA RTX 6000 Ada Generation, SM 8.9
Python: 3.10
PyTorch: 2.10.0+cu128
Transformers: 4.57.0
FlashAttention: 2.7.4.post1
FlashInfer: 0.6.6, optional legacy/reference backend only
GCC/G++: 13.x
```

## Install

Create an environment and install CUDA-enabled PyTorch before installing the Python package dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install CUDA-enabled torch separately.
# Tested command:
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0+cu128

python -m pip install -r requirements.txt
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
```

Optional legacy/reference backend:

```bash
python -m pip install flashinfer-python==0.6.6
```

`flashinfer` is not required for the S²ANTA-Flash/S²ANTA-Prop comparisons.

## Build the S²ANTA CUDA extensions

The checked-in S²ANTA-Prop build script targets SM 8.6 / 8.9. The example commands below use SM 8.9, matching RTX 6000 Ada. Other GPU architectures are not claimed for this release unless the extension build scripts are modified and the kernels are revalidated.

```bash
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="8.9"
unset CUDAARCHS

python -m pip install -v --no-build-isolation ./flash_style_fused_santa_fixed_tile_budgets
python -m pip install -v --no-build-isolation ./santa_fused_proportional_tile_budgets
```

The installed extension module names are:

```text
santa_flash_batch_cuda
santa_prop_batch_cuda
```

## Quick single-prompt example

This command runs the model-loading path, prefill/cache setup, exact FA2 decode, S²ANTA-Flash decode, and S²ANTA-Prop decode on a short prompt.

```bash
python tutorial_code/inference_tutorial.py \
  --model-name meta-llama/Meta-Llama-3.1-8B-Instruct \
  --prompt-file tutorial_code/prompt.txt \
  --dtype bf16 \
  --backends fa2 santa_flash santa_prop \
  --santa-s 32 \
  --max-new-tokens 8 \
  --output-file ./outputs/inference_tutorial_output.json
```

The default generation surface uses Hugging Face `generate(custom_generate=...)`. For older Transformers builds or for a lower-level runtime check, pass:

```bash
--generation-surface manual
```

## Synthetic benchmark example

The included `data/sample_qa2.jsonl` file is synthetic. It is provided to make the benchmark path runnable and to document the expected record format. It is not intended to reproduce paper numbers.

```bash
python tutorial_code/benchmark_longctx.py \
  --model-name meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dataset ./data/sample_qa2.jsonl \
  --backends fa2 santa_flash santa_prop \
  --dtype bf16 \
  --quick-mode \
  --batch-size 2 \
  --num-examples 2 \
  --target-prompt-token-length 512 \
  --prompt-length-mode truncate \
  --truncation-side left \
  --max-new-tokens 8 \
  --prefill-chunk-size 512 \
  --santa-s 256 \
  --santa-seed 1690 \
  --lockstep-stop-mode fixed \
  --output-dir ./outputs/quick
```

For a synthetic run that exercises the 32k truncation path, use `data/sample_qa2_32k.jsonl`:

```bash
python tutorial_code/benchmark_longctx.py \
  --model-name meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dataset ./data/sample_qa2_32k.jsonl \
  --backends fa2 santa_flash santa_prop \
  --dtype bf16 \
  --quick-mode \
  --batch-size 2 \
  --num-examples 4 \
  --target-prompt-token-length 32768 \
  --prompt-length-mode truncate \
  --truncation-side left \
  --max-new-tokens 32 \
  --prefill-chunk-size 1024 \
  --santa-s 2048 \
  --santa-seed 1690 \
  --lockstep-stop-mode fixed \
  --output-dir ./outputs/quick_32k
```

## Paper-scale benchmark template

To reproduce paper-scale measurements, provide an evaluation JSONL with the same schema as the synthetic samples. The paper evaluation data is not included in this repository.

```bash
python tutorial_code/benchmark_longctx.py \
  --model-name meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dataset /path/to/evaluation.jsonl \
  --backends fa2 santa_flash santa_prop \
  --dtype bf16 \
  --batch-sizes 1 2 4 \
  --num-examples 24 \
  --warmup-runs 1 \
  --timed-runs 3 \
  --target-prompt-token-length 32768 \
  --prompt-length-mode truncate \
  --truncation-side left \
  --max-new-tokens 256 \
  --prefill-chunk-size 1024 \
  --santa-s 2048 \
  --santa-seed 1690 \
  --lockstep-stop-mode fixed \
  --output-dir ./outputs/benchmark_fa2_santa_flash_santa_prop
```

## Compare saved generations

After a benchmark run, compare any two saved backends:

```bash
python tutorial_code/compare_outputs.py \
  --generations ./outputs/benchmark_fa2_santa_flash_santa_prop/bs4/generations.jsonl \
  --backend-a fa2 \
  --backend-b santa_flash \
  --phase timed

python tutorial_code/compare_outputs.py \
  --generations ./outputs/benchmark_fa2_santa_flash_santa_prop/bs4/generations.jsonl \
  --backend-a fa2 \
  --backend-b santa_prop \
  --phase timed
```

## JSONL schema

Each line is one JSON object:

```json
{
  "index": 0,
  "input": "prompt text passed to the tokenizer/model",
  "outputs": ["acceptable answer"],
  "length": 1234,
  "length_w_model_temp": 1234,
  "answer_prefix": " Answer:"
}
```

`benchmark_longctx.py` tokenizes the `input` field directly and applies `--target-prompt-token-length` according to `--prompt-length-mode`. The `length` and `length_w_model_temp` fields are retained for compatibility with the full evaluation-file schema; they are not used to choose the prompt length.

## Notes and limitations

- The runtime is intentionally narrow: single GPU, contiguous KV cache, fixed-length lockstep batch, greedy decoding, no vLLM, no paged KV, and no serving scheduler.
- The Python runtime shares prefill/cache setup across backends and swaps only the decode attention backend.
- `santa_flash` and `santa_prop` use the same benchmark harness and output format.
- Both S²ANTA extension modules expose batched and scalar decode entry points. The benchmark harness uses the batched path and records the actual backend mode in output metadata.
- `flashinfer` and `sdpa` adapters are retained as optional references; the main comparison commands use `fa2`, `santa_flash`, and `santa_prop`.
- The checked-in S²ANTA-Prop build script targets SM 8.6 / 8.9. The example commands target RTX 6000 Ada, SM 8.9. Other GPU architectures are not claimed for this release unless the setup scripts are modified and the kernels are revalidated.

## License

The software is released under the license in `LICENSE`.
