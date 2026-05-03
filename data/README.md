# Synthetic sample data

These JSONL files are synthetic and are included only so the public repository shows the expected schema and can run a short functional benchmark without shipping the paper/rebuttal validation data.

## Files

- `sample_qa2.jsonl`: use this with `tutorial_code/benchmark_longctx.py` for public smoke tests.
- `sample_fwe.jsonl`: optional schema example for a frequency/counting-style record. It is not needed if the public repo removes the rebuttal/error-analysis scripts.
- `sample_qa2_32k.jsonl`: optional synthetic long-context file intended for commands that keep `--target-prompt-token-length 32768`. This is heavier than `sample_qa2.jsonl` and is not needed for a quick smoke test.

## JSONL schema

Each line is one JSON object with the same fields as the private validation files:

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

The benchmark script tokenizes `input` directly. `length` and `length_w_model_temp` are retained for schema compatibility, but are not used to choose prompt length.

For the included `sample_qa2.jsonl`, use a small truncation target such as `--target-prompt-token-length 512`. The sample is not intended to reproduce paper numbers.

For full-32k smoke tests, use `sample_qa2_32k.jsonl`; for normal CI-style checks, prefer `sample_qa2.jsonl` with `--target-prompt-token-length 512`.
