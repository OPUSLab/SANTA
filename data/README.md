# Synthetic sample data

These JSONL files are synthetic examples for documenting the expected schema and running short functional checks without distributing the paper evaluation data.

## Files

- sample_qa2.jsonl: small synthetic benchmark example.
- sample_qa2_32k.jsonl: optional synthetic long-context example.

## JSONL schema

Each line is one JSON object with the same fields as the full evaluation files:

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

For checking the 32k truncation path, use `sample_qa2_32k.jsonl`. For quick functional checks, prefer `sample_qa2.jsonl` with `--target-prompt-token-length 512`.
