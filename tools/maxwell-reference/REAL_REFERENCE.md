# Maxwell-V1.0-Flash: real-weight reference comparison

Goal: the engine's greedy continuation and per-position logprobs on the FP8 checkpoint against the release's own code on the BF16
checkpoint. Lets the catalog row leave `ValidationPending`.

## Reference files (BF16 repo `maxwell-model/Maxwell-V1-Flash`, commit `5dd7c6f62165342b5ffc5f0c5b38c2198e82c751`, Apache-2.0)

Sizes from the HF tree API on 2026-11-26. No LFS sha256 is listed in the non-shard files; the shards carry LFS sha256 in the tree API (`lfs.oid`).

| File | Bytes | Note |
|---|---:|---|
| `model-00001.safetensors` … `model-00034.safetensors` | 2,147,888,692 to 2,148,365,332 each (34 files) | BF16 shards |
| `model-00035.safetensors` | 593,357,142 | last shard, sha256 `cfea0bf381feb9f8c5e2a0cb1e45d123b7dfa2fc83b13a5eeafc73f08836d2ae` |
| `model.safetensors.index.json` | 3,031,212 | weight map |
| `config.json` | 2,123 | |
| `generation_config.json` | 133 | |
| `configuration_maxwell_moe.py` | 9,491 | release code, required by `trust_remote_code` |
| `modeling_maxwell_moe.py` | 93,907 | release code, required by `trust_remote_code` |
| `tokenizer.json` | 9,664,766 | |
| `tokenizer_config.json` | 2,593 | |
| `vocab.json` | 4,947,414 | |
| `special_tokens_map.json` | 181 | |
| `chat_template.jinja` | 7,754 | |
| `LICENSE` | 11,357 | |

Total shards: 35 files, 73,625,607,468 bytes (73.6 GB). Download only with the user's yes; the GPU host has about 2.1 TB free per the peer.

Also needed for the engine side (already in the catalog): FP8 repo `maxwell-model/Maxwell-V1-Flash-FP8` at `6f91969de0d39c9fdbb98306e8826ba4f661125b`, `model.safetensors` 38,588,885,068 bytes, sha256 `3b8b158c1a0656bda3642904a0c5040da7d901fec6167ebff26bf41d69c741ff`.

## Container and versions

The tiny-config reference ran in the CPU reference image with transformers 4.46.3 (PARITY_VERIFICATION.md). The real-weight image is not built yet. Proposed, to confirm before the run:

- Base: `nvidia/cuda:12.4.1-runtime-ubuntu22.04` with Python 3.10 or 3.11.
- `pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124`
- `pip install transformers==4.46.3 accelerate==1.1.1 safetensors numpy`

The release modeling code is loaded with `trust_remote_code=True` from the checkpoint directory; do not import it from anywhere else. Record `torch`, `transformers` and `cuda` in `run.json`.

Memory: BF16 needs about 74 GB of GPU memory, or RAM for the offloaded part (`device_map="auto"`). A 24 GB card cannot hold it; use an 80 GB card or a host with at least 150 GB RAM for offload.

## Invocation

```bash
# 1. Reference: BF16 greedy, 40 tokens, raw logits per step
python -I tools/maxwell-reference/dump_real.py --ckpt /models/maxwell-bf16 --prompt-file /runs/prompt.txt \
    --new-tokens 40 --out /runs/maxwell-ref --max-memory "0:70GiB,cpu:200GiB"

# 2. Engine greedy run (FP8, 4090 or equivalent): same prompt text, same 40 tokens
hartsy text "$(cat /runs/prompt.txt)" -m maxwell-v1-flash -b cuda --device cuda:0 --max-tokens 40 --temperature 0 \
    --scores-out /runs/engine/greedy.tsv --set paths.modelsRoot="/models" --set paths.sideModelAutofetch=false -q

# 3. Engine teacher-forced run: the engine scores the reference tokens, from /runs/maxwell-ref/reference.tsv
hartsy text "$(cat /runs/prompt.txt)" -m maxwell-v1-flash -b cuda --device cuda:0 --max-tokens 40 --temperature 0 \
    --teacher-force /runs/maxwell-ref/reference.tsv --scores-out /runs/engine/forced.tsv \
    --set paths.modelsRoot="/models" --set paths.sideModelAutofetch=false -q

# 4. Compare
python -I tools/maxwell-reference/compare_real.py --ref /runs/maxwell-ref/reference.tsv \
    --engine-greedy /runs/engine/greedy.tsv --engine-forced /runs/engine/forced.tsv --tol-logprob 0.10
```

Reference decode (dump_real.py): an explicit greedy loop over the release's forward, not `model.generate()`. `generation_config.json` (eos ids 2 and 200001, no sampling) therefore never affects the tokens. The release computes lm_head in F32 (`modeling_maxwell_moe.py`: `F.linear(lm_input.float(), lm_head.weight.float())`), so the reference logits are F32; the script refuses to run otherwise. A stop id in the continuation is an error: choose a prompt that runs the full 40 tokens, because the engine stops there. The stop ids are read from the checkpoint files (`eos_token_id` in `generation_config.json` and `config.json`), not hard-coded. Every forward passes `cache_position` and `position_ids` explicitly (prefill 0..L-1, then one position per step), so positions do not depend on the cache object.

Comparison order: greedy tokens are compared over the common prefix before any length check, so a divergence is reported as a divergence, and a length difference after a matching prefix is reported as a length failure.

Checks before trusting the result:
- `prompt_ids.json` length (reference) equals the engine's prompt token count in its log. This is a length check only: the engine does not dump its prompt ids, so equal lengths do not prove equal ids. A rendering difference with the same length would pass this check.
- The engine run must not use the thinking toggle unless the reference also does. `dump_real.py` renders without `enable_thinking`.
- `--teacher-force` needs `--scores-out` (the CLI enforces it). Greedy means `--temperature 0`.

Unverified: `--teacher-force` and `--scores-out` reach the Maxwell path through the shared generation pipeline (`ForcedTokenIds`, `RecordTokenScores`), but no run has checked that yet.

## Pass metric

- Greedy tokens: the engine's 40 greedy ids equal the reference's at every position.
- Argmax: with the reference tokens teacher-forced, argmax equals the reference argmax at every position.
- Logprob: `|engine − reference|` of the reference token is at most 0.10 nats at every position.

`compare_real.py` exits 0 only when all three hold. On failure it names the first failing position and prints the reference's top-1 minus top-2 logprob margin there (from the `second_id` / `second_logprob` columns that dump_real.py writes after the five engine columns). A margin near 0 means a near-tie; a large margin with a flip means a real divergence.

The 0.10-nat tolerance and the argmax rule are proposals. The FP8-vs-BF16 noise floor for this model has not been measured. If a run fails by a small margin, bisect before changing the tolerance: compare per-layer hidden states (not in this package; the engine has no Maxwell layer dump yet).

Not covered here: a prompt longer than 4,096 tokens (exercises the sparse indexer; needs a second reference run), and per-layer cosine as in `tools/ernie5-reference/`.
