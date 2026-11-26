"""Greedy reference decode of Maxwell-V1.0-Flash on the release's BF16 weights, with the release's own modeling code.

Run in the reference container on a GPU host with about 74 GB of GPU memory or RAM for offload. Not for the Windows venv.
See tools/maxwell-reference/REAL_REFERENCE.md for the files, the image and the pass metric.

    python -I tools/maxwell-reference/dump_real.py --ckpt /models/maxwell-bf16 --prompt-file prompt.txt --new-tokens 40 \
        --out /runs/maxwell-ref --max-memory "0:70GiB,cpu:200GiB"

<ckpt> holds the 35 BF16 shards, model.safetensors.index.json, config.json, configuration_maxwell_moe.py,
modeling_maxwell_moe.py, tokenizer.json, tokenizer_config.json, vocab.json, special_tokens_map.json, chat_template.jinja
and generation_config.json, all at commit 5dd7c6f62165342b5ffc5f0c5b38c2198e82c751.

Decoding is an explicit greedy loop, not model.generate(): generation_config.json (eos ids 2 and 200001) and any processor it
implies never run. The release's forward computes lm_head in F32 (modeling_maxwell_moe.py: F.linear(lm_input.float(),
lm_head.weight.float())), so the logits are F32 by construction; the script asserts that.

A stop id in the generated tokens is an error: the engine stops there, so the two sides would have different lengths.

Writes into <out>:
  prompt_ids.json        the prompt token ids (the engine must produce the same count)
  reference_tokens.json  the greedy continuation ids
  reference.tsv          index, token_id, logprob, top_id, top_logprob, second_id, second_logprob (raw F32 log-softmax)
                         The first five columns are the engine's --scores-out format; the last two are the runner-up.
  run.json               versions, revision, file hashes, device map, timings, peak memory
"""

import argparse
import hashlib
import json
import os
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

CHECKPOINT_COMMIT = "5dd7c6f62165342b5ffc5f0c5b38c2198e82c751"
STOP_IDS = {2, 200001}  # generation_config.json eos_token_id


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--new-tokens", type=int, default=40)
    p.add_argument("--out", required=True)
    p.add_argument("--max-memory", default=None, help='e.g. "0:70GiB,cpu:200GiB"; unset lets accelerate choose')
    return p.parse_args()


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def greedy(model, prompt_ids, new_tokens, device):
    """Explicit greedy decode. Returns the generated ids and per-step rows of (token, logp, top_id, top_lp, second_id, second_lp)."""
    rows = []
    past = None
    cur = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        for step in range(new_tokens):
            out = model(input_ids=cur, past_key_values=past, use_cache=True, return_dict=True)
            past = out.past_key_values
            logits = out.logits[0, -1]
            if logits.dtype != torch.float32:
                raise RuntimeError(f"logits are {logits.dtype}, not float32; the release's lm_head F32 path did not run")
            logp = torch.log_softmax(logits, dim=-1)
            top_lp, top_id = torch.topk(logp, 2)
            token = int(top_id[0])
            if token != int(torch.argmax(logits)):
                raise RuntimeError(f"step {step}: greedy token differs from argmax of the raw logits")
            if token in STOP_IDS:
                raise RuntimeError(f"step {step}: the reference emitted stop id {token}; choose a prompt that runs {new_tokens} tokens")
            rows.append((token, float(logp[token]), int(top_id[0]), float(top_lp[0]), int(top_id[1]), float(top_lp[1])))
            cur = torch.tensor([[token]], dtype=torch.long, device=device)
    return [r[0] for r in rows], rows


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    started = time.time()

    tok = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)
    with open(args.prompt_file, encoding="utf-8") as f:
        prompt_text = f.read()
    messages = [{"role": "user", "content": prompt_text}]
    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tok(rendered, add_special_tokens=False)["input_ids"]
    with open(os.path.join(args.out, "prompt_ids.json"), "w") as f:
        json.dump(prompt_ids, f)

    config = AutoConfig.from_pretrained(args.ckpt, trust_remote_code=True)
    config._attn_implementation = "eager"
    max_memory = None
    if args.max_memory:
        max_memory = {}
        for part in args.max_memory.split(","):
            key, value = part.split(":", 1)
            max_memory[int(key) if key.isdigit() else key] = value
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="auto", max_memory=max_memory,
    ).eval()
    load_seconds = time.time() - started

    device = next(model.parameters()).device
    t0 = time.time()
    generated, rows = greedy(model, prompt_ids, args.new_tokens, device)
    decode_seconds = time.time() - t0

    with open(os.path.join(args.out, "reference_tokens.json"), "w") as f:
        json.dump(generated, f)
    header = "index\ttoken_id\tlogprob\ttop_id\ttop_logprob\tsecond_id\tsecond_logprob"
    lines = [header]
    for i, (token, lp, top_id, top_lp, second_id, second_lp) in enumerate(rows):
        lines.append(f"{i}\t{token}\t{lp:.9g}\t{top_id}\t{top_lp:.9g}\t{second_id}\t{second_lp:.9g}")
    with open(os.path.join(args.out, "reference.tsv"), "w") as f:
        f.write("\n".join(lines) + "\n")

    cuda = torch.cuda.is_available()
    info = {
        "checkpoint_commit": CHECKPOINT_COMMIT,
        "config_json_sha256": sha256_of(os.path.join(args.ckpt, "config.json")),
        "generation_config_json_sha256": sha256_of(os.path.join(args.ckpt, "generation_config.json")),
        "modeling_sha256": sha256_of(os.path.join(args.ckpt, "modeling_maxwell_moe.py")),
        "torch": torch.__version__, "transformers": __import__("transformers").__version__, "cuda": torch.version.cuda,
        "decode": "explicit greedy loop; generate() not used", "stop_ids": sorted(STOP_IDS),
        "prompt_tokens": len(prompt_ids), "new_tokens": len(generated),
        "load_seconds": round(load_seconds, 1), "decode_seconds_incl_prefill": round(decode_seconds, 1),
        "peak_cuda_bytes_per_device": [torch.cuda.max_memory_allocated(d) for d in range(torch.cuda.device_count())] if cuda else [],
        "device_map_devices": sorted({str(v) for v in getattr(model, "hf_device_map", {}).values()}),
    }
    with open(os.path.join(args.out, "run.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
