"""Greedy reference decode of Maxwell-V1.0-Flash on the release's BF16 weights, with the release's own modeling code.

Run in the reference container on a GPU host with about 74 GB of GPU memory or RAM for offload. Not for the Windows venv.
See tools/maxwell-reference/REAL_REFERENCE.md for the files, the image and the pass metric.

    python -I tools/maxwell-reference/dump_real.py --ckpt /models/maxwell-bf16 --prompt-file prompt.txt --new-tokens 40 \
        --out /runs/maxwell-ref --max-memory "0:70GiB,cpu:200GiB"

<ckpt> holds the 35 BF16 shards, model.safetensors.index.json, config.json, configuration_maxwell_moe.py,
modeling_maxwell_moe.py, tokenizer.json, tokenizer_config.json, vocab.json, special_tokens_map.json, chat_template.jinja
and generation_config.json, all at commit 5dd7c6f62165342b5ffc5f0c5b38c2198e82c751.

<prompt-file> holds the user text. It is rendered with the checkpoint's chat template (add_generation_prompt=True) so the
ids match what the engine builds from the same text.

Writes into <out>:
  prompt_ids.json      the prompt token ids (the engine must produce the same count)
  reference_tokens.json  the greedy continuation ids
  reference.tsv        the --scores-out format: index, token_id, logprob, top_id, top_logprob (raw logits, before any processor)
  run.json             versions, device map, timings, peak memory
"""

import argparse
import json
import os
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--new-tokens", type=int, default=40)
    p.add_argument("--out", required=True)
    p.add_argument("--max-memory", default=None, help='e.g. "0:70GiB,cpu:200GiB"; unset lets accelerate choose')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    started = time.time()

    tok = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)
    prompt_text = open(args.prompt_file, encoding="utf-8").read()
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
            key, value = part.split(":")
            max_memory[int(key) if key.isdigit() else key] = value
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="auto", max_memory=max_memory,
    ).eval()
    load_seconds = time.time() - started

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=args.new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            output_logits=True,
            return_dict_in_generate=True,
            eos_token_id=None,
            pad_token_id=tok.pad_token_id,
        )
    decode_seconds = time.time() - t0

    generated = out.sequences[0, len(prompt_ids):].tolist()
    with open(os.path.join(args.out, "reference_tokens.json"), "w") as f:
        json.dump(generated, f)

    rows = ["index\ttoken_id\tlogprob\ttop_id\ttop_logprob"]
    for i, (token_id, logits) in enumerate(zip(generated, out.logits)):
        logp = torch.log_softmax(logits[0].float(), dim=-1)
        top_lp, top_id = logp.max(dim=-1)
        rows.append(f"{i}\t{token_id}\t{logp[token_id].item():.9g}\t{int(top_id)}\t{top_lp.item():.9g}")
    with open(os.path.join(args.out, "reference.tsv"), "w") as f:
        f.write("\n".join(rows) + "\n")

    peak = [torch.cuda.max_memory_allocated(d) for d in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    info = {
        "torch": torch.__version__, "transformers": __import__("transformers").__version__, "cuda": torch.version.cuda,
        "prompt_tokens": len(prompt_ids), "new_tokens": len(generated), "load_seconds": round(load_seconds, 1),
        "decode_seconds_incl_prefill": round(decode_seconds, 1), "peak_cuda_bytes_per_device": peak,
        "device_map_devices": sorted({str(v) for v in getattr(model, "hf_device_map", {}).values()}),
    }
    with open(os.path.join(args.out, "run.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
