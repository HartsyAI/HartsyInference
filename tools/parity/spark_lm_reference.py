"""Upstream Spark-TTS LM reference runs (CPU, F32) for the engine's parity checks.

Loads the LLM exactly as SparkTTS._initialize_inference does (AutoTokenizer / AutoModelForCausalLM on the LLM dir; the
F32 dtype is pinned because upstream's transformers 4.46.2 loads F32 by default) and calls generate the way
SparkTTS.inference does. Two modes:

  greedy: do_sample=False for --steps tokens on one prompt; writes the generated ids, the first-step top logits and
          the global/semantic ids parsed with upstream's regexes to --out (JSON).
  sample: do_sample=True, temperature 0.8 / top_k 50 / top_p 0.95 (SparkTTS.inference defaults), one run per seed,
          max_new_tokens capped by --steps; prints per-seed semantic count, whether <|im_end|> ended it, wall time.

Usage:
  python spark_lm_reference.py --llm <Spark LLM dir with tokenizer.json + config.json + model.safetensors>
      --prompt "<prompt string>" --mode greedy --steps 50 --out ref.json
  python spark_lm_reference.py --llm <dir> --prompt "<prompt>" --mode sample --steps 1500 --seeds 0 1 2
"""

import argparse
import json
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

IM_END = 151645
END_SEMANTIC = 165157


def parse(tokenizer, ids):
    text = tokenizer.decode(ids, skip_special_tokens=True)
    return ([int(t) for t in re.findall(r"bicodec_global_(\d+)", text)],
            [int(t) for t in re.findall(r"bicodec_semantic_(\d+)", text)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--mode", choices=["greedy", "sample"], required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out")
    args = ap.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    model = AutoModelForCausalLM.from_pretrained(args.llm, torch_dtype=torch.float32).eval()
    print(f"model dtype {model.dtype}, attn {model.config._attn_implementation}")
    inputs = tokenizer([args.prompt], return_tensors="pt")
    n_prompt = inputs.input_ids.shape[1]

    with torch.no_grad():
        if args.mode == "greedy":
            first = model(**inputs).logits[0, -1].float()
            top = torch.topk(first, args.top)
            out = model.generate(**inputs, max_new_tokens=args.steps, do_sample=False)
            ids = out[0, n_prompt:].tolist()
            glob, sem = parse(tokenizer, ids)
            result = {"prompt_ids": inputs.input_ids[0].tolist(), "greedy_ids": ids,
                      "first_top_ids": top.indices.tolist(), "first_top_logits": top.values.tolist(),
                      "global": glob, "semantic": sem, "torch": torch.__version__}
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(result, f)
            print(f"greedy {len(ids)} ids, {len(glob)} global + {len(sem)} semantic; first top1 {top.indices[0].item()}")
            return

        for seed in args.seeds:
            torch.manual_seed(seed)
            t0 = time.time()
            out = model.generate(**inputs, max_new_tokens=args.steps, do_sample=True, top_k=50, top_p=0.95,
                                 temperature=0.8)
            ids = out[0, n_prompt:].tolist()
            glob, sem = parse(tokenizer, ids)
            print(json.dumps({"seed": seed, "generated": len(ids), "global": len(glob), "semantic": len(sem),
                              "im_end": IM_END in ids,
                              "end_semantic_at": ids.index(END_SEMANTIC) if END_SEMANTIC in ids else None,
                              "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
