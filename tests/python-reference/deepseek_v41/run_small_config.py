"""Run the UNMODIFIED upstream inference/model.py on CPU at a small config and dump intermediates.

kernel.py needs tilelang; kernel_ports.py stands in for it via a sys.modules shim. Weights are seeded random
(upstream allocates with torch.empty). bf16 dense + no fp4 experts, so fp8_gemm/fp4_gemm are not exercised; the
Engram table is the one fp8 weight and is dequantized in plain torch by upstream.

Output: <out>/manifest.tsv (name, kind, dtype, shape, file) + raw little-endian .bin (f32 for floats, i32/i64 ints).
Usage: python run_small_config.py [--config default|modes] [--seed 0] [--out deepseek_v41_ref/<config>]
"""
import argparse
import os
import sys
from typing import Dict, List, Tuple

import torch

HERE: str = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kernel_ports  # noqa: E402

sys.modules["kernel"] = kernel_ports


def build_args(model_mod, kind: str, engram_rows: Tuple[int, ...] = (), engram_vocab: int = 0):
    common = dict(dtype="bf16", expert_dtype=None, max_batch_size=2, max_seq_len=256, temperature=0.0,
                  dim=256, moe_inter_dim=128, n_heads=8, q_lora_rank=128, head_dim=64, rope_head_dim=16,
                  o_groups=4, o_lora_rank=64, index_n_heads=8, index_head_dim=32, n_routed_experts=8,
                  n_activated_experts=2, route_scale=1.5, swiglu_limit=10.0, hc_mult=4, window_size=16)
    if kind == "default":
        return model_mod.ModelArgs(**common, n_layers=5, n_mtp_layers=1, compress_ratios=(0, 2, 2, 1, 1, 0),
                                   kv_source_layers=(1, 3), index_source_layers=(1, 3), index_topk=8,
                                   dspark_block_size=4, dspark_target_layer_ids=(2, 3, 4), dspark_noise_token_id=7,
                                   dspark_markov_rank=32)
    assert kind == "modes"
    # real mode pattern in miniature: SWA-only, ratio-2 sources, ratio-1 candidate source, reindex + reuse layers
    ratios = (0, 0, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0)
    return model_mod.ModelArgs(**common, n_layers=9, n_mtp_layers=3, compress_ratios=ratios, kv_source_layers=(2, 5),
                               index_source_layers=(2, 5, 7), index_topk=8, candidate_source_layer=5,
                               candidate_topk_blocks=4, candidate_block_size=2, original_seq_len=64,
                               rope_factor=16, compress_rope_theta=160000.0,
                               engram_layer_ids=(1, 4), engram_num_embeddings=engram_rows, engram_max_ngram_size=4,
                               engram_vocab_size=engram_vocab, engram_n_heads=2, engram_head_dim=32,
                               engram_pad_id=2, engram_compressed_vocab_size=99092,
                               dspark_block_size=4, dspark_target_layer_ids=(6, 7, 8), dspark_noise_token_id=7,
                               dspark_markov_rank=32)


def seeded_init(model: torch.nn.Module, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    for name, p in model.named_parameters():
        shape = tuple(p.shape)
        leaf = name.rsplit(".", 1)[-1]
        if p.dtype == torch.float8_e4m3fn:
            data = (torch.randn(shape, generator=g) * 0.5).to(torch.float8_e4m3fn)
        elif p.dtype == torch.float8_e8m0fnu:
            data = (2.0 ** -torch.randint(1, 4, shape, generator=g).to(torch.float32)).to(torch.float8_e8m0fnu)
        elif leaf == "attn_sink":
            data = torch.randn(shape, generator=g)
        elif leaf.startswith("hc_") and leaf.endswith("_fn"):
            data = torch.randn(shape, generator=g) * 0.02
        elif leaf.startswith("hc_") and leaf.endswith("_scale"):
            data = torch.full(shape, 0.5) + 0.1 * torch.randn(shape, generator=g)
        elif leaf.startswith("hc_") and leaf.endswith("_base"):
            data = torch.randn(shape, generator=g) * 0.1
        elif leaf == "bias" or name.endswith("bias_vl"):
            data = torch.randn(shape, generator=g) * 0.1
        elif len(shape) == 1:
            data = 1.0 + 0.1 * torch.randn(shape, generator=g)
        elif "embed" in name and "engram" not in name:
            data = torch.randn(shape, generator=g) * 0.5
        else:
            data = torch.randn(shape, generator=g) * (shape[-1] ** -0.5)
        p.data.copy_(data.to(p.dtype))


class Recorder:
    def __init__(self, out: str) -> None:
        self.out = out
        self.rows: List[str] = []
        self.step: str = "s0"
        os.makedirs(out, exist_ok=True)

    def save(self, name: str, t: torch.Tensor) -> None:
        if t.dtype in (torch.int32, torch.int64):
            arr, kind = t.detach().cpu().contiguous().numpy(), "int"
        elif t.is_floating_point() or t.dtype == torch.bfloat16:
            arr, kind = t.detach().cpu().to(torch.float32).contiguous().numpy(), "float"
        else:
            return
        fname = f"{self.step}.{name}.bin".replace("/", "_")
        arr.tofile(os.path.join(self.out, fname))
        self.rows.append("\t".join([f"{self.step}.{name}", kind, str(arr.dtype), "x".join(map(str, arr.shape)), fname]))

    def hook(self, name: str):
        def fn(_m, _inp, out):
            outs = out if isinstance(out, tuple) else (out,)
            for i, o in enumerate(outs):
                if isinstance(o, torch.Tensor):
                    self.save(name if len(outs) == 1 else f"{name}#{i}", o)
        return fn

    def flush(self) -> None:
        with open(os.path.join(self.out, "manifest.tsv"), "w") as f:
            f.write("\n".join(self.rows) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default=os.path.expanduser("~/dsv41-ref/upstream"))
    ap.add_argument("--config", choices=["default", "modes"], default="default")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefill", type=int, default=24)
    ap.add_argument("--decode", type=int, default=6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(HERE, "deepseek_v41_ref", a.config)
    sys.path.insert(0, os.path.join(a.upstream, "inference"))
    import model as model_mod

    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(a.seed)
    tok = None
    rows: Tuple[int, ...] = ()
    ev = 0
    if a.config == "modes":
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(a.upstream, "tokenizer.json"))
        ev = 997
        # rows = sum of the layer's primes for engram_vocab_size=997, engram_n_heads=2, ngram sizes 2..4
        from engram import find_next_prime
        seen: set = set()
        totals = []
        for _ in range(2):
            t = 0
            for _ in range(3):
                cur = ev - 1
                for _ in range(2):
                    cur = find_next_prime(cur, seen)
                    seen.add(cur)
                    t += cur
            totals.append(t)
        rows = tuple(totals)
    args = build_args(model_mod, a.config, rows, ev)
    model = model_mod.Transformer(args, tok)
    seeded_init(model, a.seed)
    model.eval()

    rec = Recorder(out)
    for name, mod in model.named_modules():
        if name and isinstance(mod, (model_mod.Block, model_mod.Attention, model_mod.Compressor, model_mod.Indexer,
                                     model_mod.MoE, model_mod.Gate, model_mod.Engram, model_mod.ParallelHead,
                                     model_mod.ParallelEmbedding, model_mod.RMSNorm)):
            mod.register_forward_hook(rec.hook(name))
    if model.engram_hash is not None:
        model.engram_hash.register_forward_hook(rec.hook("engram_hash"))

    g = torch.Generator().manual_seed(a.seed + 1)
    total = a.prefill + a.decode
    ids = torch.randint(3, args.vocab_size, (2, total), generator=g)
    rec.step = "prefill"
    rec.save("input_ids", ids[:, : a.prefill])
    out_ids, logits, main_hidden = model(ids[:, : a.prefill])
    rec.save("logits", logits)
    rec.save("output_ids", out_ids)
    if main_hidden is not None:
        rec.save("main_hidden", main_hidden)
        model.forward_spec(out_ids, main_hidden)
    for i in range(a.prefill, total):
        rec.step = f"decode{i - a.prefill}"
        rec.save("input_ids", ids[:, i : i + 1])
        out_ids, logits, main_hidden = model(ids[:, i : i + 1], i)
        rec.save("logits", logits)
        rec.save("output_ids", out_ids)
        if main_hidden is not None:
            rec.save("main_hidden", main_hidden)
            res = model.forward_spec(out_ids, main_hidden, i)
            if res is not None:
                spec_ids, spec_logits, conf = res
                rec.save("spec_ids", spec_ids)
                rec.save("spec_logits", spec_logits)
                rec.save("spec_confidence", conf)
    rec.flush()
    print(f"{a.config}: {len(rec.rows)} tensors -> {out}")


if __name__ == "__main__":
    main()
