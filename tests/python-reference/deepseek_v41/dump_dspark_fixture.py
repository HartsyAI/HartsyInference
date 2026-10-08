"""Writes the synthetic DSpark fixture (fixtures/dspark_forward.json): the upstream DSpark draft head (mtp.0-2) on top of the target
weights of model_forward.json, run through the same 11-token prefill and six decode steps as that fixture.

For every decode position the fixture records the draft that upstream's incremental window produces (drafts at positions 11 to 16,
past the 8-token window), plus the target's main_hidden rows for all 17 positions. The C# tests rebuild the window from a fresh seed at
each position and must reproduce those drafts. The mtp weights are random (seeded), so acceptance rates mean nothing here; only the
ids, logits and confidences are compared.

Run with the reference venv: ~/dsv41-ref/.venv/bin/python dump_dspark_fixture.py
"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kernel_ports  # noqa: E402

sys.modules["kernel"] = kernel_ports

TARGET_LAYERS = (3, 4, 5)
BLOCK = 5
NOISE = 31
MARKOV_RANK = 8
DRAFT_EXPERTS = 4
DRAFT_TOPK = 2
MTP_LAYERS = 3


def flat(t: torch.Tensor):
    return t.detach().float().contiguous().reshape(-1).tolist()


def init_rules(name: str, prm: torch.Tensor) -> None:
    """The same initialisation rules dump_model_fixture.py applies, by parameter name."""
    leaf = name.rsplit(".", 1)[-1]
    if leaf.endswith("fn") and leaf.startswith("hc_"):
        prm.data.normal_(0, 0.1)
    elif leaf.endswith("scale") and leaf.startswith("hc_"):
        prm.data.copy_(0.5 + 0.1 * torch.randn_like(prm))
    elif leaf.endswith("base") and leaf.startswith("hc_"):
        prm.data.normal_(0, 0.2)
    elif name.endswith("norm.weight"):
        prm.data.copy_(1.0 + 0.1 * torch.randn_like(prm))
    elif leaf == "bias":
        prm.data.normal_(0, 0.1)
    elif leaf == "attn_sink":
        prm.data.normal_(0, 0.5)
    else:
        prm.data.normal_(0, 0.4)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", default=os.path.expanduser("~/dsv41-ref/upstream"))
    p.add_argument("--out", default=os.path.join(HERE, "fixtures", "dspark_forward.json"))
    a = p.parse_args()
    sys.path.insert(0, os.path.join(a.upstream, "inference"))
    import model as mod

    # exact softmax instead of the port's bf16 probabilities, as in dump_model_fixture.py
    mod.sparse_attn = kernel_ports.sparse_attn_exact

    host = json.load(open(os.path.join(HERE, "fixtures", "model_forward.json")))
    cfg = dict(host["config"])
    cfg["compress_ratios"] = tuple(cfg["compress_ratios"]) + (0,) * MTP_LAYERS
    cfg["kv_source_layers"] = tuple(cfg["kv_source_layers"])
    cfg["index_source_layers"] = tuple(cfg["index_source_layers"])
    cfg.update(
        n_mtp_layers=MTP_LAYERS, dspark_block_size=BLOCK, dspark_noise_token_id=NOISE,
        dspark_target_layer_ids=TARGET_LAYERS, dspark_markov_rank=MARKOV_RANK,
        dspark_n_routed_experts=DRAFT_EXPERTS, dspark_n_activated_experts=DRAFT_TOPK,
    )
    args = mod.ModelArgs(**cfg)

    torch.manual_seed(1357)
    model = mod.Transformer(args, None)
    torch.set_default_dtype(torch.float32)
    # target weights come from model_forward.json, so the host fixture and this one describe the same backbone
    target = {name: torch.tensor(values).reshape(host["shapes"][name]) for name, values in host["params"].items()}
    for name, prm in model.named_parameters():
        if name in target:
            assert list(prm.shape) == list(target[name].shape), name
            prm.data = target[name].reshape(prm.shape).float().clone()
        else:
            init_rules(name, prm)
            prm.data = prm.data.float()
    model.eval()
    mtp_params = {n: flat(prm) for n, prm in model.named_parameters() if n.startswith("mtp.")}
    mtp_shapes = {n: list(prm.shape) for n, prm in model.named_parameters() if n.startswith("mtp.")}

    ids = [v for step in host["steps"] for v in step["ids"]]
    assert len(ids) == 17, "the host fixture's 11-token prefill plus six decode steps"
    ids_t = torch.tensor([ids])
    rows, drafts = [], {}
    with torch.inference_mode():
        _, _, main = model(ids_t[:, :11], 0)
        assert model.forward_spec(ids_t[:, 10:11], main, 0) is None
        rows.append(main[0])
        for pos in range(11, 17):
            _, _, main_p = model(ids_t[:, pos : pos + 1], pos)
            rows.append(main_p[0])
            out_ids, out_logits, confidence = model.forward_spec(ids_t[:, pos : pos + 1], main_p, pos)
            drafts[str(pos)] = {
                "ids": [int(v) for v in out_ids[0]],
                "logits": flat(out_logits[0]),
                "confidence": flat(confidence[0]),
            }

    out = {
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
        "target_layers": list(TARGET_LAYERS),
        "block_size": BLOCK,
        "noise_token_id": NOISE,
        "markov_rank": MARKOV_RANK,
        "window_size": cfg["window_size"],
        "ids": ids,
        "main_hidden": flat(torch.cat(rows, dim=0)),
        "main_hidden_width": 3 * cfg["dim"],
        "drafts": drafts,
        "mtp_params": mtp_params,
        "mtp_shapes": mtp_shapes,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
        fh.write("\n")
    print(f"wrote {a.out}: {len(ids)} ids, drafts at {sorted(drafts, key=int)}, mtp tensors {len(mtp_params)}")


if __name__ == "__main__":
    main()
