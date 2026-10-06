"""Writes a tiny random Clef joint schema head (released key layout) and the logits the official `JointSchemaHead`
(`joint_schema_model.py` from the Cloudflare/clef-flash release) produces for a synthetic record, for ClefJointHeadParityTests.

Usage: python head_reference.py <release dir holding joint_schema_model.py> <output dir>   (needs torch, safetensors)
"""
import importlib.util, json, sys
from pathlib import Path
import torch
from safetensors.torch import save_file

release, out = Path(sys.argv[1]), Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("jsm", release / "joint_schema_model.py")
jsm = importlib.util.module_from_spec(spec); sys.modules["jsm"] = jsm; spec.loader.exec_module(jsm)

torch.manual_seed(5)
cfg = dict(hidden_size=24, width=16, routing_layers=2, layers=2, heads=4, feedforward=32)
head = jsm.JointSchemaHead(**cfg).eval()
with torch.no_grad():
    for name, p in head.named_parameters():
        p.copy_(torch.randn_like(p) * (0.4 if p.dim() > 1 else 0.2))
    head.prior_logit_scale.fill_(1.2)
    head.joint_logit_scale.fill_(0.7)
    head.residual_gate.fill_(0.4)

vocab, seq = 50, 40
hidden = torch.randn(1, seq, cfg["hidden_size"])
input_ids = torch.randint(0, vocab, (1, seq))
lm_head = torch.randn(vocab, cfg["hidden_size"]) * 0.5
questions = (
    jsm.EncodedQuestion("dept", 1, (10, 14), ((16, 19), (21, 25), (27, 28)), ("a", "b", "c")),
    jsm.EncodedQuestion("outage", 0, (30, 32), ((33, 35), (36, 39)), ("true", "false")),
)
record = jsm.EncodedRecord(input_ids=tuple(input_ids[0].tolist()), questions=questions, record_id="t")
with torch.no_grad():
    logits = head(hidden, input_ids, torch.ones(1, seq, dtype=torch.long), [record], lm_head)[0]

out.mkdir(parents=True, exist_ok=True)
save_file({k: v.contiguous() for k, v in head.state_dict().items()}, str(out / "clef_head_tiny.safetensors"))
save_file({"hidden": hidden[0].contiguous(), "lm_head": lm_head.contiguous(),
           "input_ids": input_ids[0].to(torch.int32).contiguous()}, str(out / "clef_head_inputs.safetensors"))
json.dump({"config": cfg, "seq": seq,
           "questions": [{"id": q.question_id, "type": q.question_type, "question": list(q.question_span),
                          "options": [list(s) for s in q.option_spans], "optionIds": list(q.option_ids)} for q in questions],
           "logits": [l.tolist() for l in logits]}, open(out / "clef_head_expected.json", "w"))
print([l.tolist() for l in logits])
