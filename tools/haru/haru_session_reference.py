"""Upstream Haru duplex session (PyTorch, float32, CPU) on the tiny checkpoint, with every random draw recorded.

    python haru_session_reference.py <upstream haru/ package dir> <fixture dir holding haru_tiny.safetensors and config.json>

Runs `haru.session.DuplexSession` greedily (temperature 0) three times: a card and a voice reference, an empty prompt, and a model that
speaks first with an EOP bias. `torch.randn` is replaced by a seeded numpy generator that records each draw, so the C# session can replay
the draws in the same order. The text codec is a byte tokenizer (the card's UTF-8 bytes as ids; id 7 decodes to U+FFFD, any other id to
chr(33 + id % 90)). Writes haru_session_expected.safetensors (in.*, run{A,B,C}.*) and haru_session_expected.json (texts) into the fixture dir.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

package, fixture = sys.argv[1:3]
sys.path.insert(0, package)
fixture = Path(fixture)

from haru.config import HaruConfig  # noqa: E402
from haru.models.haru import HaruModel  # noqa: E402
from haru.models.vibevoice_codec import CodecStream  # noqa: E402
import haru.session as session_module  # noqa: E402
from haru.audio import prepare_reference  # noqa: E402

torch.set_num_threads(1)
config = HaruConfig.from_dict(json.loads((fixture / "config.json").read_text()))
model = HaruModel(config).eval().requires_grad_(False)
model.load_state_dict(load_file(str(fixture / "haru_tiny.safetensors")))


class _Encoded:
    def __init__(self, ids):
        self.ids = ids


class ByteTokenizer:
    def encode(self, text, add_special_tokens=False):
        return _Encoded(list(text.encode("utf-8")))

    def decode(self, ids, skip_special_tokens=False):
        return "".join("�" if i == 7 else chr(33 + i % 90) for i in ids)


model.tokenizer = ByteTokenizer()
session_module.open_stream = lambda codec, **kwargs: CodecStream(codec)

rng = np.random.default_rng(11)
draws: list[np.ndarray] = []
real_randn = torch.randn


def fake_randn(*size, generator=None, device=None, dtype=None, **kwargs):
    shape = tuple(size[0]) if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)) else tuple(size)
    values = rng.standard_normal(shape).astype(np.float32)
    draws.append(values.reshape(-1).copy())
    return torch.as_tensor(values, device=device, dtype=dtype)


torch.randn = fake_randn

FRAME = config.frame_samples
user = (0.1 * rng.standard_normal(6 * FRAME)).astype(np.float32)
reference = (0.1 * rng.standard_normal(5 * FRAME + 700)).astype(np.float32)
tensors: dict[str, torch.Tensor] = {"in.user": torch.from_numpy(user), "in.reference": torch.from_numpy(reference)}
texts: dict[str, list[str]] = {}


def run(name: str, card: str, ref, **options) -> None:
    draws.clear()
    cfg = session_module.DuplexConfig(temperature=0.0, seed=1, **options)
    session = session_module.DuplexSession(model, system_prompt=card, reference=ref, config=cfg)
    frames = [session.start()]
    for i in range(6):
        frames.append(session.step(user[i * FRAME : (i + 1) * FRAME]))
    tensors[f"{name}.tokens"] = torch.tensor([f.token for f in frames], dtype=torch.int32)
    tensors[f"{name}.audio"] = torch.from_numpy(np.stack([f.audio for f in frames]))
    tensors[f"{name}.prompt_length"] = torch.tensor(session.prompt_length, dtype=torch.int32)
    for i, values in enumerate(draws):
        tensors[f"{name}.draw.{i}"] = torch.from_numpy(values)
    texts[name] = [f.text for f in frames]


run("runA", "You are Haru.", reference)
run("runB", "", None)
run("runC", "Haru speaks first.", reference, speak_first=True, eop_bias=2.0)

# Reference preparation: silence trimmed at -45 dBFS with a margin, last seconds kept.
clip = np.concatenate([np.zeros(9000, np.float32), (0.2 * rng.standard_normal(30000)).astype(np.float32), np.zeros(11000, np.float32)])
prepared = prepare_reference(clip, max_seconds=1.0)
tensors["prep.in"] = torch.from_numpy(clip)
tensors["prep.out"] = torch.from_numpy(np.ascontiguousarray(prepared))
tensors["prep.full"] = torch.from_numpy(np.ascontiguousarray(prepare_reference(clip)))

torch.randn = real_randn
save_file({k: v.contiguous() for k, v in tensors.items()}, str(fixture / "haru_session_expected.safetensors"), metadata={"format": "pt"})
(fixture / "haru_session_expected.json").write_text(json.dumps(texts, ensure_ascii=False, indent=1))
print("wrote", fixture, {k: v.tolist() for k, v in tensors.items() if k.endswith(".tokens")}, {k: texts[k][:3] for k in texts})
