"""Records Hugging Face beam search on a toy model for the C# BeamSearch tests.

The toy model's next-token logits are a fixed function of the sequence so far (`toy_logits`, mirrored in
BeamSearchTests). HF transformers 4.52.1 (IndexTTS's pin) runs `generate(num_beams=..., do_sample=False)` with the
settings below; the generated tokens are written per case.

    tts-ref/idx25-venv/bin/python tools/parity/beam_search_reference.py \
        tests/HartsyInference.Audio.Tests/Fixtures/beam_search_reference.json
"""
import json
import sys
from importlib.metadata import version

import torch
from torch import nn
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

VOCAB = 40
EOS = 39
PROMPT = [1, 1, 1, 38]
CASES = [
    {"name": "upstream_like", "num_beams": 3, "max_new_tokens": 40, "repetition_penalty": 10.0, "length_penalty": 0.0, "min_new_tokens": 0},
    {"name": "length_cap", "num_beams": 3, "max_new_tokens": 6, "repetition_penalty": 1.0, "length_penalty": 1.0, "min_new_tokens": 0},
    {"name": "min_new_tokens", "num_beams": 4, "max_new_tokens": 30, "repetition_penalty": 1.3, "length_penalty": 0.5, "min_new_tokens": 9},
    {"name": "long", "num_beams": 3, "max_new_tokens": 40, "repetition_penalty": 10.0, "length_penalty": 1.0, "min_new_tokens": 15},
]


def toy_logits(seq):
    last, n = seq[-1], len(seq)
    row = [((v * 7 + last * 13 + n * 5) % 31) * 0.25 + v * 0.001 for v in range(VOCAB)]
    row[EOS] += (n - 12) * 0.6
    return row


class Toy(PreTrainedModel, GenerationMixin):
    config_class = PretrainedConfig

    def __init__(self, config):
        super().__init__(config)
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask=None, **kwargs):
        b, t = input_ids.shape
        logits = torch.zeros(b, t, VOCAB, dtype=torch.float32)
        for i in range(b):
            logits[i, -1] = torch.tensor(toy_logits(input_ids[i].tolist()), dtype=torch.float32)
        return CausalLMOutputWithPast(logits=logits)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}


def main():
    model = Toy(PretrainedConfig(vocab_size=VOCAB, is_encoder_decoder=False)).eval()
    out = {"transformers": version("transformers"), "vocab": VOCAB, "eos": EOS, "prompt": PROMPT, "cases": []}
    for case in CASES:
        with torch.no_grad():
            seq = model.generate(
                torch.tensor([PROMPT]), num_beams=case["num_beams"], do_sample=False,
                max_new_tokens=case["max_new_tokens"], min_new_tokens=case["min_new_tokens"] or None,
                repetition_penalty=case["repetition_penalty"], length_penalty=case["length_penalty"],
                eos_token_id=EOS, pad_token_id=EOS, use_cache=False, early_stopping=False)
        out["cases"].append({**case, "tokens": seq[0, len(PROMPT):].tolist()})
        print(case["name"], out["cases"][-1]["tokens"])
    with open(sys.argv[1], "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
