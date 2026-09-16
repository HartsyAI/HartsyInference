"""Dumps SheetSage2's symbolic contract from the released implementation, for the C# port to be checked against.

Covers the two pieces that need no weights: the ScoreTokenizer's vocabulary layout and the PromptGrammarState's
allow-mask. Both are checkpoint facts — the decoder's output projection is as wide as the vocabulary and every
id's meaning comes from which range it falls in, so an off-by-one mis-reads every transcription instead of
failing. The neural stages get their own dump once weights are in hand (gates S2 onward).

The source is read out of a ComfyUI checkout rather than imported, so nothing here needs comfy or torch
installed; torch is stubbed because `allowed()` builds its mask as a tensor.

    python3 dump_sheetsage2_reference.py --source /path/to/ComfyUI/comfy/audio_encoders/sheetsage2.py

Writes sheetsage2_reference/grammar.json. Never point --source at a running ComfyUI install's live service; a
checkout is fine, it is only read.
"""
import argparse
import json
import re
import sys
import types
from pathlib import Path

# Grammar states worth pinning: a fresh decode, each field that closes earlier ones, the two states that demand
# a partner token, and the shift-run limit that stops a decode walking forward forever.
STATES = {
    "fresh": [],
    "after_time": [("time", 10)],
    "after_meter": [("meter", 3)],
    "after_pitch": [("pitch", 60)],
    "after_key": [("key", 1)],
    "after_pitch_duration": [("key", 1), ("pitch", 60), ("duration", 2)],
    "shift1": [("subbeat_shift", 0)],
    "shift4": [("subbeat_shift", 0)] * 4,
    "shift5": [("subbeat_shift", 0)] * 5,
    "event_then_shift": [("key", 1), ("subbeat_shift", 1)],
}

PROBES = ("subbeat_shift", "time", "meter", "eighth_position", "structure", "key",
          "majmin_chord", "full_chord", "pitch", "duration")

# Clip lengths worth pinning: inside one window, exactly one window, one second over (which pulls the second
# window back so it still reads a full one), and several multi-window cases.
WINDOW_DURATIONS = (30.0, 120.0, 300.0, 301.0, 450.0, 600.0, 905.0, 1200.0)


def load_window_plan(source: Path):
    """Executes just sliding_window_plan, which depends on nothing."""
    src = source.read_text()
    namespace = {}
    exec(src[src.index("def sliding_window_plan"):src.index("def overlap_prefix")], namespace)
    return namespace["sliding_window_plan"]


def load_reference(source: Path):
    """Executes just the constants, ScoreTokenizer and PromptGrammarState out of the released module."""
    src = source.read_text()
    blocks = []
    for block in re.split(r"\n(?=class |[A-Z_]+ = |def )", src[src.index("CHROMATIC_SHARPS = ("):]):
        if (block.startswith(("class ScoreTokenizer", "class PromptGrammarState", "FIELD_TO_INDEX"))
                or re.match(r"^[A-Z_]+ = ", block)):
            blocks.append(block)

    class _Mask:
        def __init__(self, n):
            self.data = [False] * n

        def __setitem__(self, key, value):
            if isinstance(key, slice):
                for i in range(*key.indices(len(self.data))):
                    self.data[i] = value
            else:
                self.data[key] = value

        def __getitem__(self, key):
            return self.data[key]

    torch = types.ModuleType("torch")
    torch.bool = object()
    torch.zeros = lambda n, dtype=None, device=None: _Mask(n)
    sys.modules["torch"] = torch
    namespace = {"torch": torch}
    exec("\n".join(blocks), namespace)
    return namespace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Path to the released sheetsage2.py")
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("sheetsage2_reference"))
    args = parser.parse_args()

    namespace = load_reference(args.source)
    tokenizer = namespace["ScoreTokenizer"]()
    grammar = namespace["PromptGrammarState"]

    def token(field: str, offset: int) -> int:
        return getattr(tokenizer, f"{field}_token_start") + offset

    def mask_for(updates):
        state = grammar(tokenizer)
        for field, offset in updates:
            state.update(token(field, offset))
        allowed = state.allowed("cpu").data
        return {
            "allowedCount": sum(allowed),
            "eos": allowed[tokenizer.eos_token],
            **{name: allowed[token(name, 0)] for name in PROBES},
        }

    payload = {
        "tokenCount": tokenizer.n_tokens,
        "chordLabels": len(tokenizer.full_chord_labels),
        "meterPairs": len(tokenizer.meter_pairs),
        "structureLabels": len(namespace["STRUCTURE_LABELS"]),
        "durationTemplates": len(namespace["DURATION_TEMPLATES"]),
        "promptPrefix": tokenizer.prompt_prefix(),
        "ranges": [{"name": name, "start": start, "end": end} for name, start, end in tokenizer.ranges],
        "tokenTypes": {str(t): tokenizer.token_type(t)
                       for t in (0, 1, 2, 3, 7, 260, 517, 30_516, 31_012, 31_037, 31_677)},
        "states": {name: mask_for(updates) for name, updates in STATES.items()},
    }

    plan = load_window_plan(args.source)
    payload["windows"] = {
        f"{duration:g}": [
            {
                "start": window["start"],
                "end": window["end"],
                "acceptStart": window["accept_start"],
                "acceptEnd": window["accept_end"],
                "prefixEnd": window["prefix_end"],
                "generationStop": window["generation_stop"],
            }
            for window in plan(duration)
        ]
        for duration in WINDOW_DURATIONS
    }

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "grammar.json"
    path.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {path} ({payload['tokenCount']} tokens, {len(payload['states'])} grammar states, "
          f"{len(payload['windows'])} window plans)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
