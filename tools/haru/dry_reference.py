"""Penalties of the upstream Haru DRY sampler (`haru.sampling.DryState`) on fixed token histories, as JSON for the C# DryPenaltyStep.

    python dry_reference.py <path to the upstream repo's haru/ package dir> <out json>
"""

from __future__ import annotations

import json
import random
import sys

package, out = sys.argv[1:3]
sys.path.insert(0, package)
from haru.sampling import DryConfig, DryState  # noqa: E402

VOCAB = 64


def case(name: str, history: list[int], exempt: list[int], **params) -> dict:
    config = DryConfig(**params)
    state = DryState(config, set(exempt))
    for token in history:
        state.update(token)
    return {
        "name": name,
        "config": {
            "multiplier": config.multiplier, "base": config.base, "allowed_length": config.allowed_length,
            "lookback": config.lookback, "max_match_length": config.max_match_length, "max_penalty": config.max_penalty,
        },
        "exempt": exempt,
        "history": history,
        "penalties": {str(t): p for t, p in sorted(state.penalties.items())},
    }


rng = random.Random(3)
cases = [
    case("random_small_vocab", [rng.randrange(2, 12) for _ in range(400)], [0, 1]),
    case("pattern_with_exempt_gaps", [t for _ in range(60) for t in (3, 4, 5, 0, 1, 0)], [0, 1]),
    case("history_longer_than_the_window", [rng.randrange(2, 6) for _ in range(1500)], [0, 1], lookback=64, max_match_length=8),
    case("allowed_length_two", [rng.randrange(2, 5) for _ in range(300)], [0, 1], allowed_length=2, multiplier=1.5, base=2.0),
    case("capped_by_max_penalty", [7, 8, 9] * 40, [], max_penalty=3.0),
    case("single_token", [5], [0, 1]),
    case("empty", [], [0, 1]),
    case("only_exempt", [0, 1, 0, 1, 0], [0, 1]),
]
with open(out, "w") as f:
    json.dump({"vocab": VOCAB, "cases": cases}, f, indent=1)
print("wrote", out, [len(c["penalties"]) for c in cases])
