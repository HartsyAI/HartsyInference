"""Compares a multi-speaker reference dump (h1_multi_dump.py) with the engine's step-0 probe. Report only: no tolerance.

  python h1_multi_compare.py <H1_MULTI_DUMP_DIR> <engine diagnostics.longcatProbeDump dir> [<engine diagnostics.longcatConditioningDump dir>]

For every probe the two sides share: the relative RMS difference (against the reference) and the largest absolute difference. For
each reference attention map also the region each noisy token takes (the largest value; regions past the third count as
background): the share of tokens that agree, the count of each class on both sides and how many tokens differ in each direction.
Names one side lacks are listed, as are probes whose element counts differ. With the engine's conditioning dump, the audio tokens the
engine projected from the substituted embedding are compared too. numpy only.
"""

import os
import re
import sys

import numpy as np


def load(path):
    return np.load(path).astype(np.float64)


def probes(directory):
    found = {}
    for file in os.listdir(directory):
        match = re.fullmatch(r"call00_probe_(.+)\.npy", file)
        if match:
            found[match.group(1)] = os.path.join(directory, file)
    return found


def relative_rms(actual, expected):
    norm = np.sqrt(np.mean(expected ** 2))
    error = np.sqrt(np.mean((actual - expected) ** 2))
    return error / norm if norm > 0 else error


def compare(name, actual, expected):
    if actual.size != expected.size:
        print(f"{name:34s} SIZE MISMATCH: the engine holds {actual.shape}, the reference {expected.shape}")
        return None
    actual = actual.reshape(expected.shape)
    print(f"{name:34s} relRMS {relative_rms(actual, expected):.3e}  max abs {np.max(np.abs(actual - expected)):.3e}  {expected.shape}")
    return actual


def classes(attention_map):
    return np.minimum(np.argmax(attention_map, axis=0), 2)


def order(name):
    return [(1, int(part), "") if part.isdigit() else (0, 0, part) for part in re.split(r"(\d+)", name)]


def main():
    if len(sys.argv) not in (3, 4):
        raise SystemExit(__doc__)
    reference = probes(os.path.join(sys.argv[1], "probe"))
    engine = probes(sys.argv[2])
    shared = sorted(set(reference) & set(engine), key=order)
    if not shared:
        raise SystemExit("the two dumps share no probe")
    mismatched = []
    for name in shared:
        expected = load(reference[name])
        actual = compare(name, load(engine[name]), expected)
        if actual is None:
            mismatched.append(name)
            continue
        if not name.endswith(".ref_attn_map"):
            continue
        ours, theirs = classes(actual), classes(expected)
        counts = " ".join(f"class{c}: reference {np.sum(theirs == c)}, engine {np.sum(ours == c)};" for c in range(3))
        moved = " ".join(f"{a}->{b}: {np.sum((theirs == a) & (ours == b))};" for a in range(3) for b in range(3) if a != b)
        print(f"  argmax agreement {np.mean(ours == theirs):.6f} of {theirs.size} tokens. {counts} reference->engine {moved}")
    if len(sys.argv) == 4:
        tokens = os.path.join(sys.argv[3], "audio_tokens_step00.npy")
        if "audio_tokens" in reference and os.path.exists(tokens):
            compare("audio_tokens (conditioning dump)", load(tokens), load(reference["audio_tokens"]))
    print("only in the reference:", sorted(set(reference) - set(engine), key=order))
    print("only in the engine:", sorted(set(engine) - set(reference), key=order))
    print(f"compared {len(shared) - len(mismatched)} probes, size mismatches: {mismatched}")


if __name__ == "__main__":
    main()
