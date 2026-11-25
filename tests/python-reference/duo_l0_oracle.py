"""L0 oracle for the Duo packed layout (HartsyInference PR 2a).

Runs upstream inclusionAI/Duo's own prepare_sequence (duo/data/sequence.py) and build_mask (duo/models/realization/modules/masks.py)
on the cases in duo_l0_cases.json, and writes the expected groups, per-row positions, tags, time ids and causal mask rectangles.
The C# test compares its DuoLayoutBuilder and DuoMask output with this file.

Run in the hartsy-ref:cpu image, never on the Windows host. Upstream source is read from --src (a checkout of the pinned commit
with only the five files listed in the PR); nothing is downloaded here.

  python -I duo_l0_oracle.py --src /duo --cases duo_l0_cases.json --out duo_l0_expected.json --commit <sha>
"""

import argparse
import json
import sys

import torch


def build_session(case):
    n1, n2 = case["sharedDescription"], case["sharedCue"]
    blocks_in = case["blocks"]
    rows = n1 + n2 + sum(b["bodyRows"] for b in blocks_in)
    text_groups = [
        {"section": "assistant_description", "block": -1, "start": 0, "end": n1},
        {"section": "speech_cue", "block": -1, "start": n1, "end": n1 + n2},
    ]
    pos = n1 + n2
    blocks = []
    for b, blk in enumerate(blocks_in):
        text_groups.append({"section": "assistant_body", "block": b, "start": pos, "end": pos + blk["bodyRows"]})
        pos += blk["bodyRows"]
        t, h, w = blk["video"]
        steps = blk["audioSteps"]
        spans = [[float(ms), float(ms) + 40.0] for ms in blk["frameStartsMs"]]
        blocks.append({
            "start_ms": float(blk["startMs"]),
            "end_ms": float(blk["startMs"]) + 1000.0,
            "assistant": {
                "video": torch.zeros(24, t, h, w),
                "audio": torch.zeros(2, 32, steps),
                "video_weight": torch.ones(t),
                "audio_weight": torch.ones(steps),
                "video_spans": spans,
            },
        })
    rt, rh, rw = case["reference"]
    return {
        "protocol": "reciprocal_frame_chatml_v5",
        "sample_id": case["sampleId"],
        "intent_states": torch.zeros(rows, 5120),
        "text_groups": text_groups,
        "blocks": blocks,
        "reference": torch.zeros(24, rt, rh, rw),
        "clean_blocks": case["clean"],
    }


def run_case(case, prepare_sequence, build_mask):
    sessions = [(build_session(s), True) for s in case["samples"]]
    targets = [set(s["noisy"]) for s in case["samples"]]
    prep = prepare_sequence(sessions, seed=0, max_seq_len=0, target_blocks=targets, rope=case["rope"])
    inputs, layout = prep["inputs"], prep["layout"]
    groups = layout["groups"]
    rects = build_mask(groups, groups, "cpu", mode="causal")
    rect_rows = [
        [int(qa), int(qb), int(ka), int(kb)]
        for (qa, qb), (ka, kb) in zip(rects.q_ranges.tolist(), rects.k_ranges.tolist())
    ]
    return {
        "name": case["name"],
        "rope": case["rope"],
        "liveRows": layout["live_rows"],
        "totalRows": layout["total_rows"],
        "groups": [[g["sample"], g["block"], g["kind"], g.get("modality"), g["start"], g["end"]] for g in groups],
        "positions": inputs["img_position_ids"][0].tolist(),
        "tags": inputs["token_tags"].tolist(),
        "timeIds": inputs["inverse_indices"].tolist(),
        "maskRects": rect_rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.src)

    from duo.data.sequence import prepare_sequence
    from duo.models.realization.schedulers.flow import grid
    from duo.models.realization.modules.masks import build_mask

    with open(args.cases, encoding="utf-8") as handle:
        cases = json.load(handle)["cases"]
    expected = {
        "source": "inclusionAI/Duo",
        "commit": args.commit,
        "torch": torch.__version__,
        "cases": [run_case(c, prepare_sequence, build_mask) for c in cases],
        "grids": [
            {"steps": s, "shift": sh, "times": grid(s + 1, shift=sh).tolist()}
            for s, sh in [(4, 5.0), (50, 1.0)]
        ],
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(expected, handle, indent=1)
    print(f"wrote {len(expected['cases'])} cases to {args.out}")


if __name__ == "__main__":
    main()
