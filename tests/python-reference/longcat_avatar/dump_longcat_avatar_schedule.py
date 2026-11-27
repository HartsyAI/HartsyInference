"""Reference sigma lists for the LongCat-Video-Avatar pipeline (get_timesteps_sigmas, pipeline_longcat_video_avatar.py @ 58eb336).

The method is taken from the pinned source by its AST and executed with a stand-in for self, so the pipeline module's heavy imports (diffusers,
transformers, pyloudnorm) are not needed. Writes schedule.json next to the avatar audio dump (same HARTSY_LONGCAT_AVATAR_DIR).

Run in the hartsy-ref:cpu image only (never the Windows host):

    python dump_longcat_avatar_schedule.py --repo /src --out /out
"""

import argparse
import ast
import json
import os
import textwrap
import types

import numpy as np
import torch


def load_method(repo):
    path = os.path.join(repo, "longcat_video", "pipeline_longcat_video_avatar.py")
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "LongCatVideoAvatarPipeline":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "get_timesteps_sigmas":
                    code = textwrap.dedent(ast.get_source_segment(source, item))
                    namespace = {"torch": torch, "np": np}
                    exec(code, namespace)
                    return namespace["get_timesteps_sigmas"]
    raise RuntimeError("get_timesteps_sigmas not found in the pinned pipeline source")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    method = load_method(args.repo)
    cases = [
        ("avatar-v1.0", False, 50), ("avatar-v1.0", False, 4), ("avatar-v1.0", True, 50), ("avatar-v1.0", True, 12),
        ("avatar-v1.6", True, 50), ("avatar-v1.5", True, 8), ("avatar-v1.5", False, 50),
    ]
    out_cases = []
    for model_type, distill, steps in cases:
        # Per the pipeline's __init__: 1000 timesteps; 8 distill samples for avatar-v1.5, 50 otherwise.
        stand_in = types.SimpleNamespace(
            model_type=model_type,
            num_timesteps=1000,
            num_distill_sample_steps=8 if model_type == "avatar-v1.5" else 50,
        )
        sigmas = method(stand_in, steps, use_distill=distill).tolist()
        out_cases.append({"model_type": model_type, "distill": distill, "steps": steps, "sigmas": sigmas})

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "schedule.json"), "w", encoding="utf-8") as handle:
        json.dump({"cases": out_cases}, handle, indent=2)
    print(json.dumps({"cases": len(out_cases)}))


if __name__ == "__main__":
    main()
