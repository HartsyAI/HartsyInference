"""H1 multi-speaker reference runner: the pinned multi demo (with h1-multi.patch) at a fixed size, with the step-0 dump.

Run from the root of the reference checkout (meituan-longcat/LongCat-Video at 58eb336e) holding this file, h1_multi_dump.py and
h1_seq.py, with h1-sdpa.patch, h1-sdpa-cross.patch, h1-int8-meta.patch and h1-multi.patch applied (see README.md).

- The avatar pipeline picks its size from the input image's aspect-ratio bucket; this runner fixes it to H1_SIZE
  (height x width, default 368x640).
- H1_SEED (default 42), H1_SEPARATE, H1_SEQ, H1_NUM_FRAMES and H1_DUMP_DIR come from the patch.
- H1_MULTI_DUMP_DIR turns the step-0 dump on (h1_multi_dump.py); H1_MULTI_BLOCKS names its blocks (default 0,1,24,47).
- H1_EAGER_MAP=1 replaces torch.compile with the identity before the reference is imported. The reference compiles the
  reference-map function; use this only if that compilation fails, and say so in the report.

Example:
  H1_ATTN=sdpa H1_SEQ=1 H1_SEED=1234 H1_MULTI_DUMP_DIR=$E/multi-dump H1_DUMP_DIR=$E/frames \\
  RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29512 \\
  python h1_multi_runner.py --checkpoint_dir $M/LongCat/avatar-1.6 --model_type avatar-v1.6 --use_distill --use_int8 \\
    --mix_steps 4 --mix_alpha 1.0 --input_json assets/avatar/multi_example_1.json --resolution 480p --num_segments 1 \\
    --output_dir $E/out
"""

import os
import sys

sys.path.insert(0, os.getcwd())


def install():
    """The size override and the dump hooks, put on the imported reference."""
    from longcat_video.modules.avatar import attention as avatar_attention
    from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline

    import h1_multi_dump

    height, width = (int(v) for v in os.environ.get("H1_SIZE", "368x640").split("x"))
    # The pinned aspect-ratio buckets have no 368x640 entry, so the size is replaced at run time.
    LongCatVideoAvatarPipeline.get_condition_shape = lambda self, condition, resolution, scale_factor_spatial=32: (height, width)
    h1_multi_dump.install(LongCatVideoAvatarPipeline, avatar_attention)


if __name__ == "__main__":
    if os.environ.get("H1_EAGER_MAP", "0") == "1":
        import torch

        torch.compile = lambda function=None, **_: function if function is not None else (lambda f: f)
        print("[H1_MULTI] torch.compile is the identity (H1_EAGER_MAP=1)", flush=True)

    import run_demo_avatar_multi_audio_to_video as demo

    install()
    demo.generate(demo._parse_args())
