"""H1 reference runner: the pinned LongCat-Video avatar demo at the size and seed the engine's 1.6 runs used.

Run from the root of the reference checkout (meituan-longcat/LongCat-Video at 58eb336e), after the venv and
`git apply h1-reference.patch` (see README.md). It does not change the demo's own code beyond the patch:

- The avatar pipeline picks its size from the input image's aspect-ratio bucket. This runner fixes the size to
  H1_SIZE (default 368x640, height x width) by replacing `get_condition_shape` at run time.
- The patch makes the seed H1_SEED (default 42; set 1234 for the H1 run).
- The patch writes the final latent and every decoded frame to H1_DUMP_DIR when it is set.

Example (one process, the demo needs torch.distributed):
  H1_SEED=1234 H1_DUMP_DIR=/home/hartsy/evidence/h1-ref-run1 \
  RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29511 \
  python h1_runner.py --checkpoint_dir /mnt/model-storage/Models/LongCat/avatar-1.6 \
    --model_type avatar-v1.6 --use_distill --use_int8 --mix_steps 4 --mix_alpha 1.0 \
    --input_json <input.json> --resolution 480p --output_dir /home/hartsy/evidence/h1-ref-run1
"""

import os
import sys

sys.path.insert(0, os.getcwd())

from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline  # noqa: E402
import run_demo_avatar_single_audio_to_video as demo  # noqa: E402

_SIZE = os.environ.get("H1_SIZE", "368x640")
_HEIGHT, _WIDTH = (int(v) for v in _SIZE.split("x"))


def _fixed_condition_shape(self, condition, resolution, scale_factor_spatial=32):
    """Return the fixed (height, width) instead of the aspect-ratio bucket. The pinned buckets have no 368x640 entry."""
    return _HEIGHT, _WIDTH


LongCatVideoAvatarPipeline.get_condition_shape = _fixed_condition_shape

if __name__ == "__main__":
    args = demo._parse_args()
    demo.generate(args)
