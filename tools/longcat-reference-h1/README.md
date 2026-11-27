# H1 reference patch and runner (never merge)

Purpose: run the pinned LongCat-Video avatar demo (`meituan-longcat/LongCat-Video` @ `58eb336edc5c8c63389838727e40b8c18434e54e`)
for the H1 comparison. This branch exists so the GPU server can fetch two files. It is not for merging into HartsyInference.

Status: written and checked locally only. The patch applies cleanly to the pinned demo file (blob checked against the GitHub tree),
and the patched demo parses. NOT run anywhere: the runner has not been executed on the server or the reference checkout.

## Files
- `h1-sdpa.patch`: a `git diff` against `longcat_video/modules/avatar/attention.py` at the pinned sha. It swaps
  `flash_attn_func` for `torch.nn.functional.scaled_dot_product_attention` in the self and cross attention of the
  `enable_flashattn2` branch. It runs only when `H1_ATTN=sdpa` is set; otherwise the file behaves as the pinned file.
- `h1-reference.patch`: a `git diff` against `run_demo_avatar_single_audio_to_video.py` at the pinned sha. It adds
  `H1_SEED` (default 42), and `_h1_dump`, which writes `final_latent.npy` and `frame_NNN.png` to `H1_DUMP_DIR`. It does nothing
  unless `H1_DUMP_DIR` is set.
- `h1_runner.py`: imports the patched demo, fixes the size to `H1_SIZE` (default `368x640`, height by width) by replacing
  `LongCatVideoAvatarPipeline.get_condition_shape`, then calls the demo's `_parse_args` and `generate`.

## Apply and run (on the server, from the reference checkout)
```
cd /home/hartsy/refs/LongCat-Video
git apply /path/to/h1-reference.patch
git apply /path/to/h1-sdpa.patch
cp /path/to/h1_runner.py .
export H1_ATTN=sdpa   # required: without it the attention still calls flash_attn
```
Then the launch in `h1_runner.py`'s docstring. The input JSON is the one the engine's 1.6 runs used
(`speech_fox_join_x2_7p44s.wav`, `portrait_anime_stylised_crop.png`), in the demo's `input_json` format
(`prompt`, `cond_audio.person1`, `cond_image`).

## Substitution: flash-attn to SDPA (h1-sdpa.patch)
The pinned 1.6 config sets `enable_flashattn2: true` (`base_model_int8/config.json`, read from the HF resolve URL), and the block-sparse
path is off (`enable_bsa: false`). The patch replaces the flash call in the avatar attention with torch SDPA, gated by `H1_ATTN=sdpa`.
It covers the self and cross attention of `longcat_video/modules/avatar/attention.py` only. Not checked: that the avatar DiT never
reaches `longcat_video/modules/attention.py` (the base attention file, which also calls flash). Check that on the server with a
grep for `flash_attn` in the code path before trusting a run.

## Known gaps (not fixed here)
- `run_demo_avatar_single_audio_to_video.py` imports `audio_separator.separator` at module level. Install `audio-separator==0.30.2`
  in the venv (the version the Kim reference used), or the run fails on import.
- The size override only changes the avatar pipeline's `get_condition_shape`. Anything else that reads the aspect-ratio buckets
  is not changed.
- The demo's `save_video_ffmpeg` path still writes an mp4. The PNG frames are the lossless record.
