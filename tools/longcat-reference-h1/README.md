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
  `H1_SEED` (default 42); `_h1_dump`, which writes `final_latent.npy` and `frame_NNN.png` to `H1_DUMP_DIR` when it is set; and
  `H1_SEPARATE`. With `H1_SEPARATE` unset or `0` (the default), the speech file is used unchanged (mono, 16 kHz, cut to the clip
  length) and the Kim_Vocal_2 separator is neither constructed nor run. With `H1_SEPARATE=1` the pinned separation runs.
  The pass-through needs `audio-separator` importable, because the demo imports `separate_vocals` at module level, but it does not
  load the Kim ONNX file: the separator is not constructed in pass-through mode.
- `h1-sdpa-cross.patch`: a `git diff` against `longcat_video/modules/attention.py` at the pinned sha. The avatar DiT imports
  `MultiHeadCrossAttention` from this file (`longcat_video_dit_avatar.py` line 17), and its `enable_flashattn2` branch calls
  `flash_attn_varlen_func`. The patch replaces that call with `_h1_varlen_sdpa`: one SDPA per sample, with each query segment
  attending only to its own key segment (cu_seqlens semantics). Active only with `H1_ATTN=sdpa`.
- `h1_sdpa_check.py`: a CPU check that runs the PATCHED call sites. It builds `MultiHeadCrossAttention` and compares the patched
  path (`H1_ATTN=sdpa`) with the flash contract, using a fake `flash_attn_varlen_func` that takes the cu_seqlens the original call builds
  (cumsum). Batch sizes 1 and 3, uneven key lengths, and a length-1 segment. It also checks the helper's assertions and the no-leakage
  property. Run: `python h1_sdpa_check.py <reference root with both attention patches applied>`. It does not run flash-attn, and the
  block-sparse module is stubbed (not used here).
- `h1_runner.py`: imports the patched demo, fixes the size to `H1_SIZE` (default `368x640`, height by width) by replacing
  `LongCatVideoAvatarPipeline.get_condition_shape`, then calls the demo's `_parse_args` and `generate`.

## Apply and run (on the server, from the reference checkout)
```
cd /home/hartsy/refs/LongCat-Video
git apply /path/to/h1-reference.patch
git apply /path/to/h1-sdpa.patch
git apply /path/to/h1-sdpa-cross.patch
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
- `run_demo_avatar_single_audio_to_video.py` imports `audio_separator` at module level, through `vocal_separator.py`. Install `audio-separator==0.30.2`
  (the version the Kim reference used). Installing it does not download the Kim model; only constructing a `Separator` does. Stubbing it
  is possible, but the stub must provide `audio_separator.separator.uvr_lib_v5.spec_utils`, so installing is the simpler path.
- The size override only changes the avatar pipeline's `get_condition_shape`. Anything else that reads the aspect-ratio buckets
  is not changed.
- The demo's `save_video_ffmpeg` path still writes an mp4. The PNG frames are the lossless record.
