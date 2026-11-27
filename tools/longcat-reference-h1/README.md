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
- `h1-int8-meta.patch`: a `git diff` against `longcat_video/modules/quantization.py` at the pinned sha. It builds the INT8 DiT on the
  meta device (the stock build materialises about 63.5 GB of float32 on CPU, which exceeded the 48G host cap) and loads the shards with
  `load_state_dict(assign=True)`. Before assigning, the 486 float32 parameters receive their BF16 checkpoint tensors converted to BF16
  (the only dtype difference allowed; the stock path ends in BF16 too, by copy and then cast). INT8 buffers keep their bytes. The load
  fails on any missing or unexpected key, any other dtype or shape difference, and any tensor left on meta. Applies with or without the
  attention patches.
- `h1_int8_meta_check.py`: CPU check on a tiny config. Writes a BF16 checkpoint in the release layout, loads it with the stock and the
  patched `load_quantized_dit`, and requires every state tensor to match in value, shape and dtype and nothing on meta. Then it requires
  refusal of a checkpoint with a missing key, one with an unexpected key, and one with a wrong dtype on an INT8 buffer. It refuses a
  reference root whose `quantization.py` already has the patch, so run it on a pristine checkout (a second clone at the pinned sha, not
  the run checkout). The meta-leftover refusal is not exercised by a strict load. Run:
  `python h1_int8_meta_check.py <pristine reference root> <patched quantization.py> <release base_model_int8/config.json>`.
- `inspect_int8_meta.py`: read-only. Builds the model on meta with the same replacement and compares its keys, shapes and dtypes with the
  shard headers. The only accepted dtype difference is the 486 float32-parameter / BF16-checkpoint conversion above; the count must be
  exactly 486. Any other mismatch, missing or unexpected key, or a different count exits 1.

## Apply and run (on the server, from the reference checkout)
```
cd /home/hartsy/refs/LongCat-Video
git apply /path/to/h1-reference.patch
git apply /path/to/h1-sdpa.patch
git apply /path/to/h1-sdpa-cross.patch
git apply /path/to/h1-int8-meta.patch
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

## Sequential placement on one 23.52 GiB card (h1-seq.patch, H1_SEQ=1)
Apply after the other patches, from the run checkout: `git -c core.autocrlf=false apply h1-seq.patch` (it also adds `h1_seq.py`).
Device placement only. Order: umT5 alone and the prompt encoded once into the pipeline's own cache (a miss raises), umT5 freed;
whisper alone for the audio embedding, freed; then the INT8 DiT and both LoRAs on the GPU, as stock; generation. The DiT is
moved to the host before each VAE decode only with `H1_DECODE_OFFLOAD=1`.
Memory: `h1_seq.py` resets the CUDA peaks at each stage and logs `[H1_MEM] ...` lines (allocated and reserved, now and peak, GiB)
after each stage and each denoise forward, flushed to stdout and to `H1_MEM_LOG` (default `h1_mem.log`).
Fallback: `H1_NUM_FRAMES=49` (1 mod 4, at least 14) runs 49 frames instead of 93. The demo's arithmetic supports it unchanged; the
comparison then covers 49 frames only.
Refused, not run differently: `--use_fp8`, `--compile_dit`, context parallel > 1, avatar-v1.0, `H1_SEPARATE=1`.

Check (CPU, tiny umT5, real pipeline class from the pinned package): `h1_seq_check.py <pristine reference root> <dir with h1_seq.py>`.
It covers the cache result equal to a fresh encode, the three cache misses, pipe.to with the shim, the frame override, the DiT and
LoRA move, and the forward and decode hooks. It does not cover the demo's H1 block, the GPU residency, or the memory numbers.

## Per-step dump, reference side (h1-step-dump.patch, H1_STEP_DUMP_DIR; off by default)
Apply after h1-seq.patch: `git -c core.autocrlf=false apply h1-step-dump.patch` (it edits `longcat_video/pipeline_longcat_video_avatar.py`
at three denoise loops and adds `h1_step_dump.py`). With `H1_STEP_DUMP_DIR` unset nothing is written or copied. Set, each denoise step
writes `call{n:02d}_step{i:02d}_{t,noise,latent}.npy` (float32): the timestep, the model output after the pipeline's sign flip, and the
latents after the scheduler step. Call n counts denoise calls from 0 (one per generate stage). Check: `python h1_step_dump_check.py <dir with h1_step_dump.py>`.
