# LongCat-Video base pipelines: per-step reference dump

Reference for phase B4 (`LongCatVideoPipeline`, the flow-match sigma schedule, the continuation chain). The dumper runs the pinned reference's own
`LongCatVideoPipeline` methods (`generate_t2v`, `generate_i2v`, `generate_vc` with its default KV cache and `enhance_hf`, and a two-segment
chain), with the reference's own `FlowMatchEulerDiscreteScheduler`, its DiT, and its CFG combination, on the tiny checkpoint that the B1 dump wrote.

Pinned commit: `meituan-longcat/LongCat-Video` @ `58eb336edc5c8c63389838727e40b8c18434e54e`.
Source the dump needs (text only): `longcat_video/pipeline_longcat_video.py`, `longcat_video/modules/scheduling_flow_match_euler_discrete.py`,
`longcat_video/utils/bukcet_config.py`, and the four demo scripts (`run_demo_text_to_video.py`, `run_demo_image_to_video.py`,
`run_demo_video_continuation.py`, `run_demo_long_video.py`) for the call shapes. The DiT, attention and `context_parallel` modules come from the B1
checkout of the same commit; `autoencoder_kl_wan.py` is fetched but replaced by a stub, because the VAE is a latent seam here.

Run in the `hartsy-ref:cpu` image only (never the Windows host). Assemble the source tree (`longcat_video/` with all of the above) into a scratch
folder, then:

```
python dump_longcat_pipeline.py --repo /src --ckpt /b1/ckpt --out /out/longcat_pipeline_reference_tensors
```

`--ckpt` is the `ckpt/` folder of the B1 dump (`longcat_dit_reference_tensors/ckpt`). The output (gitignored by `*_reference_tensors/`) holds a copy
of that checkpoint, `meta.json` (scenarios, schedules, every recorded tensor's shape), and raw F32 files.

Stubbed, so NOT verified by this dump: the Wan VAE encode and decode (latents are queued by the caller), umT5 (fixed embeddings with a valid length),
the pixel round trip of the continuation chain (the chain runs in latent space), and the shipped scheduler `shift` (the pinned repo does not ship
`scheduler_config.json`; the dump uses shift 3 and records it). flash-attn is replaced by SDPA, as in the B1 dump, and the varlen stub handles the
CFG batch of two.

Scenarios: `t2v` and `t2v_distill` (4 steps, guidance 4), `i2v` (one conditioning latent frame), `vc` (continuation, 2 context latents, 13 steps
after `enhance_hf`), `vc_nocache` (the same call with `use_kv_cache=False`, recorded for the equivalence note below), and `chain0`/`chain1`.

Why `vc_nocache` is not the port's target: the reference's cached path computes the CFG scale over the noisy frames only, and its no-cache branch
computes it over every frame, including the conditioning frames whose prediction it then discards. The two final latents differ (relRMS 1.2 on this
tiny checkpoint). The C# pipeline matches the cached path, the reference default.

C# side: set `HARTSY_LONGCAT_PIPELINE_DIR` to the output directory and run
`dotnet test tests/HartsyInference.Diffusion.Tests --filter FullyQualifiedName~LongCatPipelineParityTests`. The test is skipped, and so is not
evidence, until the variable is set. Tolerance relRMS 1e-5 per schedule, velocity, stepped latent and final latent; the report is written next to
the dump as `pipeline-parity-report.txt`.

## Refinement dump

`dump_longcat_refine.py` runs the pinned `generate_refine` with the DiT's block-sparse attention enabled, with an image condition
and without one, on the same tiny checkpoint. The reference's attention kernel is Triton and needs CUDA: the dump replaces
`flash_attn_bsa_3d` with the reference's own blocking and top-k selection (compiled from `block_sparse_attention/bsa_interface.py`)
and torch SDPA over the selected blocks. It needs the whole pinned checkout, because it reads that file. With `--triton` the
reference's own kernel runs instead: that needs a CUDA card and an image with torch, triton, einops, safetensors, tqdm, Pillow,
regex and a C compiler (`TRITON_F32_DEFAULT=ieee`); the DiT stays on the CPU. The tiny blocks are the released 4x4x4, because the
kernel does not compile 2x2x2.

```
TORCHDYNAMO_DISABLE=1 python dump_longcat_refine.py --repo /src --ckpt /b1/ckpt --out /out/longcat_refine_reference_tensors
HARTSY_LONGCAT_REFINE_DIR=/out/longcat_refine_reference_tensors dotnet test tests/HartsyInference.Diffusion.Tests \
    --filter FullyQualifiedName~LongCatRefinementParityTests
```

The test takes a `--triton` dump. For a stand-in dump set `HARTSY_LONGCAT_REFINE_ATTENTION=stand-in` as well.

The VAE and the pixel preparation are not verified by it: the latents of the upsampled clip and of the condition frames are queued.

## Real-weight runner

`run_base_reference.py` runs the pinned base pipeline on the released weights for a comparison with the engine, on a card that cannot hold the BF16 DiT
and umT5 together. It encodes the text once and frees umT5, spreads the DiT over GPU and host with accelerate (block by block), starts from a saved
latent and writes that latent, the final latent, every frame as PNG and `report.json` (optionally the latent after each step). Attention is torch SDPA
instead of flash-attn; the report lists every way it differs from `run_demo_text_to_video.py`. Its own header gives the command and the defaults.

It needs the pinned reference's requirements (torch 2.6.0, transformers 4.41.0, diffusers 0.35.1, einops, ftfy, loguru) plus accelerate 1.6.0.
Not in the Windows venv.

The engine side of the comparison starts from the runner's `noise.npy`:
`hartsy video "<prompt>" -m longcat-video -n "<negative prompt>" --seed 1234 --set diagnostics.longcatNoiseNpy=<out>/noise.npy`.

To try it without the released files, `make_tiny_base_checkpoint.py` writes a tiny model folder from the reference's own classes (random weights; its
LoRA has rank 4, so the distilled call takes `--lora-dim 4`). That trial checks that the runner runs, not what it computes at full size.
