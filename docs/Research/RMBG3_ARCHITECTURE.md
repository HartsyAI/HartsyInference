# Bria RMBG 3.0

Source snapshot 2026-11-23. RMBG 3.0 is not a segmentation network: it is Fibo 2.0 tuned to draw an alpha matte as an
edit of the source image, so the engine runs it as a variant of the Fibo 2.0 recipe. Status and open items:
[MODEL_STATUS_VISION.md](../Checklists/MODEL_STATUS_VISION.md#rmbg-30).

## Pins

| What | Value |
|---|---|
| Model | `briaai/RMBG-3.0` @ `87330e3f87d9d856a21453e5acd120219984a66b` (created 2026-11-10; gated `auto`: a form, then immediate access) |
| Ungated Bria repack the catalog uses | `briaai/RMBG-3.0-Comfy` @ `63880aa074782b09e7ae17184960dfd6de70ab85`, tagged `cc-by-nc-4.0` |
| Transformer (Comfy repack) | `diffusion_models/rmbg3_bf16.safetensors` 16,452,934,194 B, sha256 `a5df048f799025510f29a743c0dca0d4f1362478ec00f6f7c86ee66d57b4e5d9`; `rmbg3_int8_convrot.safetensors` 8,408,562,474 B, sha256 `bc29b9a87ed378596c5ca5bc0c5bdfaa8aabc8415d33f3c767f9a3cbd9fd566d` |
| Encoder and VAE | the Fibo 2.0 assets (`SideModels.Fibo2Qwen3VL4B`, `Fibo2Qwen3VL4BInt8`, `Fibo2Vae`): the Comfy repack's files have identical sha256 |
| Reference code | huggingface/diffusers @ `98dfc1aa4146e0d8bcbdb2d00ff68e698ce3304c`: `pipeline_bria_fibo2.py`, `transformer_bria_fibo2.py`, `image_processor.py` (diffusers 0.42 or newer; no `trust_remote_code`) |
| Scheduler config | read from the public mirror `ml-resources/Fibo-2.0-edit-turbo` @ `fce030b05e455ac1bd800cf67d78c177e846ae10` (the original is gated) |

## Facts that decide the implementation

- **Same weights layout as Fibo 2.0.** A header read of `rmbg3_bf16.safetensors` and `Comfy-Org/Fibo-2.0` @ `4742d4bec410034142c8e65559e5f00c6f29eccb`
  `fibo2_dev_bf16.safetensors` gives 634 tensors with identical keys, dtypes and shapes; `rmbg3_int8_convrot` and
  `fibo2_edit_turbo_int8_convrot` give 1104 tensors, also identical. The file hashes differ, so RMBG 3.0 is a fine-tune. The loader, text
  encoder, VAE and edit path are the Fibo 2.0 ones; the variant (`Fibo2Variants.Rmbg3`) changes the defaults and the instruction only.
- **One instruction.** `{"edit_instruction": "Generate a detailed grayscale alpha matte. Map the opaque foreground to white and the background to black. Produce soft, anti-aliased grayscale gradients at the edges of the subject to represent fine details and transparency."}`;
  the card says a changed or missing instruction gives a coloured picture. diffusers wraps it as `{"image": <vision marker>, "edit_instruction": ...}`
  (`_get_edit_prompt`), which is what `Fibo2EditPrompt.Build` writes. Settings: 4 steps, guidance 1.0.
- **Sigma shift is dynamic.** The scheduler has `use_dynamic_shifting: true`, `base_shift 0.5`, `max_shift 1.15`, `base/max_image_seq_len 256/4096`,
  so `mu` is linear in the latent token count (`(H/16)·(W/16)`) and unclamped. The engine's Fibo 2.0 recipe otherwise uses a fixed 1.15 (ComfyUI's);
  they agree only at 1024×1024. The RMBG 3.0 variant follows diffusers (`Fibo2Recipe.DynamicShift`); whether ComfyUI does the same is unverified.
- **Preprocessing.** The trained size for the source aspect ratio (multiple of 16, within ±10 % of 1024²), a Lanczos cover-resize and a centre crop; Qwen3-VL
  gets a copy of at most 768 px; no mean/std normalisation. `Fibo2EditImages.TrainedSize` and `Prepare` implement the ComfyUI form of this.
- **Output.** The VAE-decoded RGB picture, then `result.convert("L").resize(source.size, Image.LANCZOS)`; white is the subject. Soft 8-bit alpha, no sigmoid.
  `Rmbg3Matte.AlphaFromPicture` reproduces PIL's integer luma and `Resample.LanczosPil8`. Because the input is cropped to the trained aspect and the
  matte is stretched back to the source size, an off-ratio picture gets a slightly misregistered matte; the engine matches the card.
- **Weights are not tiny.** Card: "32 GB or more" of GPU memory in bf16 (peak 28.5 GiB on an L40S), about 6 s per image (vendor figures). The int8 build is
  8.41 GB with a 4.86 GB encoder, so on 12 GB the encoder and transformer must not be resident together.
- **Attention hints.** The int8 transformer carries 40 `layers.N.attention.comfy_attention.config` U8 blobs, each `{"attention": "comfy_kitchen_int8"}`
  (all 40 read, 2026-11-25). The engine has no int8 attention, so it runs float attention; the loader warns that outputs differ from ComfyUI by that
  quantization error and refuses a malformed blob. The bf16 builds carry no hints. The 35-byte blob is the same on every block.
- **Encoder file.** The int8 encoder is the pinned Comfy-Org `qwen3vl_4b_int8_convrot.safetensors` (4,858,559,240 B, sha256 `32de113c…`). Its vision tower
  is dense (315 BF16 tensors; I8 only in the text blocks, 253). A same-named file from another build (4,450,477,142 B, sha `5ebf6c30…`, an I8 vision tower)
  satisfied the old target name and failed in `Qwen3VlVisionEncoder.LoadWeights`. The target now lives in its own folder, and a file at the pinned path
  whose hash differs is refused.

## Licence

CC BY-NC 4.0. The card: "Weights are released under Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0), the same terms as the
Fibo 2.0 family." and "Commercial use requires a separate agreement with Bria AI." The LICENSE file in `briaai/RMBG-3.0` is the unmodified CC BY-NC 4.0
legal code (blob `fe463e0f7888bbf8b82e42d55f5743508ddafb7e`; read through a mirror matched by git object id, since the original is gated). The Hub metadata
says `license_name: bria-fibo` and the gate links Bria's pricing page; the discrepancy is unresolved. The catalog records the terms and does not enforce them.

## Not included: VRMBG-3.0

`briaai/VRMBG-3.0` @ `59716e19a6cc97f91311edea190938219c097b76` is a separate ~220M BiRefNet-class video matting model (6-channel input: the frame and the previous
frame times its matte; 1024², sigmoid). Its gate is `manual` and needs a commercial agreement ("Academic access will be granted upon request"); there is no
download path and no repo LICENSE file, so there is nothing to repack. Its code and weight hashes were not read.

## Open questions

- Real-weight parity against the diffusers pipeline (and ComfyUI) on saved noise: token ids, text taps, reference latents, the sigma list, per-step velocity,
  the decoded picture and the 8-bit matte. Nothing has run on real weights or a GPU, and neither has the Fibo 2.0 edit path it uses.
- Whether `Fibo2EditImages` (ComfyUI's crop and 768 px grid) equals diffusers' `image_processor.py` to a few 8-bit levels.
- Memory and speed on a 12 GB card; the int8 ConvRot kernels on sm_86.
- The `bria-fibo` licence name against the CC BY-NC 4.0 text.
- `ImageRequest.RemoveBackgroundModel = rmbg3` runs after the generating pipeline's lease is released (`ImagesService`), so the 8B matter can load; whether the
  generator is actually evicted on a 12 GB card, and the round trip speed, are unmeasured.
