# Which workloads the Vulkan tiled default moves

`numerics.vkTiledAttention` defaults to `auto`: a dense head-major attention call on Vulkan whose operands and output are
all F32 takes `tiled.f32` instead of `flash.scalar` at head dim 128 or below, unless the cooperative-matrix row takes it
first (head dim 64 or 128 with the F16 permission or F16 operands, on a device that has that kernel). The plan asks a
planner change for "every workload whose candidates can change"
([queue](../proposed-queue.md), evidence cadence). This is that list, as a source audit at `6f9ee3d72`.

**What this is and is not.** Nothing here was run. A read-only sub-agent traced the consumers in two passes; I re-read
the rows marked *checked* and the passages named under "How a call gets there". Line numbers are as of that commit.
Token counts are given only where the source states them. The rules themselves are executed evidence: the
[route matrix and parity tests](vulkan-tiled-f32-ab.md#routes-and-parity).

## How a call gets there

- **Permission.** The shared layer sends `AllowF16` as the F16 permission (`AttentionLayerConfig.cs:126`, *checked*);
  the default is false. A direct `backend.ScaledDotProductAttention(...)` call without `allowF16` has none.
- **Operand type.** The layer computes in `ComputeDType` when the config sets it, else in the type it is handed.
  Vulkan reports no F16 activations and no BF16 (`IBackend.Residency.cs:9`, `VulkanBackend.cs:212`, *checked*), so
  families gated on those run F32 there: Wan, LTX-2, Moss Image, FLUX.3, YuE 2, the VAEs. `DitDtype.Act`
  (`DitRuntimeFlags.cs:22`, *checked*) has no backend check and follows `numerics.ditF16` (default on), so Flux,
  Flux.2, Chroma, Krea 2, Ideogram 4, Z-Image turbo, OmniGen2's main path and the Hunyuan3D DiT hand the layer F16 on
  Vulkan and **do not move**.
- **Entry.** With the cooperative-matrix kernel a batch-1 layer call may go token-major; one that kernel declines is
  permuted into the dense entry, BF16 operands widened to F32 first, and reaches the same rows.
- **Not this entry.** Causal LLM decode and prefill use `FlashAttention` (the generic transformer, Mllama, Qwen 3.5,
  LFM 3, MusicGen, the Moshi depformer, Dia self-attention, Gemma 2 with its softcap). They do not move.

## Workloads that move on every device

All-F32 operands, and either no F16 permission or a head dim the cooperative-matrix row does not take.

| Workload | Call site | Head dim | Bias | Tokens (where stated) | Weights on the 3060 box |
| --- | --- | --- | --- | --- | --- |
| BiRefNet matting, Swin-L windows (*checked*) | `Vision/Swin/SwinOps.cs:236` | 32 at every stage (embed 192, heads 6/12/24/48) | per head and window | 144 a window (window 12); 484 windows as the batch at stage 1 of a 1,024 px input | **yes**: `Models/Vision/BiRefNet-matting/model.safetensors` |
| Qwen3-4B text encoder, used by Flux.2 Klein 4B, Z-Image, Zeta-Chroma (*checked*) | `LlamaStyleEncoder.cs:1094` | 128 (`LlamaStyleEncoderConfig.cs:200`) | causal | prompt length; not stated | **yes**: `Models/text_encoders/qwen_3_4b.safetensors`, with `Stable-Diffusion/Flux2/flux-2-klein-4b.safetensors` and `VAE/Flux/flux2-vae.safetensors` |
| Other Llama-style text encoders at head dim 128 (Qwen3 0.6B/8B, Llama 3.1 8B, Mistral Small 3, Ministral 3B, Qwen2.5-VL) | same call | 128 | causal | prompt length | no |
| T5 / UMT5 encoder (Wan's text encoder among others) | `T5Block.cs:136` | 64 | relative-position bias | prompt length | **yes** for UMT5-XXL: `Models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors`, with the Wan 2.1 and 2.2 checkpoints |
| DINOv3 | `Dinov3Layer.cs:78` | hidden / heads, value not in source | none | not stated | **yes**: `Models/DINOv3/model.safetensors` |
| DINOv2 | `Dinov2VisionEncoder.cs:318` | hidden / heads, value not in source | none | not stated | **yes**: `Models/3D/sam3d/dinov2_vitl14_reg.safetensors` |
| SD 1.5 UNet, self and cross (*checked* `CrossAttentionBlock.cs:217`) | `CrossAttentionBlock.cs:281`, `:287` | 40 and 80 move; its 160 level stays on `naive` | none | latent grid: 4,096, 1,024, 256 at 512 px | no |
| CLIP text encoder | `ClipTextEncoder.cs:578` | hidden / heads, value not in source | causal | not stated | no |
| Qwen2.5-VL and Qwen3-VL vision towers | `Qwen25VlVisionEncoder.cs:446`, `Qwen3VlVisionEncoder.cs:328`; LLM `Qwen25VlEncoder.cs:228`, `Qwen3VlEncoder.cs:227` (*checked*) | 80 and 72 | 2.5: yes; 3: none | not stated | no |
| Whisper encoder and decoder | `WhisperEncoder.cs:308`, `WhisperDecoder.cs:404`, `:416` | hidden / heads, value not in source | causal on decoder self | not stated | no |
| Kokoro PL-BERT | `KokoroPlBert.cs:211` | value not in source | none | not stated | no |
| SigLIP | `SiglipVisionEncoder.cs:332`, `:202` | value not in source | none | not stated | no |
| SAM 2 (Hiera, decoder) and SAM 3 (text, ViT, segmentation head) | `HieraBlock.cs:184`, `DecoderAttention.cs:77`, `Sam3TextTransformerLayer.cs:91`, `Sam3VitBlock.cs:154`, `Sam3SegmentationHead.cs:122` | Hiera stage 1: 96, 56 or 72 by size | some | not stated | no |
| BERT | `BertModel.cs:120` | 64 | yes | not stated | no |
| SD3 joint and dual attention | `JointBlock.cs:189`, `:201` | 64 | none | not stated | no |
| Z-Image base | `ZImageBlock.cs:96` | 128 | yes | not stated | no |
| Lumina 2 | `Lumina2Block.cs:88`, `:295` | 96 | none | not stated | no |
| OmniGen2 table path | `OmniGen2Block.cs:380` | 120 | none | not stated | no |
| HunyuanImage | `HunyuanImageTransformer.cs:753` | 128 | none | not stated | no |
| AceStep DiT | `AceStepDit.cs:490` | 128 | none | not stated | no |
| MiniMax-H3 | `MiniMaxH3Transformer.cs:1028` | 128 | none | not stated | no |
| Dia cross-attention | `DiaAttention.cs:198` | 128 | yes | not stated | no |
| BiCodec | `BiCodec.cs:227` | value not in source | none | not stated | no |
| Moshi main transformer | `MoshiTransformer.cs:288` | 128 | yes | not stated | no |
| F5-TTS | `F5DitBlock.cs:111` | value not in source | none | not stated | no |
| CosyVoice decoder | `CausalConditionalDecoder.cs:353` | 64 | yes | not stated | no |

"No" in the last column means no file for it under `Models/` on the RTX 3060 box (listed 2026-11-21). The RTX 4090
host was not listed for this report.

Operand types not fully traced, so these rows are less certain: AceStep, MiniMax-H3, Moshi's cached keys and values,
Lumina 2 at batch 1.

Named by the first pass and **not re-verified**: Utonia, UniRig PTv3, TRELLIS.2 flow, CoLa3D, DeepSeek V4.1 vision,
MoonViT, Anima and its LLM adapter, the Lumina 2 context refiner, Lens, SanaWm (three sites), Mimi, Moonshine, HuBERT,
Demucs, PocketTTS, the Qwen3-TTS vocoder, RT-DETR, CLIPSeg, Cosmos AR, SAM-MoT, XPart, the Wan 2.2 VAE, IP-Adapter,
HiDream's cross branch, the LLM `T5Model`. Weights for Utonia, CoLa3D and TRELLIS.2 are on the 3060 box.

Corrections the second pass made to the first, which the PR text had repeated: MusicGen (flash entry), the SD-family
VAE's attention (one head of width 512, past every tiled bucket) and the Gemma 4 text encoder (head dims 256 and 512)
do **not** move; HairDeck refuses Vulkan outright; Dia, Moshi and OmniGen2 move only at the one call listed above.

## Workloads that move only without the cooperative-matrix kernel

F32 operands with the F16 permission at head dim 64 or 128. On an NVIDIA card these take `cm2.flash` and are unchanged;
on a device without `VK_NV_cooperative_matrix2` they move from `flash.scalar` to `tiled.f32`. Not re-verified call by
call: Wan (weights on the 3060 box), LTX-2, SDXL UNet, HiDream self-attention, Kandinsky 5, AuraFlow, Boogu, Ernie,
Hunyuan3D VAE and paint blocks (weights on the 3060 box), Trellis, TripoSR, MERT 2, HeartCodec, Matrix-Game 3, Moss
text, the Z-Image context refiner, F-Lite, Idol, and the F32 fallbacks of Flux, Flux.2 and Chroma. No such device was
available, so none of this was executed.

## Is there a biased call past 1 GiB of scores at head dims 129 to 256?

That is the one call the default gives the tiled kernel in that bucket. **Consumers of that shape exist; none reaches
1 GiB at the lengths the source states.**

| Consumer | Head dim | Bias | Tokens the source states | Scores, 16 heads, batch 1 |
| --- | --- | --- | --- | --- |
| Gemma 3 12B text encoder for LTX-2 (`LlamaStyleEncoder.cs:1094`, `LlamaStyleEncoderConfig.cs:433` *checked*, `LtxVideo2Recipe.cs:38`) | 256 | causal | 256 | 4 MiB |
| The same encoder for SanaWm (`SanaWmGemma3RefinerTextEncoder.cs:19`) | 256 | causal | 1,024 | 64 MiB |
| Gemma 4 text encoder, sliding layers, for LTX-2.5 (`Gemma4TextEncoder.cs:502`, `Gemma4Tokenizer.cs:21` *checked*) | 256 | yes | padded up to at least 1,024 | 64 MiB |

With 16 heads the scores pass 1 GiB only beyond 4,096 tokens. No prompt-length cap was found for either encoder (1,024
is a floor for Gemma 4, not a ceiling), and whether either runs at a batch above 1 was not traced. So under `auto`
these calls stay on `naive`, where they were; a prompt past 4,096 tokens would go to the tiled kernel instead of
holding two score buffers of more than 1 GiB each.

This also bears on the rule: these are real biased calls at head dim 256 that the former `auto` (now `all`) would have
moved to the kernel that is about half as fast there.

Others above head dim 128: SD 1.5's 160 level (no bias, at most 256 tokens at 512 px), the SD-family VAE at 512 and
Gemma 4's global layers at 512 (both past the tiled kernel), Gemma 2 2B at 256 (flash entry). SDXL, the Hunyuan3D
paint UNet and SAM 2 are at 128 or below.

## The smallest real run that exercises the moved row

Nothing below was run for this report, and nothing needs downloading on the RTX 3060 box.

1. **BiRefNet matting** is the smallest: one 844 MB file and one image in, a matte out.
   - File: `Models/Vision/BiRefNet-matting/model.safetensors` (884,878,856 bytes).
   - Every Swin block issues a biased, all-F32, head-dim-32 call with the windows as the batch, which is the moved row.
   - Command, once with the default and once with `--set numerics.vkTiledAttention=off`, same image:
     `hartsy vision <image.png> -m birefnet-matting --model-path Models/Vision/BiRefNet-matting/model.safetensors --mode matting -b vulkan`
   - Expectation to state before running: the two mattes agree to the last bit or within an 8-bit step (both kernels
     are exact F32, 1.7e-7 apart on synthetic calls), and the default is not slower.
   - Caveat: [model status](../../../docs/Checklists/MODEL_STATUS_VISION.md) records BiRefNet as verified on the CPU
     backend only. Whether it completes on Vulkan at all is not known; if it does not, that is a separate finding.
2. **Flux.2 Klein 4B** is the smallest image generation on the box whose path includes the moved row: its Qwen3-4B text
   encoder (36 layers, causal bias, head dim 128). Its transformer hands the layer F16 and does not move, so this
   checks the encoder's output and time, not the denoiser's.
   - Files: `Models/Stable-Diffusion/Flux2/flux-2-klein-4b.safetensors`, `Models/text_encoders/qwen_3_4b.safetensors`,
     `Models/VAE/Flux/flux2-vae.safetensors`.
   - Whether the 12 GB card runs it on Vulkan was not tried.
3. SD 1.5 would be the clearest generation (the UNet itself moves at head dims 40 and 80), but its weights are not on
   the box.
