# 3D weights: which files are linked, which are repacked and self-hosted

Inventory of every file the 3D models read, taken 2026-11-12 from the Hugging Face tree API (sizes, LFS sha256, revisions, licence
tags), the local `Models/3D` tree (sizes only, read-only), the GitHub release API and the licence texts. It decides, per file,
**link**, **repack-and-host** or **do-not-host**. Nothing was uploaded and no weights were downloaded for it. Not legal advice.

## Policy applied

[`PickleCheckpointRepacker`](../../src/HartsyInference.ModelAssets/PyTorch/PickleCheckpointRepacker.cs) states the mirror-vs-link
policy; this inventory applies it:

1. **link** when upstream is already safetensors (or a format the engine reads natively) and reasonably sized;
2. **link** to a trustworthy third-party safetensors re-upload with a verifiable hash in preference to a repack of our own;
3. **repack-and-host** only when upstream is pickle-only (or a trim is a real size win), and only when the licence lets us redistribute;
4. a hosted file always keeps its upstream URL as the fallback, so the mirror is never a hard dependency;
5. **do-not-host** when the repository is gated, the licence forbids it, or the licence is unknown.

A repack is a format change: tensor names, shapes, dtypes, order and values are unchanged (verified tensor for tensor, see
[Verification](#verification-the-repack-must-pass)). The Tencent licences treat a modified file as needing a notice (3b), so each
hosted file carries one.

## Licence terms that bind a host

| Licence | Files | Redistribution terms (from the licence text) |
|---|---|---|
| Tencent Hunyuan 3D 2.1 Community License (2025-06-13; read at `tencent/Hunyuan3D-2.1` `LICENSE`) | Paint 2.1, 2.1 shape | Territory is the world **excluding the EU, UK and South Korea** (1l). Distribution is "exclusively in the Territory" (3), with a copy of the licence for every recipient (3a), "prominent notices" on modified files (3b), a `Notice` text file for any distribution that is not a Hosted Service (3d), and the Acceptable Use Policy passed on (5a). More than 1 million monthly active users in the preceding month needs a licence from Tencent, and the request must describe plans to modify the Works (4). Outputs and results may not be used "to improve any other AI model" (5b). Use, reproduction, modification, distribution or display of the Works or Output outside the Territory is unlicensed (5c). Section 3e: a service built on the Works names its real provider and states that Tencent is not affiliated. Hong Kong law. |
| Tencent Hunyuan 3D Omni Community License (2025-09-26; read at `tencent/Hunyuan3D-Omni` `License.txt`) | Omni | The same clauses with the same section numbers (1l, 3, 3a, 3b, 3d, 3e, 4, 5a to 5c, 8b). |
| Tencent Hunyuan 3D 2.0 and 3D-Part licences | 2.0 family, Part | Same family (recorded in [HUNYUAN3D_ARCHITECTURE.md](HUNYUAN3D_ARCHITECTURE.md#licence-tencent-hunyuan-3d-20-community-license-agreement), [HUNYUAN3D_PART_ARCHITECTURE.md](HUNYUAN3D_PART_ARCHITECTURE.md#licence)). Nothing of these is hosted: their files are safetensors upstream. |
| MIT | TRELLIS.2, Pixal3D, UniRig, BiRefNet-matting | Keep the copyright and licence notice with the files. |
| Apache-2.0 | NAF (repository), DINOv2, HairDeck | Keep the licence and notices. The NAF release asset itself states no licence of its own. |
| SAM License (2025-11-19; read from `Models/3D/sam3d/SAM3D_LICENSE`) | SAM 3D Objects, SAM 3 | Distribution "may only [be done] under the terms of this Agreement" with a copy of it (1b i); not for Trade-Controls targets or military, nuclear, espionage or weapons end uses (1b v); suing Meta over the materials ends the licence and you indemnify Meta (5b). The Hugging Face repositories are `gated: manual`. |
| DINOv3 License (Meta, custom) | DINOv3 | Gated, manual approval. |
| CC BY-NC 4.0 | Utonia (`Pointcept/Utonia`), RMBG-2.0 (Bria licence, gated) | Non-commercial: not hosted. RMBG-2.0 is not used by the engine (BiRefNet-matting stands in). |

**What self-hosting a Tencent file therefore commits HartsyWeb to:** serve it only to the Territory (block the EU, UK and South
Korea at the edge), ship the licence, the `Notice` and a change notice beside it, pass the Acceptable Use Policy on in its terms,
and accept the 1M-user clause. The hosted Tencent set is 25.8 GB; the other hosted files (UniRig, NAF) carry no territory limit.
Linking to Hugging Face needs none of this, which is why only pickle-only files are hosted.

## Decision table

Sizes are bytes. "Repo @ revision" is the Hugging Face revision the sizes and hashes were read at. The original sha256 of a local
pickle is recorded by the repack run (see [Verification](#verification-the-repack-must-pass)); sizes of the local files equal the upstream sizes in every row below.

### Repack and host (pickle-only upstream)

| Id | Model | Upstream file | Size | Upstream sha256 | Licence | Local copy | Why host |
|---|---|---|---|---|---|---|---|
| `paint21-unet` | Hunyuan3D Paint 2.1 | `tencent/Hunyuan3D-2.1` @ `0b946776` `hunyuan3d-paintpbr-v2-1/unet/diffusion_pytorch_model.bin` | 3,925,293,863 | `675a1b5c…5b8` | Tencent 3D 2.1 | yes | only a pickle upstream |
| `paint21-vae` | Hunyuan3D Paint 2.1 | same repo `hunyuan3d-paintpbr-v2-1/vae/diffusion_pytorch_model.bin` | 334,707,217 | `1b4889b6…ddc` | Tencent 3D 2.1 | yes | only a pickle upstream (the 2.0 VAE safetensors is a different, 167 MB file) |
| `omni-dit` | Hunyuan3D-Omni | `tencent/Hunyuan3D-Omni` @ `70e803bf` `model/pytorch_model.bin` | 12,203,119,470 | `01c95e5f…a24` | Tencent 3D Omni | yes | only a pickle upstream; F32 |
| `omni-vae` | Hunyuan3D-Omni | same repo `vae/pytorch_model.bin` | 1,311,145,138 | `379995ca…340` | Tencent 3D Omni | yes | only a pickle upstream |
| `omni-cond` | Hunyuan3D-Omni | same repo `cond_encoder/pytorch_model.bin` | 464,828 | `f3f308e7…432` (catalog) | Tencent 3D Omni | yes | only a pickle upstream |
| `shape21-dit` | Hunyuan3D-2.1 shape (no catalog id yet) | `tencent/Hunyuan3D-2.1` @ `0b946776` `hunyuan3d-dit-v2-1/model.fp16.ckpt` | 7,366,389,768 | `6b519fc7…03f5e` | Tencent 3D 2.1 | yes (a repack already sits beside it) | only a pickle upstream; the bundle holds `model.*`, `conditioner.*`, `vae.*` |
| `shape21-vae` | Hunyuan3D-2.1 shape | same repo `hunyuan3d-vae-v2-1/model.fp16.ckpt` | 655,648,152 | `5cbe97f2…288a` | Tencent 3D 2.1 | yes | only a pickle upstream |
| `unirig-skeleton` | UniRig | `VAST-AI/UniRig` @ `36842e2b` `skeleton/articulation-xl_quantization_256/model.ckpt` | 1,439,617,174 | `d8cf9b42…0f4` | MIT (model card); see caveat | yes | only a pickle upstream (Lightning `.ckpt`) |
| `unirig-skin` | UniRig | same repo `skin/articulation-xl/model.ckpt` | 4,375,464,854 | `9d40cf42…c33` | MIT (model card); see caveat | yes | only a pickle upstream (Lightning `.ckpt`) |
| `naf` | Pixal3D (NAF upsampler) | `github.com/valeoai/NAF` release `model`, asset `naf_release.pth` (published 2025-11-24) | 2,664,431 | `c096c1ab…c98f` (the GitHub asset digest equals the catalog's pinned value) | repository Apache-2.0; the asset states none | yes | a pickle, and a GitHub release asset the catalog cannot fetch today |

Host-set total if every row goes up: 31,614,514,895 B (31.6 GB): 25,796,768,436 B Tencent, 5,815,082,028 B UniRig, 2,664,431 B NAF.

Caveats on three rows:

- **UniRig.** The code and the model card are MIT, but `src/model/michelangelo/*.py` carries a GPL-3.0 header and the dataset
  (`Seed3D/Articulation-XL2.0`) is CC BY 4.0 ([UNIRIG_ARCHITECTURE.md](UNIRIG_ARCHITECTURE.md#licenses-read-from-the-files)).
  The files are published by their authors under MIT; whether the GPL header reaches the weights is not settled here. Ship the
  MIT text and the dataset attribution with them.
- **NAF.** Apache-2.0 covers the repository; the release asset has no licence text. Pixal3D's own notes treat it as Apache-2.0
  "with that caveat". At 2.7 MB it is the cheapest file to host and the one with the weakest licence statement: the user decides.
- **Omni and the 2.1 DiT are large.** Omni's DiT is F32 (12.2 GB; the loader narrows it to F16 on read). Casting to F16 would halve
  the download but changes dtypes, so the repack keeps F32; a trim is a separate decision (see [Open decisions](#open-decisions)).

### Link (already safetensors upstream, or a safetensors third-party copy with a hash)

| Models | Files | Where | Licence |
|---|---|---|---|
| TRELLIS.2 | flows, decoders (`ckpts/*.safetensors`, 0.15 to 2.6 GB) | `microsoft/TRELLIS.2-4B` @ `af44b45f`, `microsoft/TRELLIS-image-large` @ `25e0d31f` | MIT |
| Pixal3D | three projection flows, texture flow, decoders (up to 5.5 GB each) | `TencentARC/Pixal3D` @ `b0cb2e1b` | MIT |
| Hunyuan3D-2, Fast, Turbo, mini, 2mv, FlashVDM VAE | `*/model.fp16.safetensors` (3.8 to 4.9 GB, 0.4 GB) | `tencent/Hunyuan3D-2` @ `9cd649ba`, `-2mini`, `-2mv` | Tencent 3D 2.0 |
| Hunyuan3D Paint 2.0 | UNet (3.66 GB) and VAE (167 MB) safetensors | `tencent/Hunyuan3D-2` @ `9cd649ba` | Tencent 3D 2.0 |
| Hunyuan3D-Part (P3-SAM, X-Part) | `p3sam`, `model`, `conditioner`, `shapevae` safetensors (0.45 to 6.6 GB) | `tencent/Hunyuan3D-Part` @ `67717446` | Tencent 3D-Part |
| SAM 3D Objects | the generators, decoders, MoGe, DINOv2 | `apozz/sam-3d-objects-safetensors` @ `cbb995af`: third-party conversion, sha256 pinned in the catalog (policy rule 2). Meta's own `facebook/sam-3d-objects` is gated pickle | SAM License (carry `SAM3D_LICENSE`) |
| HairDeck | image, colour, chain, card VAE models | `hyz317/HairDeck` @ `7d2e07c5` | Apache-2.0 |
| DINOv2-large / giant (Omni, Paint 2.1) | `model.safetensors` | `facebook/dinov2-large`, `facebook/dinov2-giant` | Apache-2.0 |
| BiRefNet-matting | `model.safetensors` (885 MB) | `ZhengPeng7/BiRefNet-matting` @ `eccde0a8` | MIT |
| Sonata (inside P3-SAM and the X-Part conditioner) | already bundled in the Part safetensors | `facebook/sonata` also lists `model.safetensors`; its weights licence is unstated | unverified |

### Do not host

| File | Reason |
|---|---|
| `facebook/dinov3-vitl16-pretrain-lvd1689m` `model.safetensors` | Gated by Meta with manual approval, custom DINOv3 licence. |
| `facebook/sam3` `sam3.pt` (HairDeck segmenter, 3.45 GB) | Gated (manual), SAM License; the user accepts it with their own account. |
| `facebook/sam-3d-objects` `checkpoints/*.ckpt` (local `Models/3D/sam3d-objects/*.ckpt`) | Gated (manual), pickles; the catalog links the safetensors conversion instead. |
| `Pointcept/Utonia` `utonia.pth` (CoLa3D's encoder, local `Models/Utonia`) | CC BY-NC 4.0: non-commercial. |
| `briaai/RMBG-2.0` | CC BY-NC (Bria licence), gated, and not used by the engine. |
| CoLa3D decomposers and flows (`cola3d_*_q8/q20.pt`, `ss_flow`, `slat_flow`, `geometry_decoder.pt`, local `Models/CoLa3D`) | No catalog entry, and the upstream repository and the weights' licence are not recorded in this repository. Revisit once both are known. |
| Real-ESRGAN `RealESRGAN_x4plus.pth` (67,040,989 B, opt-in Paint view upscaler, local `Models/3D/hunyuan3d-paint21`) | Not in the catalog and not downloaded with a model; the copy comes from `lllyasviel/Annotators`, whose licence tag is `other`. |
| `tencent/Hunyuan3D-Omni` `model/pytorch_model_ema.bin` (12.2 GB) | Unused by the engine. |
| TripoSR, TRELLIS (v1) | No catalog assets and no local files: nothing to inventory. |

## Verification the repack must pass

For each hosted file the repack run records, in a manifest beside the output: source repository, revision and path, source size and
sha256, the repacked size and sha256, tensor count, the licence and its text file, the date and the tool build. The repacked file
must equal its source tensor for tensor: the same names in the same order, the same shapes and dtypes, and bitwise equal values,
each side read through its own loader (the safe-subset pickle reader and the safetensors loader). The run never executes a pickle.

## Open decisions

1. **Territory.** Self-hosting the 25.8 GB of Tencent files needs an EU, UK and South Korea block on HartsyWeb. If the user does
   not want that, host only UniRig (and NAF) and leave the Tencent pickles linked and repacked locally on first use.
2. **Trim or cast.** Omni's DiT as F16 (6.1 GB) and the unused halves of the Omni VAE (the loader reads only the decoder) would cut
   the download. Not done: it changes dtypes or drops tensors.
3. **NAF's licence.** Host it under the repository's Apache-2.0 or keep it a GitHub-only fetch.
