# 3D weights: which files are linked, which are repacked and self-hosted

Inventory of every file the 3D models read, taken 2026-11-12 from the Hugging Face tree API (sizes, LFS sha256, revisions, licence
tags), the local `Models/3D` tree (sizes only, read-only), the GitHub release API and the licence texts. It decides, per file,
**link**, **repack-and-host** or **do-not-host**. Nothing was uploaded and no weights were downloaded for it. Not legal advice.

## Policy applied

[`PickleCheckpointRepacker`](../../src/HartsyInference.ModelAssets/PyTorch/PickleCheckpointRepacker.cs) states a four-rule
mirror-vs-link policy; this inventory applies it and extends it with the licence condition in rule 3 and the do-not-host rule 5:

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
Linking to Hugging Face avoids the distribution terms (3, 3a, 3b, 3d, 5a), which is why only pickle-only files are hosted. A service that
runs these models still carries 3e, 4 and 5b/5c whichever way the bytes arrive.

## Decision table

Sizes are bytes. "Repo @ revision" is the Hugging Face revision the sizes and hashes were read at; the hashes are the LFS oids of the tree API and equal the catalog's pins where the catalog lists the file. The original sha256 of a local
pickle is recorded by the repack run (see [Verification](#verification-the-repack-must-pass)); sizes of the local files equal the upstream sizes in every row below.

### Repack and host (pickle-only upstream)

| Id | Model | Upstream file | Size | Upstream sha256 | Licence | Local copy | Why host |
|---|---|---|---|---|---|---|---|
| `paint21-unet` | Hunyuan3D Paint 2.1 | `tencent/Hunyuan3D-2.1` @ `0b946776` `hunyuan3d-paintpbr-v2-1/unet/diffusion_pytorch_model.bin` | 3,925,293,863 | `675a1b5cd0098b2002637c443946529c03c5cd54427f40245263350feb3dd5b8` | Tencent 3D 2.1 | yes | only a pickle upstream |
| `paint21-vae` | Hunyuan3D Paint 2.1 | same repo `hunyuan3d-paintpbr-v2-1/vae/diffusion_pytorch_model.bin` | 334,707,217 | `1b4889b6b1d4ce7ae320a02dedaeff1780ad77d415ea0d744b476155c6377ddc` | Tencent 3D 2.1 | yes | only a pickle upstream (the 2.0 VAE safetensors is a different, 167 MB file) |
| `omni-dit` | Hunyuan3D-Omni | `tencent/Hunyuan3D-Omni` @ `70e803bf` `model/pytorch_model.bin` | 12,203,119,470 | `01c95e5f2588e324ebc05268231c92b3b1d9e83e5163723100ac958226c12a24` | Tencent 3D Omni | yes | only a pickle upstream; F32 |
| `omni-vae` | Hunyuan3D-Omni | same repo `vae/pytorch_model.bin` | 1,311,145,138 | `379995ca170d8a899019125f389ba8692b2e35625ff64ddc3fdaa8c9302ac340` | Tencent 3D Omni | yes | only a pickle upstream |
| `omni-cond` | Hunyuan3D-Omni | same repo `cond_encoder/pytorch_model.bin` | 464,828 | `f3f308e73153000ed566e985e2b1e38e9b40c12e23b7e6ecf89ce3ea1699c432` (not LFS; the catalog's hash) | Tencent 3D Omni | yes | only a pickle upstream |
| `shape21-dit` | Hunyuan3D-2.1 shape (no catalog id yet) | `tencent/Hunyuan3D-2.1` @ `0b946776` `hunyuan3d-dit-v2-1/model.fp16.ckpt` | 7,366,389,768 | `6b519fc7242f78e9b5f47ea4d55668fe3d944a2d27332f4ca68d29a6ff603f5e` | Tencent 3D 2.1 | yes (a repack already sits beside it) | only a pickle upstream; the bundle holds `model.*`, `conditioner.*`, `vae.*` |
| `shape21-vae` | Hunyuan3D-2.1 shape | same repo `hunyuan3d-vae-v2-1/model.fp16.ckpt` | 655,648,152 | `5cbe97f25e6e7abd4bccc80ab07524ec0c86d24118486a9ba49bb5dfb070288a` | Tencent 3D 2.1 | yes | only a pickle upstream |
| `unirig-skeleton` | UniRig | `VAST-AI/UniRig` @ `36842e2b` `skeleton/articulation-xl_quantization_256/model.ckpt` | 1,439,617,174 | `d8cf9b42d56e7bc316d293597ecf8c4c39a8631cdd44a261ecf388242fabd0f4` | MIT (model card); see caveat | yes | only a pickle upstream (Lightning `.ckpt`) |
| `unirig-skin` | UniRig | same repo `skin/articulation-xl/model.ckpt` | 4,375,464,854 | `9d40cf42fb9d4c10d8b373d9f4f557c6fbbc5ebacae7ae6b3dce5a0d8a18bc33` | MIT (model card); see caveat | yes | only a pickle upstream (Lightning `.ckpt`) |
| `naf` | Pixal3D (NAF upsampler) | `github.com/valeoai/NAF` release `model`, asset `naf_release.pth` (published 2025-11-24) | 2,664,431 | `c096c1ab2217a5c3ac136365f721685e2201379cb69d509cfb0261183847c98f` (the GitHub asset digest equals the catalog's pinned value) | repository Apache-2.0; the asset states none | yes | a pickle, and a GitHub release asset the catalog cannot fetch today |

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
| Hunyuan3D-2, Fast, Turbo, mini, 2mv, FlashVDM VAE | `*/model.fp16.safetensors` (3.8 to 4.9 GB, 0.4 GB) | `tencent/Hunyuan3D-2` @ `9cd649ba`, `tencent/Hunyuan3D-2mini` @ `f90a0f7d`, `tencent/Hunyuan3D-2mv` @ `3a761b53` | Tencent 3D 2.0 |
| Hunyuan3D Paint 2.0 | UNet (3.66 GB) and VAE (167 MB) safetensors | `tencent/Hunyuan3D-2` @ `9cd649ba` | Tencent 3D 2.0 |
| Hunyuan3D-Part (P3-SAM, X-Part) | `p3sam`, `model`, `conditioner`, `shapevae` safetensors (0.45 to 6.6 GB) | `tencent/Hunyuan3D-Part` @ `67717446` | Tencent 3D-Part |
| SAM 3D Objects | the generators, decoders, MoGe, DINOv2 | `apozz/sam-3d-objects-safetensors` @ `cbb995af`: third-party conversion, sha256 pinned in the catalog (policy rule 2). Meta's own `facebook/sam-3d-objects` is gated pickle | SAM License (carry `SAM3D_LICENSE`) |
| HairDeck | image, colour, chain, card VAE models | `hyz317/HairDeck` @ `7d2e07c5` | Apache-2.0 |
| DINOv2-large / giant (Omni, Paint 2.1) | `model.safetensors` | `facebook/dinov2-large` @ `47b73eef` (the catalog's pin), `facebook/dinov2-giant` (no revision pinned here) | Apache-2.0 |
| BiRefNet-matting | `model.safetensors` (885 MB) | `ZhengPeng7/BiRefNet-matting` @ `eccde0a8` | MIT |
| Sonata (inside P3-SAM and the X-Part conditioner) | already bundled in the Part safetensors | `facebook/sonata` also lists `model.safetensors`; the model card states no licence, the `facebookresearch/sonata` README says the code is Apache-2.0 and the weights CC-BY-NC 4.0 (restricted by the NC terms of HM3D and ArkitScenes), read 2026-11-12 | CC-BY-NC 4.0 (non-commercial); do not host |

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

## Repack run and the checks it passed (2026-11-12)

`tools/WeightsMirror` repacked the ten files of the table above into a staging folder outside the repository and outside `Models/`
(28,696,874,440 B against 31,614,514,895 B of pickles), after hashing each local pickle to its upstream pin. It records in
`manifest.json`: source repository, revision and path, source size and sha256, the repacked size and sha256, tensor count and dtypes,
the licence with its text files and what a host owes, the verification and the date. The run never executes a pickle.

Each repack equals its source tensor for tensor (names in the same order, shapes, dtypes, bitwise values). The safetensors side was
read through the safetensors loader. The pickle side was read through the whole-file safe-subset reader for the eight sources up to
5 GB and the tensor-by-tensor reader (the one that wrote the repack) for the two largest, and 9 of 10 were also compared through
PyTorch (`torch.load(weights_only=True, mmap=True)`); PyTorch refuses the UniRig skin checkpoint, which holds non-tensor Lightning
objects, so that one has the whole-file reader check only. The 2.1 shape DiT also equals the repack that already sat beside its
pickle in `Models/`. The UniRig skin repack is a third of its pickle's size because the pickle reader exposes only the state tensors.

The steps to upload, the licence files still to be fetched and the terms a host accepts are in
[tools/WeightsMirror/README.md](../../tools/WeightsMirror/README.md). Nothing is uploaded.

## Open decisions

1. **Territory.** Self-hosting the 25.8 GB of Tencent files needs an EU, UK and South Korea block on HartsyWeb. If the user does
   not want that, host only UniRig (and NAF) and leave the Tencent pickles linked: the 3D loaders read them in place through the
   safe parser (no 3D path calls `PickleCheckpointRepacker`; its callers are audio), which is today's behaviour.
2. **Consumer-side work (done, #871).** `ModelAsset.Repack` lists a mirror path and the repack's own hash, the upstream file stays the
   fallback, the mirror is off until `paths.weightsMirrorUrl` is set, and the Omni, UniRig and NAF loaders (Paint 2.1's already did)
   read a repack beside the pickle. NAF is fetched from its GitHub release by `ModelAsset.Url`. What remains: UniRig and the 2.1 shape
   model have no catalog assets, so their repacks are read when placed but nothing fetches them or asks the mirror for them.
3. **Trim or cast.** Omni's DiT as F16 (6.1 GB) and the unused halves of the Omni VAE (the loader reads only the decoder) would cut
   the download. Not done: it changes dtypes or drops tensors.
4. **NAF's and UniRig's licences.** NAF: host it under the repository's Apache-2.0 or keep it a GitHub-only fetch. UniRig: whether the
   GPL-3.0 header on its Michelangelo code reaches the weights is unsettled (see the caveat above); host under the published MIT or link.
