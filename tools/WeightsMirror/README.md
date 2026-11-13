# Hosting the repacked 3D weights

Nothing here has been uploaded. This is the upload set, the checks that gate it and the exact steps, for the day the owner approves an upload.
Which files are hosted and why: [WEIGHTS_HOSTING_3D.md](../../docs/Research/WEIGHTS_HOSTING_3D.md).

The engine can fetch a safetensors repack of a pickle-only checkpoint from a weights mirror, and falls back to the upstream file on any failure (not found, wrong bytes,
timeout, a region block). The mirror is off until `paths.weightsMirrorUrl` names its base URL; no host is built into the engine.

## The staging folder

A folder outside the repository and outside `Models/` (`<staging>` below; the repack run made it on the maintainer's desktop). Layout: `<model id>/<the pickle's path with the extension .safetensors>`, which is
the `MirrorPath` the catalog lists for the file, plus `manifest.json` (sources, revisions, hashes, tensor counts, verification and licence of every file).

It also holds `upload-checklist.md` (per model: title, description, licence summary, tags, hashes and sizes, for entering the files in a web UI by hand) and
`covers/` (one image per model). They are written for the maintainer's upload and are not part of the repository.

| Path under the mirror base | Bytes | sha256 | Licence | A catalog asset lists it |
|---|---|---|---|---|
| `hunyuan3d-paint21/unet/diffusion_pytorch_model.safetensors` | 3,924,737,656 | `01736c5f5800528fd8487bcf01f541d29e86b6be04ff57800b7dd3f9206075a8` | Tencent 3D 2.1 | yes |
| `hunyuan3d-paint21/vae/diffusion_pytorch_model.safetensors` | 334,643,532 | `ed31a6bcc917e4da1b694281b7242f1246ffb5adba47eb81133d1bbae7a7ac3b` | Tencent 3D 2.1 | yes |
| `hunyuan3d-omni/model/pytorch_model.safetensors` | 12,203,113,408 | `1418d6af744d0f4a55a6b161c6f10f231dd9adf199c1267852b14ac09d720fcd` | Tencent 3D Omni | yes |
| `hunyuan3d-omni/vae/pytorch_model.safetensors` | 1,311,033,724 | `53c96d1691201da7d72823f3f4415500a6b7f32e7a5f48235d6192245145f5f6` | Tencent 3D Omni | yes |
| `hunyuan3d-omni/cond_encoder/pytorch_model.safetensors` | 463,840 | `a119c75aaca803e7567e8bcd06e32fe173ba5b582cd2c36ecddb001cae2bde03` | Tencent 3D Omni | yes |
| `pixal3d/naf_release.safetensors` | 2,654,296 | `383095c45e08e5d8025251254f068d6cd3215722f54df5389944afb0b2c1fba2` | Apache-2.0 (the NAF repository; the release asset states none) | yes |
| `hunyuan3d-21/dit/model.fp16.safetensors` | 7,365,944,442 | `88978c701cc5e6ac9b95acf1c2dc77c506a62c76309f70a4cc8f42c3dfb34a67` | Tencent 3D 2.1 | no (no catalog id yet) |
| `hunyuan3d-21/vae/model.fp16.safetensors` | 655,539,450 | `34c8c81898514408b2bb9c0fe345ecfc7a70f586b02a9321412b708a2b700c00` | Tencent 3D 2.1 | no (no catalog id yet) |
| `unirig/skeleton/articulation-xl_quantization_256/model.safetensors` | 1,440,605,624 | `7f89ca91db8394d9ccb5c3b2a9a8010978c7e40bd65759027b81fcc1a3dbba70` | MIT | no (no catalog assets yet) |
| `unirig/skin/articulation-xl/model.safetensors` | 1,458,138,468 | `789206ca282f4d813c6b367f24ef69ea67a88404bdf301c8edabecdfd93b50ef` | MIT | no (no catalog assets yet) |

The engine asks the mirror only for the six files a catalog asset lists (17,776,646,456 B; all but NAF are Tencent-licensed). The other four are staged and verified but
change nothing until a catalog entry downloads them. The full set is 28,696,874,440 B.

## Before anything is uploaded

1. **The files match the manifest.** `dotnet run --project tools/WeightsMirror -c Release -- check --staging <staging>` ends with `10 of 10 files match the manifest`.
2. **The change notices are written.** `... -- notices --staging <staging>` writes `<model id>/CHANGES.txt` (what was converted, from which revision, that the tensor values
   are unchanged): the Tencent licences require a prominent notice on a modified file (3b).
3. **The licence texts are beside the files.** The Tencent licences also need a copy of the licence and a `Notice` file with every distribution (3a, 3d); MIT and Apache-2.0
   need their text. The six files (80,701 bytes) were fetched with the owner's approval on 2026-11-13 and sit under
   `<staging>/<model id>/<file name>` (the 2.1 files in both `hunyuan3d-paint21` and `hunyuan3d-21`). To redo it, fetch each from its source:

   | File | Source | Bytes | git blob SHA-1 |
   |---|---|---|---|
   | `LICENSE` | `https://huggingface.co/tencent/Hunyuan3D-2.1/raw/0b94677654c57bb9a6b6845cd7b704ccf551d327/LICENSE` | 17,915 | `d875bf81e661c42846be26b83417ea31bfc4d3a7` |
   | `Notice.txt` | `https://huggingface.co/tencent/Hunyuan3D-2.1/raw/0b94677654c57bb9a6b6845cd7b704ccf551d327/Notice.txt` | 16,644 | `bd3dea50102cdd5628a7a803c5d0c68a522a29ff` |
   | `License.txt` | `https://huggingface.co/tencent/Hunyuan3D-Omni/raw/70e803bfb4e127d534049d8ab8c8cb511780d485/License.txt` | 16,966 | `5674ec7220abe6787d0f3c7e56945b4e7bb7715f` |
   | `Notice.txt` (Omni) | `https://huggingface.co/tencent/Hunyuan3D-Omni/raw/70e803bfb4e127d534049d8ab8c8cb511780d485/Notice.txt` | 16,695 | `cbecd5c4252c0c7d66872de370467184181bb1c6` |
   | `LICENSE` (NAF) | `https://raw.githubusercontent.com/valeoai/NAF/4a73dfbe8851d44a7c62a6c6aba2a59fd9ebcf87/LICENSE` | 11,357 | `7a4a3ea2424c09fbe48d455aed1eaa94d9124835` |
   | `LICENSE` (UniRig) | `https://raw.githubusercontent.com/VAST-AI-Research/UniRig/cf7622f2be1de5ae315478ed8d21681b95dc9ff3/LICENSE` | 1,124 | `a20d482774136ee1858d92964e896031598902b4` |

   `... -- licenses --staging <staging>` compares each file's git blob SHA-1 with the one above, lists what is missing, and records each verified file's SHA-256 and the
   time it was first verified in `manifest.json` next to its source URL, for the upload record (it ends with `All 8 licence files are in place`; a rerun of `run` keeps the record).
   The two GitHub files are pinned to the last commit that changed them, so their blob hashes are reproducible.
4. **The terms are decided** (next section).

## Territory and terms

- The Tencent Hunyuan 3D 2.1 and Omni licences allow distribution only inside the Territory: the world excluding the EU, the UK and South Korea. A host that serves these files
  has to refuse those three regions (for example HTTP 451 by country at the edge). The engine treats any error as a reason to use the upstream file, which Hugging Face serves under its own terms.
- The licences also bind the host to pass the Acceptable Use Policy on, to name itself as the provider of a service built on the Works without implying Tencent's endorsement (3e),
  to get a licence from Tencent above 1 million monthly active users (4), and to keep Output from improving another AI model (5b).
- UniRig is MIT; credit the Articulation-XL2.0 dataset (CC BY 4.0). Whether the GPL-3.0 header on UniRig's Michelangelo code reaches the weights is not settled.
- NAF's repository is Apache-2.0; its release asset states no licence.

## Server requirements

HTTPS; a plain `GET` with a `Content-Length`; the paths above relative to the base URL; no authentication and no query string in the base URL; no redirect to plain HTTP; no size limit at a
proxy (the Omni DiT is 12.2 GB); for the Tencent files, a region block.

## Upload

The host is not known, so these are templates. Nothing is run until the owner approves the list.

```
rsync -av --partial --checksum "<staging>/hunyuan3d-paint21/" <user>@<host>:<web root>/<mirror path>/hunyuan3d-paint21/
```

Repeat for each model folder in the approved list; each folder carries its licence and notice files and its `CHANGES.txt`. Then, from another machine, compare what is served with the manifest for each file
(PowerShell; use `curl.exe`, since `curl` is an alias there; on bash use `curl` and `sha256sum check.bin`):

```
curl.exe -sIL https://<host>/<mirror path>/<path>                              # a final 200 and the Content-Length above
curl.exe -sL https://<host>/<mirror path>/<path> -o check.bin
(Get-FileHash check.bin -Algorithm SHA256).Hash                                # equals the sha256 above, case aside
```

## Wire it in

```
hartsy settings set paths.weightsMirrorUrl https://<host>/<mirror path>
```

Unset it, or set it empty, to turn the mirror off; nothing else changes. A file already on disk, as the pickle or as its repack, is never fetched again. To try a mirror on this machine
first, serve the staging folder over `http://localhost:<port>` (plain http is accepted for this machine only) and point the setting there; every file is hash-checked on arrival.

## Rollback

Delete the files from the server or unset the setting. The engine uses the upstream file whenever the mirror does not answer or serves bytes that miss the pinned hash.

## Redoing the repack

`run --plan tools/WeightsMirror/mirror-plan.json --models-root <Models> --staging <staging>` repacks from the local pickles (it refuses a source whose sha256 is not the upstream pin);
`verify` checks the staged repacks against their sources again; `annotate` records a check made elsewhere. The repack of the four smaller files (NAF, the Omni control encoder, the Paint 2.1 VAE, the 2.1 shape VAE) was run a
second time into a scratch folder and gave the same sha256; the larger ones were not re-run. The unit tests of the repack are `StreamingPickleRepackTests` and `StreamingPickleRepackLayoutTests`.
