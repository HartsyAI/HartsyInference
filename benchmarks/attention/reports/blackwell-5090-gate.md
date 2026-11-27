# Blackwell gate: what ran on an RTX 5090 and what did not

The plan makes a consumer Blackwell card the final CUDA and Vulkan gate. This records the four passes that ran tests
on a real RTX 5090, each at the same commit and each on a different card. They are numbered 1, 2, 3 and 5; no results
for a pass 4 are recorded.

**The gate is partly met.**

- **CUDA without cuDNN (passes 1 and 2):** the per-project counts give 42 failed tests in the project runs (first
  recorded as 45, unreconciled). They include 11 that route through cuDNN, which the pods did not have.
- **CUDA with cuDNN (pass 5):** those 11 pass. 31 tests still failed in the project runs: 21 Vulkan tests that had only
  a software device, 8 with missing checkpoints, the Core preload test (out of VRAM) and the Kimi packed-expert CUDA
  test, which is unexplained.
- **A fault on the third card (pass 3):** `CUDA_ERROR_ILLEGAL_ADDRESS`, cause not established. The other three passes
  did not show it.
- **Vulkan did not run on Blackwell hardware.** No pass saw a usable NVIDIA Vulkan device.
- **Stages 3 and 4** of the campaign ([blackwell-rental-campaign.md](blackwell-rental-campaign.md)) were not run.

## What was run

| | |
| --- | --- |
| Commit | `f8b9d4b4b` (the `main` tip at the time), checked out detached on the pod, in all four passes |
| Card | NVIDIA GeForce RTX 5090, 32,607 MiB, compute capability 12.0, driver 595.112.07, secure cloud, $1.39/hr (recorded for passes 1 and 2) |
| Image | `kalebbroo/hartsy-gpu-runner:cuda13.0-ssh`, .NET SDK 10.0.404, cuBLAS present (4 files), cuDNN absent as shipped (0 files) |
| Stages | the LLM GPU lane as a probe (`GpuIntegration`), whole Core, Audio, Diffusion, Vision, LLM and ThreeD projects, Engine PSA tests, then the LLM `GpuBenchmark` timing tests (recorded for passes 1 and 2) |

| Pass | Pod | Card | Time | Cost | cuDNN | Illegal address |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `3tbjjlz7h629tx` | first | 12 min of tests, pod stopped at 14 min | about $0.33 | absent | none |
| 2 | `jfmbvx1fn0frpe` | second | 11 min of tests, pod stopped at 13 min | about $0.32 | absent | none |
| 3 | `h5isulyrqg4d1o` | third, GPU UUID `3cb5d6d5-...` | 27 min of tests | about $0.72 | downloaded, not installed | yes |
| 5 | `z4enbbuqqhoj2g` | fourth, GPU UUID `0919cf68-...`, bus 21:00.0 | 14 min of tests | about $0.38 | installed | none |

Passes 1 to 3 ran on 2026-11-22 and 2026-11-23. The date of pass 5 was not recorded. The commits of pull request #1062
bound it: the revision of 2026-11-23 21:36 UTC says no attempt after pass 3 had reached a card, and the revision of
2026-11-24 01:12 UTC records pass 5.

Pass 2 differs from pass 1 only in also installing `libxext6` and `libx11-6` before the Vulkan check. Passes 1 and 2
both copied their results back with a matching sha256, no test host crashed and no stage hit its time budget.

## Results

Counts for passes 1 and 2 are from the TRX files. The RTX 3060 column is the whole-project run on Windows at the
earlier commit `4b5f9d02c`; no failure was recorded there.

| Project | Listed | RTX 3060 passed / failed | Passes 1 and 2: ran | Passes 1 and 2: passed / failed | Pass 3: failed | Pass 5: passed / failed |
| --- | --- | --- | --- | --- | --- | --- |
| Core | 472 | 472 / 0 | 472 | 471 / 1 | 2 | 471 / 1 |
| Audio | 155 | 153 / 0 | 153 | 153 / 0 | 6 | 153 / 0 |
| Diffusion | 406 | 396 / 0 | 396 | 396 / 0 | 2 | 396 / 0 |
| Vision | 74 | 70 / 0 | 70 | 70 / 0 | not recorded | 70 / 0 |
| LLM (whole) | 849 | 815 / 0 | 815 | 782 / 33 | 81 | 793 / 22 |
| ThreeD | 1,466 ran of 1,457 listed | 1,435 / 0 | 1,425 | 1,417 / 8 | 25 | 1,417 / 8 |
| Engine PSA attention | 2 | not run | 2 | 2 / 0 | 1 | 2 / 0 |
| LLM GPU lane (probe) | not recorded | not recorded | not recorded | 24 failed | 40 | not recorded |

The ThreeD figures do not reconcile: 1,466 ran against 1,457 listed, without saying which run the 1,466 belongs to, and
passes 1 and 2 ran 1,425 where the RTX 3060 passed 1,435. Neither difference is explained. Audio, Diffusion, Vision and
LLM ran fewer tests than were listed, matching the RTX 3060 pass counts; the reasons are not recorded.

In passes 1 and 2 the Trellis2 attention test passed at every shape, including 14,748 tokens.

### Failures in the project runs, by cause

| Failed | Passes 1 and 2 | Pass 5 | Cause | Counts against Blackwell? |
| --- | --- | --- | --- | --- |
| UniRig 4 and CoLa3D 4 (ThreeD) | 8 | 8 | missing checkpoint files; the pod has no weights | no, expected from the 4090 run |
| Core `PreloadUnderPressureGpuTests` | 1 | 1 | out of VRAM; in passes 1 and 2 `OutOfVramException` with 81 MB of 32,110 MB free after the preload filled the card, the same kind of failure as on the Linux 4090 | no, recorded as a Linux benchmark-lane fact |
| LLM `Kimi.PackedMoeTests.Cuda_PackedExperts_MatchTheF32Dequantization` | 1 | 1 | in passes 1 and 2 -0.237698555 expected, -0.237098411 actual, outside 3 decimals; fails on the 4090 too and passes on the 3060 | recorded, not explained |
| Vulkan tests in the whole-LLM run (route and A/B tests) | 21 | 21 | the only Vulkan device was `llvmpipe`, a CPU rasteriser with no cooperative matrices; in passes 1 and 2 they assert tiled rows that a software device does not take. The pass 5 count is its 22 LLM failures less the Kimi test | no, a pod problem; not evidence about the Vulkan code |
| CUDA tests that route through cuDNN | 11 | 0 | cuDNN was not in the image and its download was refused by the closed proxy, so each took a fallback row and failed its `cudnn.*` row assertion | no; all 11 pass in pass 5 with cuDNN installed |
| **Total** | **42** | **31** | | |

Passes 1 and 2 failed the same test lines (compared by name; the lists are identical). Their number was first recorded
as 45, which does not match the 42 above. The 9 failed timing tests, all of the Vulkan A/B kind and giving no timing,
are outside this table (51 if none repeats a project-run failure), and so are the GPU lane's 24. How these sets
overlap, and how 45 was counted, is not recorded. The CUDA timing rows ran in passes 1 and 2; they were not compared
across cards.

## Pass 3: an illegal address on the third card

`CUDA_ERROR_ILLEGAL_ADDRESS` (error 700) was recorded as appearing in every test process: first in
`SolAttentionGpuTests.OnCuda...` of the GPU lane, then in Core (`SdpaTiledGpuTests`, tile height 512), Audio, ThreeD,
Engine PSA and the whole LLM run. After the first fault in a process, every later CUDA call in it fails. That list does
not name Diffusion (2 failures in this pass) or Vision (no count), so the record does not show the error in those two.

The cause is **not established**. It is either a faulty card or host, or an out-of-bounds access in a kernel that only
some runs trigger. The fault appeared on one of the four cards. Its absence in pass 5 fits a faulty card or host for
pass 3; it does not prove it, because the fault was not reproduced on the third card either. No memory-checker result is
recorded: pass 3 carried none, and the attempts that followed it, set up to run `compute-sanitizer` on the Sol attention
tests if the fault recurred, had not reached a card when pass 3 was written up.

## cuDNN and downloads

| When | cuDNN | Downloads recorded |
| --- | --- | --- |
| Pass 1 | absent; 11 tests failed on a fallback row | none; test processes ran with the HTTP proxy variables pointed at a closed local port |
| Pass 2 | absent; the same 11 tests | the `libxext6` and `libx11-6` packages (size not recorded); same proxy setting for the test processes |
| Pass 3 | 0 files installed, because the extraction step matched nothing | cuDNN 9.21.0.82 for CUDA 13 (915,264,756 bytes, sha256 and size matching NVIDIA's published manifest) |
| Pass 5 | installed: 27 files, `libcudnn.so.9` present; the 11 tests pass | cuDNN, sha256 and size matching NVIDIA's manifest (version and byte count not recorded for this pass) |
| Vulkan diagnosis | not applicable | Ubuntu's libvulkan1 1.4.341 (157,194 bytes) |

In passes 1 and 2, `/root` grew from 294 MB to 1,754 MB, all of it the build. The extraction step that failed in pass 3
was fixed before cuDNN installed in pass 5. The passing cuDNN rows rest on that one pass, on one card.

## Vulkan

Before the first pod, the chance of an NVIDIA Vulkan device being visible was put at about 60%. None became usable.

| Pass | What the pod showed |
| --- | --- |
| 1 | the loader logged `libXext.so.6: cannot open shared object file` while loading `libGLX_nvidia.so.0` |
| 2 | with that library installed, the message changed to `Could not get 'vkCreateInstance' via 'vk_icdGetInstanceProcAddr' for ICD libGLX_nvidia.so.0` |
| 3 | the full graphics library set (`libnvidia-glvkspirv`, `-eglcore`, `-glcore`, `-gpucomp`, `-rtcore`) was present and `ldd` of `libGLX_nvidia` showed no missing library; the same `vkCreateInstance` error, so the library set is not the cause |
| 5 | only `llvmpipe` visible. The NVIDIA ICD manifest was present and the loader opened `libGLX_nvidia.so.595.112.07` (strace), but `vkCreateInstance` could not be resolved through `vk_icdGetInstanceProcAddr`. strace does not show the symbol lookup |

In passes 1 and 2, `NVIDIA_DRIVER_CAPABILITIES=all` was set when the pod was created and was unset in the SSH shell.

### Diagnosis, 2026-11-24

Vulkan had not run on NVIDIA hardware on any RunPod pod tried. The diagnosis counts two 5090 and two 4090 passes plus
diagnostic pods. This file describes four 5090 test passes, and which two were counted is not recorded; the 4090 passes
and the diagnostic pods have no other record here. Tested and ruled out as the cause:

- **Loader version:** the image ships Ubuntu's libvulkan1 1.3.275. Ubuntu's own libvulkan1 1.4.341 (archive sha256
  `b4aca7a5...`) was used in place of it, and the loader log shows 1.4.341 in use. The same failure follows.
- **Missing entry points:** both the injected `libGLX_nvidia.so.0` and the NVIDIA installer's
  `libGLX_nvidia.so.595.112.07` export `vkCreateInstance`, `vk_icdGetInstanceProcAddr` and
  `vk_icdNegotiateLoaderICDInterfaceVersion`.
- **Environment:** with `NVIDIA_DRIVER_CAPABILITIES=all` and `NVIDIA_VISIBLE_DEVICES=all` set explicitly, both ICDs fail
  the same way. The container's startup environment has `NVIDIA_DRIVER_CAPABILITIES=all` and
  `NVIDIA_VISIBLE_DEVICES=void`.

The pass 5 record said the installer's driver userspace (423 MB, sha256 published) was the only untested remedy and had
not been downloaded. The diagnosis then uses the installer's library, but records no download of it or of the installer.

What remains: the ICD reports that `vk_icdGetInstanceProcAddr` cannot return `vkCreateInstance`, and strace shows it
never opens `/dev/nvidiactl` or a DRM node before failing. The cause inside the driver is not identified, and the image,
loader and environment tests do not decide it. A different image was not tried. It would not change the injected driver
libraries, so an image rebuild alone is not expected to help.

The rental campaign says that if the device is still missing with `NVIDIA_DRIVER_CAPABILITIES=all` set, the finding is
that the Vulkan half cannot be run at this provider. This is that finding, for this image.

## Departure from the campaign plan

The campaign stops after stage 1 if no NVIDIA Vulkan device appears with the graphics capability requested. Pass 1 found
none, yet it ran the whole-project suites, and three more test passes followed. No reason is recorded. The campaign also
quotes stage 1 at no more than $0.70; the recorded spend is above that.

## What this leaves unvalidated

- Every Vulkan attention row on Blackwell hardware (tiled F32, flash, cooperative matrix, decode), and the Vulkan timing
  tests.
- The cause of the pass 3 `CUDA_ERROR_ILLEGAL_ADDRESS`.
- Stage 3 (frozen Kitchen operation corpus and comparison) and stage 4 (Qwen and Krea primaries), which need weights
  and, for the corpus, the runner that was removed from the repository.
- The Kimi packed-expert result, which differs on both Linux cards.

## Pod cost and incidents

| Item | Recorded cost (approximate) |
| --- | --- |
| Test passes 1, 2, 3 and 5 | $0.33 + $0.32 + $0.72 + $0.38 = $1.75 |
| Two RTX PRO 4500 Blackwell pods (EU-CZ-1, $0.72/hr) and four RTX 5090 pods that never answered SSH and ran no test | $1.50 "in all", recorded after pass 3 |
| Total recorded after pass 3, before pass 5 | $3.27 |
| The 4090 passes and diagnostic pods of the Vulkan diagnosis | not recorded |

These figures do not reconcile, so no total for the campaign is given. The $3.27 was recorded with passes 1 to 3 at
$1.37, but $1.37 + $1.50 is $2.87, which is $0.40 less. $0.40 is what the first revision of this report gave for the two
PRO 4500 pods (beside $0.19 for the one 5090 pod then counted, in a total of $1.24), so the $1.50 may leave them out;
nothing recorded says so. With pass 5, $1.75 + $1.50 is $3.25, while $3.27 + $0.38 is $3.65. Both are below the
campaign's $9.73 ceiling, and neither includes the unrecorded costs.

One of the unanswered pods ran about 40 minutes before it was stopped, because the queue answer that named it did not
parse and its id was lost. Several requests for a 5090 expired in the queue over about a day before one deployed. When
pass 3 was written up, every pod was stopped, none had been terminated and no volume had been touched; no later check of
pod state is recorded here.
