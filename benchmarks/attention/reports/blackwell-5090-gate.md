# Blackwell gate: what ran on an RTX 5090 and what did not

The plan makes a consumer Blackwell card the final CUDA and Vulkan gate. This records the three runs that reached a
real RTX 5090 on 2026-11-22 and 2026-11-23. **The gate is partly met**: on two cards the CUDA lane ran on Blackwell
and passed everywhere except where the pod lacked cuDNN; on a third card the same commit raised
`CUDA_ERROR_ILLEGAL_ADDRESS` in every test process (see the third pass), which is unexplained. The **Vulkan half did not run on Blackwell**, because the NVIDIA
Vulkan driver never became visible in the pod. Stages 3 and 4 of the campaign
([blackwell-rental-campaign.md](blackwell-rental-campaign.md)) were not run.

Expectations were written before the first pod existed and are not edited here; where they were wrong, this says so.

## What was run

| | |
| --- | --- |
| Commit | `f8b9d4b4b` (the `main` tip at the time), checked out detached on the pod |
| Card | NVIDIA GeForce RTX 5090, 32,607 MiB, compute capability 12.0, driver 595.112.07, secure cloud, $1.39/hr |
| Image | `kalebbroo/hartsy-gpu-runner:cuda13.0-ssh`, .NET SDK 10.0.404, cuBLAS present (4 files), cuDNN absent (0 files) |
| Stages | probe (LLM `GpuIntegration`), whole Core, Audio, Diffusion, Vision, LLM, ThreeD projects, Engine PSA tests, then the LLM `GpuBenchmark` timing tests |
| Downloads | none: test processes ran with the HTTP proxy variables pointed at a closed local port; `/root` grew from 294 MB to 1,754 MB, all of it the build |
| Runs | pass 1 (pod `3tbjjlz7h629tx`): 12 min of tests, pod stopped at 14 min, about $0.33. Pass 2 (pod `jfmbvx1fn0frpe`): 11 min of tests, pod stopped at 13 min, about $0.32 |

Pass 2 differs from pass 1 only in that the pod script also installs `libxext6` and `libx11-6` before the Vulkan
check (see Vulkan below). Both runs copied their results back with a matching sha256. No test host crashed and no stage
hit its budget in either run.

## Results

Counts are from the TRX files. The RTX 3060 column is the whole-project run on Windows at the earlier commit
`4b5f9d02c`; no failure was recorded there.

| Project | Listed | RTX 3060 passed / failed | RTX 5090 ran | RTX 5090 passed | RTX 5090 failed |
| --- | --- | --- | --- | --- | --- |
| Core | 472 | 472 / 0 | 472 | 471 | 1 |
| Audio | 155 | 153 / 0 | 153 | 153 | 0 |
| Diffusion | 406 | 396 / 0 | 396 | 396 | 0 |
| Vision | 74 | 70 / 0 | 70 | 70 | 0 |
| LLM (whole) | 849 | 815 / 0 | 815 | 782 | 33 |
| ThreeD | 1,466 ran of 1,457 listed | 1,435 / 0 | 1,425 | 1,417 | 8 |
| Engine PSA attention | 2 | not run | 2 | 2 | 0 |

The two passes failed the same 45 test lines (compared by name; the lists are identical). The Trellis2 attention
test passed at every shape, including 14,748 tokens, and the Audio, Diffusion, Vision and Engine suites passed.

### Failures, by cause

| Failed | Cause | Counts against Blackwell? |
| --- | --- | --- |
| UniRig 4 and CoLa3D 4 (ThreeD) | each message says a checkpoint file is missing; the pod has no weights | no, expected from the 4090 run |
| Core `PreloadUnderPressureGpuTests` | `OutOfVramException`: 81 MB of 32,110 MB free after the preload filled the card; the same kind of failure as on the Linux 4090 | no, recorded as a Linux benchmark-lane fact |
| LLM `Kimi.PackedMoeTests.Cuda_PackedExperts_MatchTheF32Dequantization` | -0.237698555 expected, -0.237098411 actual, outside 3 decimals; fails on the 4090 too and passes on the 3060 | recorded, not explained |
| 21 Vulkan tests in the whole-LLM run (route and A/B tests) | the only Vulkan device was `llvmpipe`, a CPU rasteriser with no cooperative matrices; they assert tiled rows that a software device does not take | no, a pod problem; not evidence about the Vulkan code |
| 11 CUDA tests that route through cuDNN | cuDNN is not in the image and its download was refused by the closed proxy, so each took a fallback row and asserted a `cudnn.*` row | not shown to; cuDNN absence is the likely cause and no run with cuDNN has confirmed it, so those CUDA rows are **unvalidated** on Blackwell |

The 9 failed timing tests are all of the Vulkan A/B kind and give no timing. The CUDA timing rows ran; they were
not compared across cards here.

## Third pass: a card that faulted

Pod `h5isulyrqg4d1o`, a third physical card (GPU UUID `3cb5d6d5-...`; the first two passes ran on others), same
commit, 27 minutes of tests, about $0.72. **`CUDA_ERROR_ILLEGAL_ADDRESS` (error 700) appeared in every test process**:
first in `SolAttentionGpuTests.OnCuda...` of the GPU lane, then in Core (`SdpaTiledGpuTests`, tile height 512), Audio,
ThreeD, Engine PSA and the whole LLM run. After the first fault in a process, every later CUDA call in it fails.
Counts against the earlier passes: LLM GPU lane 40 failed (24), LLM whole 81 (33), ThreeD 25 (8), Core 2 (1), Audio 6
(0), Diffusion 2 (0), Engine PSA 1 (0). Passes 1 and 2 at the same commit had no such error.

The cause is **not established**. It is either a faulty card or host, or an out-of-bounds access in a kernel that only
some runs trigger; one card in three cannot tell them apart. The run did not carry a memory checker. Later passes were
set up to run `compute-sanitizer` on the Sol attention tests if it recurs, but none reached a card.

The same pass downloaded cuDNN 9.21.0.82 for CUDA 13 (915,264,756 bytes, sha256 and size matching NVIDIA's published
manifest) and installed 0 files because the extraction step matched nothing; cuDNN stayed absent. That step was fixed
afterwards and has not been run.

## Vulkan

Expectation 2 (an NVIDIA Vulkan device visible) was put at about 60% and was **missed**. In pass 1 the loader logged
`libXext.so.6: cannot open shared object file` while loading `libGLX_nvidia.so.0`. Pass 2 installed that library and
the message changed to `Could not get 'vkCreateInstance' via 'vk_icdGetInstanceProcAddr' for ICD libGLX_nvidia.so.0`:
the container carries an NVIDIA GL library without a usable Vulkan entry point. The third pod carried the full
graphics library set (`libnvidia-glvkspirv`, `-eglcore`, `-glcore`, `-gpucomp`, `-rtcore`), `ldd` of `libGLX_nvidia` showed no
missing library, and the loader still reported the same `vkCreateInstance` error, so the library set is not the cause. `NVIDIA_DRIVER_CAPABILITIES=all` was
set when the pod was created and was unset in the SSH session. The rental campaign said that only if the device was
still missing with that variable set would the campaign stop with the finding that the Vulkan half cannot be run at this
provider. This is that finding, for this image. A different image or a host with the graphics capability injected is
the only route left; none was tried.

## What this leaves unvalidated

- Every Vulkan attention row on Blackwell hardware (tiled F32, flash, cooperative matrix, decode).
- The cuDNN attention rows on Blackwell. Provisioning cuDNN is a download from NVIDIA and is not approved.
- Stage 3 (frozen Kitchen operation corpus and comparison) and stage 4 (Qwen and Krea primaries), which need weights
  and, for the corpus, the runner that was removed from the repository.
- The Kimi packed-expert result, which differs on both Linux cards.

## Pod cost and incidents

Two RTX PRO 4500 Blackwell pods (EU-CZ-1, $0.72/hr) and four RTX 5090 pods never answered SSH and were stopped
with no test run, about $1.50 in all; one of them ran about 40 minutes before it was stopped, because the queue
answer that named it did not parse and the driving script lost its id (since fixed: it now finds and stops the running
pod, and the watchdog stops an unreachable pod after about 10 minutes). The three runs that tested cost about
$1.37. Total about $3.27 against the $9.73 ceiling. Every pod is stopped, none was terminated, and no volume was touched. Several requests for a 5090 expired in
the queue over about a day before one deployed.
