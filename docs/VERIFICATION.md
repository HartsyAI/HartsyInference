# Verifying a numerical change on a GPU

Two tools under `tools/` turn "tested on a GPU with real generations" into something a PR can show. Both are built with
`dotnet build tools/OpParity` and `dotnet build tools/GenCapture`; their executables are `hartsy-opparity` and
`hartsy-gencapture`. Fixtures and captures live outside the repository; the PR links the directory and pastes the
summaries.

## Op parity: `hartsy-opparity`

Records what a backend op computes today and checks what it computes after a change, on the same inputs.

```bash
# On the baseline commit, BEFORE the implementation is modified, per backend the machine has:
hartsy-opparity record --op attention --backend cuda --fixtures /data/fixtures
hartsy-opparity record --op attention --backend vulkan --fixtures /data/fixtures
hartsy-opparity record --op attention --backend cpu --fixtures /data/fixtures

# On the PR head, and again on the final commit after the old code is deleted:
hartsy-opparity check --op attention --backend cuda --fixtures /data/fixtures
```

- A **case** is one op invocation with seeded operands covering one caller's parameter set. Case sets live in
  `tools/OpParity/Cases/`; `hartsy-opparity list` prints them. A new op family adds a case class and registers it.
  Families today: `smoke`, `attention`, `layout` (the strided copy, accumulate and pad calls of [BACKEND_OPS.md](BACKEND_OPS.md)).
- `record` writes `inputs.safetensors`, `outputs.safetensors` and `manifest.json` under
  `{fixtures}/{family}/{case}/{backend}/`. The manifest holds the commit, backend and device, the hashes of the kernel
  artifacts the backend loads (PTX or SPIR-V), the settings file in force, the fixture hashes and the case's acceptance
  rule. A case the backend does not serve on the baseline is recorded as `unsupported`.
- **Coverage matrix.** `check` judges a recorded case against its fixture. A case that was unsupported on the baseline
  but is served now is judged against the CPU backend on the same inputs (the host references are the oracle). A case
  the backend still does not serve is reported as skipped, and fails the run when the case lists that backend as
  required (`RequiredBackends`, or `--require cpu,cuda,vulkan` on the command line).
- **Acceptance.** `BitIdentical` for copies, permutations and paths whose arithmetic is unchanged; `Tolerance`
  (absolute and relative, with the reason stated in the case) where indexing, reduction order, specialization or
  compiler output may change; `HostReference` for a case whose semantics the change corrects on purpose, which is
  compared against the CPU backend rather than the old output. Every output must be finite unless the case names it in
  `MayBeNonFinite` (masked logits, the LSE of a fully excluded row).
- `check` also reports when a kernel artifact's hash differs from the one recorded, so a re-parameterized kernel is
  visible even when its numbers still pass.

## Generations: `hartsy-gencapture`

Captures are produced by the CLI; the tool records what produced them and compares two capture directories.

```bash
# Baseline on main, candidate on the PR head, same commands, same seeds, into two directories.
hartsy text -m <model> --seed 1 --scores-out cap/llama.tsv "<prompt>"                        # per-token log-probabilities
hartsy text -m <model> --seed 1 --scores-out cap/llama-forced.tsv --teacher-force base/llama.tsv "<prompt>"   # teacher-forced on the baseline's tokens
hartsy image -m <model> --seed 1 -o cap/chroma.png --set diagnostics.chromaDebugDir=cap/chroma "<prompt>"  # PNG plus the model's raw F32 layer dumps
hartsy speech -m <model> --seed 1 -o cap/dia.wav "<text>"                                     # waveform
hartsy-gencapture manifest --dir cap --backend cuda                                           # commit, device, kernel hashes, file hashes

hartsy-gencapture compare --baseline base --candidate cap --max-abs 1e-4 --max-rel 1e-3 --png-levels 0 --pcm-lsb 1 --logprob-tol 1e-3
```

- `.safetensors` and the dump sinks' raw F32 `.bin` files compare by tolerance; `.png` by decoded 8-bit levels; `.wav` by
  16-bit PCM samples; a `--scores-out` TSV by forced token ids (must match), top-token ids (must match) and
  log-probability difference. Anything else compares by hash and is reported as such.
- Text comparisons are teacher-forced: the candidate scores the baseline's token sequence, so a divergence shows up as a
  log-probability delta at the first differing position rather than as an unrelated continuation. The TSV carries the
  forced token's and the top token's log-probabilities, not the full vocabulary.
- Tolerances are per PR and stated in it: bit-identical where arithmetic is unchanged, otherwise the tolerance and why.
  A known semantic correction (an order of operations the old code had wrong) is listed with its expected delta and
  checked against the host reference with `hartsy-opparity`, not against the old generation.
- Models whose dump sink exists (the `diagnostics.*DebugDir` settings in `EngineKnobs.DumpDirs`) give per-layer raw tensors;
  for the others the decoded output is the capture.

## What a PR pastes

The `hartsy-opparity check` summary per backend, the `hartsy-gencapture compare` summary, the two manifests' commit,
device and kernel hashes, and the fixture and capture directory paths.
