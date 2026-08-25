# docs/ — what's here and what belongs here

## The rule

**A doc earns its place only if its content can't be recovered from the code.**

This is the doc-side counterpart of the testing rule in [`CODE_STYLE.md`](CODE_STYLE.md) §Testing ("a test
earns its place only if its failure would be *silent*"). A walkthrough of how a model works, written when
that model was being ported, stops earning its place the moment the C# ships — the code says it better and
the code can't go stale. What the code *cannot* tell you is where the port came from, what the reference
implementation's constants were, where two references disagreed, and which traps were hit on the way.
That is what a doc is for.

Applied 2026-08-06: 57.7k → 40.8k lines.

## The four kinds of doc

**1. `Agents/` — how to work on this repo.** [`AGENTS.md`](Agents/AGENTS.md) is the entry point and the
architecture single source of truth (shared design rules + core engine patterns); it routes to a
specialized file per task. Deliberately small and example-driven. [`CODE_STYLE.md`](CODE_STYLE.md) is
mandatory and sits alongside it.

**2. `Checklists/` — live state.** Per-modality `MODEL_STATUS_*` docs (what's verified vs pending, plus
that modality's open work), [`ROADMAP.md`](Checklists/ROADMAP.md) (cross-cutting open work),
[`PARITY_VERIFICATION.md`](Checklists/PARITY_VERIFICATION.md) (the real-weight parity authority), and
[`TROUBLESHOOTING.md`](Checklists/TROUBLESHOOTING.md) (bring-up debugging reference). Index:
[`MODEL_STATUS.md`](Checklists/MODEL_STATUS.md). These track *open* work and *verified* status, not
history — when an item ships, delete the line. Git is the archive.

**3. `Research/` — model and technique notes.** Two sub-kinds, distinguishable at a glance:

- **Provenance stubs** — every doc carrying a `> **Stub.**` banner. The model is built and verified, so
  the narrative walkthrough, restated pseudocode and resolved open questions were removed. What survives
  is *Summary*, *Key Numbers / Constants*, *Data Layouts / Formats*, **Reference Implementations**,
  **Differences Between Implementations**, and *Implementation Notes* — upstream provenance, constants to
  diff a suspect port against, and bring-up traps.
- **Full docs** — technique references that are external knowledge and not derivable from this repo at all
  (CUDA/PTX, SIMD, Vulkan/SPIR-V, GGUF and safetensors formats, quantization, schedulers, codecs), plus
  any model still in bring-up, where the "it's built, the code is the truth" premise doesn't hold yet.

**4. `MULTI_GPU.md` — the one long-form user guide**, because sharding and placement have no other home.

**5. `ENV_VARS.md` — the environment-variable inventory.** Earns its place on the *Disposition* column, not
the list: which knobs are supported controls, which are undocumented default-ON numerics switches, which
silently corrupt output, and which survive only in a doc. The names themselves are re-derivable (the file
ends with the grep), so treat the table as scaffolding for the judgement, and delete a row when its variable
goes rather than letting the list rot into another stale doc.

## Conventions

**One number, one home.** Every measured performance figure lives in
[`benchmarks/scoreboards/`](../benchmarks/scoreboards/) — one canonical table per modality with GPU, date,
baseline and source. Do not restate a number in a README. This rule exists because the copies drifted:
`benchmarks/README.md` reported Llama-3.2-1B at ~111.5 tok/s (1.94× *behind* llama.cpp) for a month after
`scoreboards/LLM.md` had it at 213.7 tok/s (1.11× *ahead*). Link to the scoreboard instead.

**Never quote a speedup against our own past.** "30× faster than where the port started", "was 451 s",
"650s → 17.7s (37×)" — all of that is useless to a reader. It says nothing about whether the engine is
good, only that it was worse before. Quote the **absolute number** and the **external baseline**:
"1.671 s/step against ComfyUI's 1.660", "9.2 s, 1.6× off the Python reference". Where the delta *is* the
finding — a bug that made one op 65× slower than it should be — write it as the symptom, not as an
achievement. `(was <wrong behavior>)` notes are fine and worth keeping: they record what a bug did, which
is diagnostic. `(was <slower number>)` is not.

**Write the References section as if it's the only part that survives** — because for a model that ships,
it is. Same for *Differences Between Implementations*: "diffusers does X, the official repo does Y, we
follow Y because Z" is unrecoverable from our code, which only records the choice, never the alternative.

**Status tables stay scannable.** A `MODEL_STATUS_*` row is a verdict sentence plus a `([details](#…))`
link; the evidence goes in that doc's `## Details` section. Don't grow a table cell into an essay.

**Bring-up journals don't belong in a reference.** `TROUBLESHOOTING.md` and the status docs keep the
reusable lesson, not the blow-by-blow. Dated run logs, reverted attempts and variance tables go to the
relevant scoreboard or `ROADMAP.md`, or stay in git history.

## Writing a new research doc

The output shape is in [`Agents/RESEARCH.md`](Agents/RESEARCH.md). Precise beats vague: "48 blocks, hidden
3072, RoPE θ=150000, factor 32", not "a large transformer with rotary embeddings". Every architecture
claim needs a repo file/line, and every risky component needs a way to check the C# against a reference
within a stated tolerance.
