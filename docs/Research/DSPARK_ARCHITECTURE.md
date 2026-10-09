# DSpark on the V4.1 host path

Status: CPU reference, synthetic evidence. Greedy drafting only. Not verified on real weights; the evidence and its limits are in
`docs/Checklists/PARITY_VERIFICATION.md`. The reference is the DSpark paper (arXiv 2607.05147, Alg. 1 and section 3) and the upstream `DSparkBlock` in the pinned
`deepseek-ai/DeepSeek-V4.1-Flash` code, which has no generation loop of its own.

## Components

- **Target taps** (`DeepSeekV41HostModel`). With `dspark_target_layer_ids` configured, `Forward` can write each position's `main_hidden`: the hc-mean of every
  target block's entry stream, taken after the block's Engram step and before its sublayers, in ascending layer order. Off unless a caller passes the output span.
- **Draft head** (`DeepSeekV41DSpark`). Three stages (window attention over the target's latents, a 128-expert feed-forward, hyper-connections), the Markov head
  that chains greedy drafts, and the confidence head. `Seed` writes the window from committed rows; `Draft` writes the anchor's latent into its slot and drafts a
  block. The first id it returns is the anchor.
- **Sequence state** (`DeepSeekV41GenerationState`). The committed tokens, the last committed final hidden row, and, when created with `recordMainRows`, each
  committed position's target rows. `SyncTo(context)` rolls back only past the first differing token.
- **Scorer** (`DeepSeekV41SpeculativeScorer`). Syncs to the context, then runs the draft once over the shared state. Reads the logits of the last context token and of
  each drafted position.
- **Proposer** (`DeepSeekV41DSparkProposer`). Syncs to the context, copies the committed target rows before the anchor into a fresh head window, drafts the anchor,
  and returns the drafted ids with no probabilities.

## Invariants

- **The window holds only committed positions.** `Draft` writes one slot, the anchor's, and block rows are never persisted. So a rejected draft needs no window
  restore: the next call rebuilds the window from the committed rows.
- **The rebuilt window equals upstream's incremental window.** At every decode position from 11 to 16, with a window of 8, a fresh `Seed` of the committed rows
  before the position reproduces upstream's draft. That covers the wrap.
- **Position 0 has no upstream draft.** Upstream's `forward_spec` at start position 0 is its prefill seed and returns nothing. The proposer drafts there from an
  empty window, so a one-token prompt gets a draft upstream would not produce. That changes acceptance only; verification corrects every draft.
- **Rollback replays the history's own call structure.** The prompt is one prefill chunk and every later token is a single-token step. The reference's chunked
  prefill and per-token decode are not arithmetically equivalent, so a rollback replays the chunk and then the single tokens, not one chunk.

## Evidence

- `DeepSeekV41DSparkFixtureTests`: the target taps and the drafts against the upstream run on a synthetic model (`dump_dspark_fixture.py`).
- `DeepSeekV41DSparkProposerTests`: the proposer's drafts at every decode position, and greedy speculation with the proposer equal to plain greedy decoding, with
  drafted rounds asserted.
- `DeepSeekV41SpeculativeScorerTests`: the scorer's rows equal per-token logits, the rollback work is counted in block passes, and greedy speculation matches plain
  decoding across prompt lengths and draft sizes.

## Not yet

- Acceptance statistics and the confidence scheduler with its fitter (`SpsProfile`, Algorithm 1). A confidence field on the draft block comes with it.
- EOS and stop truncation of an accepted draft.
- In-place truncation. A rollback still replays the kept history, which costs one decode per token after the prompt.
- The Engram tap placement on a target layer that has Engram. The synthetic model has none on a tap, and the real DSpark targets (37 to 39) have none either.
- A real-weights proposer round beyond the chain test, GPU paths, and a free-running real-weights agreement row.
