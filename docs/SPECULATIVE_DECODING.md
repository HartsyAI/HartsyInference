# Speculative decoding framework

This document covers the draft-provider framework for greedy text generation in `src/HartsyInference.LLM/Generation/Speculation/`.
It sits next to, and does not replace, two existing pieces:

- `TextGenerationPipeline.GenerateSpeculative` is the greedy verify loop. It drafts, runs one batched forward pass over the draft, and keeps the longest prefix the model agrees with.
- `src/HartsyInference.LLM/Generation/Speculative/` and `src/HartsyInference.LLM/DeepSeekV41/` hold the exact rejection-sampling contract (`RejectionSampler`, `SpeculativeLoop`, the DSpark proposer). That path is for sampled decoding with a proposal distribution. The framework here only proposes tokens and is used on the greedy path.

## Why the drafts can be swapped freely

Every draft is verified by the target model before it is emitted. A wrong guess costs a wasted forward row, never a wrong token. So a draft provider changes speed only, and the output stays byte-identical to plain greedy decoding. The existing `SpeculativeDecodeTests` check this end to end.

## Pieces

| Type | Role |
|---|---|
| `ISpeculativeDraftProvider` | `Name` plus `Propose(promptIds, generated, maxDraftLen)`. Returns up to `maxDraftLen` guessed tokens, or an empty array. |
| `PromptLookupDraftProvider` | The drafter the pipeline has always used, moved out unchanged. Fixed 3-token suffix, oldest match first, 4096-token lookback. Proposes exactly the tokens the old inline code proposed. |
| `NGramSuffixDraftProvider` | Tries suffixes from 4 tokens down to 1 and drafts the continuation of the most recent earlier match of the longest suffix that recurs. Matching is delegated to the existing `PromptLookupProposer`. |
| `SpeculationSelector` | Chooses a provider each round by measured throughput and auto-disables providers that do not pay off. |
| `SpeculationSelectorOptions` | Acceptance threshold (default 0.3), disable streak length N (default 8), throughput smoothing (default 0.25). |
| `SpeculationRound` | One round's measurements: proposed, accepted, tokens emitted, elapsed milliseconds. |

The pipeline calls `PromptLookupDraftProvider` through a static `SpecDraftProvider` field in `GenerateSpeculative`. The selector is not wired into the pipeline yet. A single provider does not need selection.

## Selector

The caller runs one loop per round:

1. `Select()` returns the provider to draft with, or `null` when every provider is disabled. `null` means decode without speculation.
2. The provider drafts. The target verifies. The caller times the whole round.
3. `Record(provider, new SpeculationRound(proposed, accepted, tokensEmitted, elapsedMs))`.

Selection rules:

- An enabled provider with no recorded round is picked first, in registration order. Every provider is measured once before any comparison.
- After that, the enabled provider with the highest throughput wins. Throughput is an exponential moving average of tokens emitted per millisecond. Ties go to the earlier provider.

The selector never reads a clock or a random source. Given the same sequence of measurements it makes the same decisions, so tests inject measurements directly.

## Auto-disable rule

A round counts toward the streak only when the provider proposed at least one token. Its acceptance rate is `accepted / proposed`.

- A round below `AcceptanceThreshold` (default 0.3) adds one to the streak.
- A round at or above the threshold resets the streak to zero.
- Rounds with no proposal change nothing. They are neither failures nor successes.
- When the streak reaches `DisableAfterRounds` (default 8), the provider is disabled for the rest of the selector's life.

A provider's throughput still counts after a no-proposal round. An empty draft has a real cost, so it lowers the moving average. Rounds with zero elapsed time add no throughput sample.

## Adding a provider

1. Implement `ISpeculativeDraftProvider`. Keep `Propose` pure over its arguments, with no state that changes output.
2. Give it a unique `Name`. The selector rejects duplicates.
3. Pass it to `new SpeculationSelector(providers, options)`.

Any draft is safe for correctness because the target verifies it. A provider that proposes garbage is caught by the auto-disable rule.
