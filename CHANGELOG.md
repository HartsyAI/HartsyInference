# Changelog

All notable changes to HartsyInference are recorded here. Versions follow `2.0.0-alpha.N` (the scheme moved
up from `1.0.0-alpha.N`; entries below that pre-date the change and keep their original numbers). The single
source of truth is `<VersionPrefix>`/`<VersionSuffix>` in `Directory.Build.props` — see
[`docs/Checklists/ROADMAP.md`](docs/Checklists/ROADMAP.md) for what a
stable release will require. Dates are UTC.

## Unreleased

## alpha.308

- **Added: `/ready?model=`, `/admin/deployments`, `/admin/capacity`, and `hartsy.status` frames.** `/ready?model=` answers 200 only while a deployment of the model is Ready. A model with no deployment answers 503 with `not_deployed`, and a deployment that is not ready reports its state: with none Ready, the latest Loading one, else the latest record, so a stale record never hides a load in progress. `GET /admin/deployments` lists deployments. `POST` loads one and waits for the load; a failed load, an unresolvable model included, is reported in its state with 200, not as an error. `DELETE /admin/deployments/{id}` unloads that deployment's model only while it still holds its device (`ITextService.UnloadDeployment`), so deleting a deployment another one replaced never unloads the replacement: 200 with `unloaded: true`, 200 with `unloaded: false` and a reason when the deployment no longer holds its device, 409 when requests on it do not finish in time (the model stays loaded), and 404 for an unknown id. `GET /admin/capacity` reports each deployment's state, its active and queued sequences, its concurrency limit, and its KV pages (pages, not bytes). A streamed chat reply sends a named `hartsy.status` event while its request waits in the queue. OpenAI clients ignore named events, so the standard stream is unchanged. Covered by readiness tests (ready, loading, no deployment, a ready deployment preferred over an older loading one, and a newer loading one reported over a stale unloaded one), the exact shape of the status frame, status-code tests for every deployment route against a scripted text service (including the 200 `failed` answer the real engine gives an unresolvable model), and a V4.1-fixture test in which unloading a replaced deployment leaves its replacement Ready and resident. Not in this change: automatic placement (the device must be named, or the engine's primary device is used), deployments and packages in `/v1/models`, and prefill progress, which the stream reports as zero for now.

- **Changed: an unknown device on `POST /admin/deployments` is a 400.** It used to load onto a CPU slot without a word, and a bad device string left a slot under that key. A recognised device that cannot load still answers 200 with `state: failed` and the reason. A test pins the refusal, and that nothing loads.

## alpha.305

## alpha.307

- **Changed: prefix-cache entries are scoped by tenant and model.** A retained prefix is stored under the request's tenant, the model the device holds, and the caller's prefix key. Each part before the key is length-prefixed, so no two scopes can collide. One tenant's prefix therefore never serves another's, and a different model on the same device is a different scope. The model part is the loaded path as written, so a path differing only in case is another scope too: an extra miss, never a wrong hit. The V4.1 host takes no prefix-cache hits in this version, exact-prompt hits included: the plan allowed those, and turning them on is a follow-up that needs the exact-prompt gate. One rule, `PrefixCacheScope.TryKey`, decides whether a request takes a hit, for both checkout and check-in. Covered by key tests (tenant, model, and ambiguous names), a store test with a live cache, in which alice's entry is never returned to bob, and a `TryKey` test in which a slot holding the loaded V4.1 fixture yields no key. The image-hash part of the key is reserved for vision and is empty for text.

## alpha.306

- **Added: deployments, and a drain that unloads in order.** `ITextService.DeployAsync` loads a model onto its device as a named deployment, without generating anything. `Deployments` lists them, and `Capacity(id)` reports the device's active and queued sequences and its KV pages (pages, not bytes). A deployment's state moves only along the transitions the state machine allows: Loading, Ready, Degraded, Draining, Failed, Unloaded. Unloading a model now cancels the scheduler's waiting requests first; they fail with `SchedulerStoppedException` (503). Only then does the unload wait for the sequences already decoding, which run to their end. If that wait times out, the model stays resident and its scheduler takes requests again. A deployment is recorded as Unloaded where its model is freed (an unload, a request that loads another model on its device, or `AlwaysFreeMemory`), so `Deployments` and `Capacity` report the same state. A deploy onto a device leaves the device's current deployment Ready until the new model is in place, and one cancelled while it waits for the device is recorded as Failed. Of two overlapping deploys of one id, only the later completes the record; the earlier loads nothing once overtaken and reports the record as it stands. With no scheduler resident, `Capacity` reports 0 active, queued and concurrent sequences. The API sets each request's tenant from the caller's identity (`local` when auth is off), and the client never sets it. Covered by state-machine tests, a drain test (the waiting requests fail, the admitted one finishes, nothing new is admitted), tenant tests, and V4.1-fixture tests through `TextService`: an unload that times out on a lease keeps serving, overlapping deploys of one id, a deploy waiting on the device, and `Capacity` agreeing with `Deployments` after an unload and after a reload. Not in this change: `ListPackagesAsync`, which needs the package discovery in the 19 stack, and deployments over the wire, which come with PR 19d. One deployment holds a device at a time: deploying onto a device replaces its model.

## alpha.305

- **Changed: the scheduler admits requests by KV pages, in FIFO order, instead of failing them when the pool is full.** A request reserves its whole prompt-plus-budget footprint at admission, so an admitted sequence never runs out of pages mid-decode. A request that does not fit waits its turn. A request larger than the whole pool is refused with 400, as is one whose budget no sequence can hold (a negative or overflowing `max_tokens`), on any model. A solo sequence, which gets its own fixed KV cache, is still charged its pages and refused above the pool size like any other. A full waiting queue (64 by default) refuses the next submission with 429 and `Retry-After: 1`. A cancelled request leaves the queue at the next admission pass, wherever it stands, so it stops counting toward that bound. A request that an admission pass leaves waiting is told its place in the queue once, as a status chunk; a request admitted at once is told nothing. A pool-less model (the V4.1 host) decodes at most four sequences at once. With `vram.continuousBatching` on, the API no longer queues text requests itself. Each pipeline request takes a slot in the server's shared queue, bounded like an HTTP request (429 when full), and each GPU round of a scheduled request waits its turn in the same queue, so LLM and image work never overlap on a device. A streamed request that finds the queue full, scheduled or pipeline, gets an error event, not a 429; the non-streamed path returns 429. Covered by tests for the queue bound, FIFO waiting under pool pressure (outputs equal to solo runs), the oversized refusal, an `int.MaxValue` budget refused with nothing reserved (pooled and pool-less), a cancelled waiter behind the head leaving before the head is admitted, queue positions (none for a request admitted at once), the pool-less cap, and the 429 mapping. Not in this change: the slow-consumer channel and its stop reason, and the status frames on the wire (PR 19d).

- **Fixed: a device round the scheduler has admitted waits for its turn, and a stopped loop refuses later requests.** The shared queue's depth bound refused a round with 429 mid-decode, which failed requests already admitted; a round now waits (`InferenceQueue.EnqueueAdmittedAsync`), while a pipeline request still takes the bounded queue (`PipelineGate`), so the bound still refuses new HTTP requests. A loop that stops (a round or release that throws) closes its intake, so a request submitted after that gets `SchedulerStoppedException` (503) instead of waiting forever. Covered by a queue test (an admitted round waits where a new request is refused) and a scheduler test (a submit after the loop stops is refused).

## alpha.304

- **Changed: the V4.1 host reference runs each routed expert once per batch of tokens, not once per token.** `DeepSeekV41MoeExecutor.Run`
  gathers the tokens routed to an expert and runs them together, so a stored-form expert is decoded once per call instead of once per token.
  Every output element is the same sequential dot product, so the results are bit-identical to the token-by-token path
  (`DeepSeekV41MoeExecutorTests.Batched_Run_Matches_Token_By_Token_Run_Bit_For_Bit`, which crosses the 256-row batch cap).

## alpha.288

- **Added: CPU reference for the DeepSeek-V4.1-Flash vision tower and aligner.** `DeepSeekV41VisionLoader.Load` reads the 266 `vision.*`, `aligner.*` and `image_*` tensors (BF16 in the official shards 1 and 2) as F32 and returns a
  `DeepSeekV41VisionModel`. `Encode(patches, gridHeight, gridWidth)` runs the tower (patch embedding, 32 pre-norm blocks of full attention with a 2D half-split rotary and a SwiGLU MLP, final RMSNorm) and the aligner (zero-padded 3x3 fold,
  exact GELU, two projections) on existing backend ops, and `ImageStart`, `ImageEnd` and `ImageNewline` expose the learned span embeddings. `DeepSeekV41VisionConfig` gains `RopeTheta` (default 10000, read from `rope_theta` or
  `vision_rope_theta`) and the derived widths. Against the unmodified upstream `vision.py` in float32 the stages agree to 1e-6 relative on a small seeded fixture and to 7e-6 on the real weights (aligner output correlation 1.000000, relL2 2.8e-6 on a
  28x28 patch grid and 1.8e-6 on 17x23). Nothing calls it yet: image preprocessing, the splice into the language model and the GPU path are not built.

## alpha.289

- **Added: `hartsy moe pack` and `hartsy moe verify`, reading GGUF expert tensors.** Packing reads a checkpoint's stacked expert tensors one expert at a time, in separate or fused gate/up form, and writes a quantized pack. Verify reads the pack back, checks each checksum, and compares values with the checkpoint. The pack's fingerprint is provisional until the runtime binds packs to topologies. Tested on synthetic GGUFs only; see `docs/MOE_PACK.md`.

## alpha.303

- **Added: heterogeneous expert execution on the CPU side.** `HeterogeneousExpertExecutor` runs a layer's planned experts: GPU assignments through `IExpertDeviceRunner`, CPU assignments through the F32 reference, with the same rows and output layout either way. Tests show the same plan gives identical output under all-CPU, all-GPU and mixed placements, using a reference device. The CUDA device runner and the asynchronous handoff are not built yet; `docs/HETEROGENEOUS_EXECUTION.md` sets the protocol they must meet.

## alpha.302

- **Added: a continuous-batching route for chat, behind `vram.continuousBatching` (off by default).** When on, a batch-capable model's chat requests go through `DynamicBatchScheduler`, so concurrent requests share one decode round instead of queuing on the slot. The slot lock covers the load only; scheduled requests run without it, and a load or unload waits (up to 120 s) for scheduled requests on the old model. That wait follows the loader's own rule for when a load replaces the model (an exact path match), and a scheduled request completes only after the scheduler has released its sequence, so no teardown is left waiting on the device gate. Requests with a prefix-cache key, an image, or `AlwaysFreeMemory` stay on the pipeline. The knob is read when a model loads, and V4.1 host sequences have no bound until the admission control of PR 18b. Until then, a scheduled sequence that cannot get a KV page fails, and the API answers 500. Default behavior is unchanged. CPU tests: concurrent V4.1 requests on the scheduler match their solo pipeline runs byte for byte on the synthetic fixture, a request completes only after its sequence is released (finished, cancelled, or failed round), a load of a path differing only in case waits for the leases, plus the lease wait and the routing rules. There is no real-generation A/B yet, so the knob stays off.
- **Changed: a stopped scheduler fails its requests instead of leaving them pending.** `Dispose` completes queued requests, and the running ones, with `SchedulerStoppedException` (queued requests used to never complete). The API returns 503 for it, not 500. The exception derives from `ObjectDisposedException`.
- **Removed: `ModelManager`.** Nothing constructed it. Its KV page-count helper is now `PagedKvPool.PageCountForBudget`.

## alpha.301

- **Added: the DSpark draft head checked on the real checkpoint past the 128-token window (CPU).** `dump_real_dspark.py --max-prompt-tokens N` extends the prompt with seeded random ids and sizes the context to fit. With a 160-token prompt the window wraps: the three draft stages agree with the unmodified upstream at relL2 at most 8e-6, and the draft ids match exactly. The chain from the C# target's own taps reproduces the ids and gives logits at 5.4e-4, but its decode tap is at 1.2e-3, the weakest number, not yet explained. The real-weights tests size their state from the prompt. No change to the model's output.

## alpha.300

- **Added: a check that a DSpark draft tap on an Engram block is taken after the Engram step (CPU, synthetic).** `DeepSeekV41EngramTapTests` applies the same Engram module to the embedding stream, with the hash ids the host computes, and requires the tap to equal the hc-mean of that stream and to differ from the mean before the step. Removing the step fails the test. The real DSpark targets have no Engram layer, so this covers a placement the real-weights chain test does not reach. No change to the model's output.

## alpha.299

- **Changed: speculative generation stops at a stop token inside an accepted draft.** `SpeculativeLoop.Generate` takes an optional set of stop tokens. The first one emitted ends the run and is kept, and nothing after it is emitted, even when it sits inside an accepted draft or is the bonus token. Without a set the run is unchanged. Covered by scripted-target tests (a stop inside an accepted draft, a stop as the bonus token, and the no-stop path). Decoding through the engine's pipeline is unchanged; wiring the set through the DeepSeek-V4.1 generation path comes with the serving work.

## alpha.298

- **Added: confidence-scheduled verification for the DeepSeek-V4.1 DSpark proposer (CPU, synthetic evidence).** `ConfidenceScheduler` implements Algorithm 1 of the DSpark paper for one sequence: the head's confidences become survival products, and the walk over drafted positions stops at the first position that does not improve the expected throughput `(1 + Σ a_j) · SPS(1 + l)`. `SpsProfile` holds the engine's steps-per-second table; measuring it is left to the deployment. `DeepSeekV41DSparkProposer` takes an optional scheduler and verifies only the prefix it chooses. On a synthetic objective the walk matches brute force wherever the objective is unimodal, and a jagged profile shows the early stop that the paper's section 5.2 search addresses (not implemented here). Plain decoding is unchanged.

## alpha.297

- **Added: a DSpark draft proposer for DeepSeek-V4.1 speculation (CPU, synthetic evidence).** `DeepSeekV41DSparkProposer` drafts from a sequence that records the target's draft rows. Each call syncs the sequence to the context, rebuilds the draft head's window from the committed rows, and drafts from the last token. The window only ever holds committed positions, so a rejected draft needs no restore. On the synthetic model the proposer's drafts reproduce upstream's at every decode position, including past the 8-token window, and greedy speculation with it reproduces plain greedy decoding token for token. The sequence records the rows the head reads only when asked to. Plain decoding is unchanged.

## alpha.296

- **Added: the DeepSeek-V4.1 DSpark draft head checked against the unmodified upstream on a synthetic model (CPU).** `dump_dspark_fixture.py` runs the upstream draft head on the host fixture's backbone, with seeded random draft weights, through an 11-token prefill and six decode steps, and records the drafts upstream's incremental window produces. The C# head, seeded afresh at each decode position from the committed rows (positions 11 to 16, past the 8-token window), reproduces the drafted ids exactly, with logits and confidences within 1e-3 relative. The target's taps for the three draft layers reproduce upstream's main hidden states at every position, and the same drafts come out when they are fed from the target's own taps. Acceptance rates are not measured on random weights. The embedding and head shared with the target are shape-checked. Plain decoding is unchanged.

## alpha.295

- **Changed: the speculative scorer on the DeepSeek-V4.1 host model no longer replays a context the sequence already holds.** `DeepSeekV41GenerationState.SyncTo` rolls back only past the first token where the sequence and the context differ, and the state keeps the last committed token's hidden row, so a call whose context is already in place runs only its draft through the blocks, and a rejection replays the kept prefix once. The state can also record each committed position's DSpark target rows (opt-in), which a draft reads. Plain decoding and the speculative outputs are unchanged.

## alpha.294

- **Added: speculative decoding on the DeepSeek-V4.1 host model, exact under greedy decoding (CPU, synthetic evidence).** `DeepSeekV41SpeculativeScorer` scores a draft over the model's own sequence state: it rolls back to the part of the context the state already holds, appends the rest of the context and the draft, and reads the logits after the context and after each drafted token. A rollback replays the committed history the way it was built: the first append as one prefill chunk, every later token one at a time. That matters because the reference's chunked prefill and its per-token decode are not arithmetically equivalent on every length (upstream differs from itself on the synthetic fixture), so a single-chunk replay changed the state and the greedy output. On the synthetic V4.1 model the scored rows equal the per-token logits for prompts of 1 to 11 tokens and drafts of 1 to 4, and greedy speculation with the prompt-lookup proposer reproduces plain greedy decoding token for token through the rejections, for prompts of 6, 9 and 11 tokens and draft limits up to 6. Plain decoding is unchanged.

## alpha.293

- **Added: the DeepSeek-V4.1 target exposes the rows its DSpark draft reads, and the draft chained from them matches the upstream oracle.** `DeepSeekV41HostModel.Forward` can write `main_hidden` for the layers in `dspark_target_layer_ids`: the hc-mean of each target block's entry stream, taken after its Engram step and before its sublayers, in upstream's layer order. The tap is off unless a caller passes the output span, so the default path is unchanged. On the real checkpoint (structural mode, the oracle's 6-token prompt and one decode step), the tapped rows agree with upstream at relL2 5.1e-7 (prefill) and 4.8e-6 (decode), and the draft built from the C# target's own taps reproduces the upstream ids exactly, with logits at relL2 2.2e-6 and confidences within 2.3e-5.

## alpha.292

- **Added: the DSpark draft forward for DeepSeek-V4.1-Flash on the CPU host reference.** `DeepSeekV41DSpark` loads the three `mtp` stages (draft attention over the target's sliding window with the block's own latents, a 128-expert feed-forward, hyper-connections) with the Markov and confidence heads, and drafts a block greedily. Against the unmodified upstream `forward_spec` on the real checkpoint (structural mode, one decode step): every stage agrees at relL2 at most 5e-6, the draft ids match exactly. The backbone's rotary moved to a shared helper with no change in output. No change to the generation path.

## alpha.291

- **Added: the speculation contract (CPU, synthetic evidence).** `SamplerChain.Distribution` exposes the distribution the sampler draws from, without consuming randomness. `RejectionSampler.Verify` is exact speculative sampling: a drafted token is accepted with probability min(1, p/q), and a rejection draws from the normalized residual max(0, p - q), so the emitted tokens follow the target distribution whatever the proposer does. `PromptLookupProposer` drafts from n-gram matches, and `SpeculativeLoop` runs draft, one scoring pass and verification over a stateless scorer. Greedy speculation reproduces plain greedy decoding; the statistical tests (chi-square at 1e6 trials, and a two-sample test against the plain sampler) and two deliberately broken samplers back the claim. No change to the existing pipeline's speculative path.

## alpha.290

- **Added: a structural oracle for the DeepSeek-V4.1 host reference, at full depth, on the real checkpoint (diagnostics, off by default).** `dump_real_layers.py` runs the unmodified upstream model through lazy weight shims, so any depth fits in RAM. `RealLayers_MatchTheUpstreamModel` compares the host layer by layer, token by token and decode step by decode step, including the expert ids each token was routed to. Diagnostic switches: `DeepSeekV41LoadOptions.QuantizeLatents` and `DeepSeekV41AttentionSettings.QuantizeLatents` (off skips the FP8/FP4 cache round trip), `DeepSeekV41Block.Probe` now also reports `route`, and `DeepSeekV41MoeLayer.RouteProbe`. No change to normal output.

## alpha.289

- **Changed: the V4.1 host reference runs each routed expert once per batch of tokens, not once per token.** `DeepSeekV41MoeExecutor.Run`
  gathers the tokens routed to an expert and runs them together, so a stored-form expert is decoded once per call instead of once per token.
  Every output element is the same sequential dot product, so the results are bit-identical to the token-by-token path
  (`DeepSeekV41MoeExecutorTests.Batched_Run_Matches_Token_By_Token_Run_Bit_For_Bit`, which crosses the 256-row batch cap).

## alpha.288

- **Added: CPU reference for the DeepSeek-V4.1-Flash vision tower and aligner.** `DeepSeekV41VisionLoader.Load` reads the 266 `vision.*`, `aligner.*` and `image_*` tensors (BF16 in the official shards 1 and 2) as F32 and returns a
  `DeepSeekV41VisionModel`. `Encode(patches, gridHeight, gridWidth)` runs the tower (patch embedding, 32 pre-norm blocks of full attention with a 2D half-split rotary and a SwiGLU MLP, final RMSNorm) and the aligner (zero-padded 3x3 fold,
  exact GELU, two projections) on existing backend ops, and `ImageStart`, `ImageEnd` and `ImageNewline` expose the learned span embeddings. `DeepSeekV41VisionConfig` gains `RopeTheta` (default 10000, read from `rope_theta` or
  `vision_rope_theta`) and the derived widths. Against the unmodified upstream `vision.py` in float32 the stages agree to 1e-6 relative on a small seeded fixture and to 7e-6 on the real weights (aligner output correlation 1.000000, relL2 2.8e-6 on a
  28x28 patch grid and 1.8e-6 on 17x23). Nothing calls it yet: image preprocessing, the splice into the language model and the GPU path are not built.

## alpha.287

- **Added: the placement planner for a routed layer.** `ExpertScheduler.Plan` reads the cache's residency and assigns each routed
  expert to the GPU (resident) or the CPU (missing), through a replaceable `IMissExecutionPolicy`. Planning pins the resident
  experts it plans on, returned as a lease, so they cannot be evicted before the layer runs; it uploads nothing, and a policy may
  not place a missing expert on the GPU. Every routed (token, slot) pair is counted once. It is allocation-free once warm: the caller owns
  the scratch, the output, the miss list and a reusable lease, and `AcquireResident` binds that lease without allocating. The GPU execution, the CPU kernels and the cross-device combine are not in this change and need the test rig.


## alpha.286

- **Added: expert packs, a quantized on-disk store for routed experts.** `ExpertPackWriter` quantizes each expert's gate, up and
  down projections with the existing GGUF codecs (Q8_0, Q4_K, Q5_K, Q6_K) into 4 KiB-aligned records, with a manifest that
  names the topology fingerprint and a SHA-256 per record. Nothing is visible until the pack completes; `ExpertPackReader` is an
  `IExpertSource` that refuses incomplete packs, other versions and other topologies, and checks each record on read.
  `ExpertPackVerifier` compares the dequantized values with the F32 source. Measured on random weights: Q8_0 3.76x smaller
  than F32 at 0.38% relative RMSE, Q4_K 7.1x smaller at 6.1%. Real checkpoints are not yet packed or verified. See
  `docs/MOE_PACK.md`.

## alpha.285

- **Added: real-weight oracle for DeepSeek-V4.1-Flash layer 0, and a diagnostic probe on the host block.** `dump_real_layers.py` runs the unmodified upstream model on the real checkpoint's first layer; `RealLayers_MatchTheUpstreamModel` (gated on `DSV41_ORACLE_DIR`) compares the host reference against it, hidden relL2 4.3e-6 against the float32 oracle. `DeepSeekV41HostModel.SetProbe` / `DeepSeekV41Block.Probe` expose per-sublayer values for such comparisons and are off by default. No behaviour change.

## alpha.284

- **Added: residency queries and no-upload acquisition on the expert cache.** `IResidencyAwareExpertCache.LookupResident` reports which
  experts are resident or uploading, without changing any state. `IResidencyAwareExpertCache.AcquireResident` pins only the resident
  experts and returns the rest as misses, so a scheduler can run misses elsewhere without uploading them. Both live on the
  new derived interface `IResidencyAwareExpertCache`; the published `IExpertCache` is unchanged, so its implementers are not
  broken. `Acquire` shares its pin-and-lease tail with the new path, and its behavior is unchanged. The CUDA cache inherits the
  behavior through `ExpertCacheBase`; its GPU tests are not run here.

## alpha.283

- **Added: expert identity with banks and a pluggable expert source.** `ExpertKey` gains a `Bank` (defaulted, so existing
  two-argument keys are unchanged), and `ExpertLayerKey` is the cache's layer identity. Two models sharing one cache each
  register their own bank, so the same layer number no longer collides; re-registering a different bank under one layer key
  still throws. `IExpertSource` (with `ExpertBacking`: resident host, memory-mapped, pack, device) supplies expert weights on
  demand, `DelegateExpertSource` adapts existing resolvers without copying them, and `ExpertBank.FromSource` builds a bank from
  one. No model or runtime path changes behavior; the CUDA project builds against the new types, and its GPU tests are not
  run here.

## alpha.282

- **Added: exact-contract reference path for routed sparse layers.** `Core.Moe.SparseFfnReference` runs a routed layer
  from a `MoeLayerDescriptor` by composing the existing backend op contracts (`MoeReference.Route`, `BuildDispatch`,
  `Combine`) with `ExpertProgramReference`, a scalar F32 execution of `ExpertProgram` (activation plus independent gate/up
  clamps). It is the correctness oracle for later backend expert kernels. Shared experts are rejected explicitly until
  they are covered. Parity against `MoeFeedForward` on identical weights holds within 1e-4 for softmax routing and for
  grouped sigmoid routing with a selection bias. Production routing is unchanged.

## alpha.281

- **Added: generic sparse (MoE) topology contracts.** New `HartsyInference.Core.Moe` describes a model as layers (dense or
  sparse), with a router (per-phase top-k, group limiting, scoring, token-kind bias), routed and shared expert groups with
  per-index shape overrides, and an `ExpertProgram` (activation plus independent gate/up clamps) instead of a hard-coded
  SwiGLU. `SparseCapabilities` is derived from the topology so runtime code checks capabilities, not model names;
  `SparseModelTopology.Fingerprint` is a SHA-256 over shapes, routing and programs for profile and pack binding.
  `MoeTopologyFactory` maps transformer MoE configs and `DeepSeekV41Topology` maps the official V4.1 config (routed
  experts, clamped SwiGLU, draft layers). No runtime behavior changes. See `docs/MOE_ARCHITECTURE.md` and
  `docs/Research/HETEROGENEOUS_MOE.md`.

## alpha.280

Release cut. New in this release's source: the voice-agent call-audio detectors (#296). The `ToolRegistry`, Clef and CUDA
release entries below merged before alpha.279 was cut but were never given a changelog section, so they are recorded here.
No code change beyond the version.

- **Added: in-band DTMF and call-progress detectors for the voice agent, optional and off by default.** New
  `HartsyInference.Audio.Dsp.Telephony`: `DtmfDetector` (Goertzel, all 16 keys, twist/dominance/second-harmonic checks,
  no false key in 614 s of speech, babble and noise) and `CallProgressClassifier` (ringback, busy, fast busy, SIT, dial
  tone and beeps by cadence, plus heuristic machine/human/hold-music/silence signals with confidences). `VoiceAgentSession`
  gains `DetectInbandDtmf`, `DetectCallProgress` and `ForwardInbandDtmfToModel` options and the
  `InbandDtmfDetected`/`CallProgressDetected` events; forwarded keys reach the model as `[INBAND DTMF n]`, distinct from
  the `[DTMF n]` of `PushDtmf`. With the options off the audio thread runs no new code; on, both detectors cost p99 48 us
  per 20 ms frame with no allocation. Details and limits: `docs/Research/VOICE_AGENT_SESSION.md`.
- **Added: `ToolRegistry` is thread-safe, supports `Remove`, and takes an optional tool timeout.** `Add`, `Remove`,
  lookup and `Definitions`/`Names` can run concurrently; `Definitions` is an immutable insertion-ordered snapshot, so
  a request in flight never sees it change. `Remove(name)` unregisters a tool. A per-tool timeout (new `Add` overloads)
  or `ToolRegistry.DefaultTimeout` cancels the handler's token on expiry and returns the usual `{"error": ...}` result
  instead of throwing, even for a handler that ignores its token; the caller's own cancellation still propagates. No
  timeout by default, so existing hosts behave as before.
- **Added: Cloudflare Clef GGUF catalog entry (`clef`, Q2_K + f16 mmproj, status ValidationPending).** The 64-layer
  Qwen3.5 hybrid VLM loads through the existing `qwen35` path; both LFS SHA-256s are pinned and match the Hugging Face
  tree metadata. bartowski no longer ships an IQ2_XXS, so Q2_K (~11.7 GB) is the smallest quant. No real-weight run has
  happened yet; the entry stays ValidationPending until the opt-in `ClefGgufRealWeightTests` passes.
- **Fixed: releasing a tensor's device copy binds the owning cache's CUDA context first.** A tensor's dispose
  callback runs on whichever thread disposes it, and in SwarmUI that is a pool thread shared by every extension's copy
  of the engine, so it can arrive with another device's context current. The residency-cache migration had dropped the
  bind from `ReleaseActivationCore` and from `DemotePromotedWeight`, the callback for a promoted weight, which CUDA
  frees with the synchronous `cuMemFree`. That call resolves against the context current on the thread, so the free
  could land in the wrong device's context or fail against it. Both paths now call the cache's `MakeCurrent`, after the
  retire gate and not before it: a retiring backend closes the gate and only then destroys its context, so a bind
  placed earlier would throw against a context that is already gone.
- **Fixed: `GpuBackendBase.FreeAllDeviceMemory` on CUDA no longer leaks Q8_1 activation sidecars.** An activation can
  carry a sidecar of three more device buffers, and the shared sweep hands buffers back without calling the
  per-activation eviction hook, so only the CUDA wrapper around it (`GpuTransferHelper.FreeAllCached(State)`) released
  them. `FreeAllDeviceMemory` holds the cache only as `IGpuResidency`, so it called the shared sweep directly and went
  around the wrapper: every sidecar alive at a model-swap boundary stayed on the card. The wrapper's body is now
  `GpuTransferHelper.State.FreeAllCached()`, so every route to the sweep runs it, and the static entry point delegates
  to it. Teardown and `EvictGpuCache` behave as before.
- **Changed: the DeepSeek-V4.1 host reference model keeps its weights in the checkpoint's own form.** Dense attention, shared-expert,
  compressor, indexer, Engram, embedding, head and routed-expert weights (FP8 E4M3 with E8M0 block scales, MXFP4, BF16) stay as
  mapped checkpoint bytes and are decoded one row window at a time inside each product (`DeepSeekV41Weight`,
  `WeightDequantizer.ToF32Rows`), instead of being widened to F32 at load. Results are bit-identical to the widened path, which stays
  available as `DeepSeekV41Residency.WidenedF32`. The Text path's resident set drops from about 31 GiB of dense F32 to the working set
  below; three real layers ran in 3 GiB of RSS.
- **Fixed: the RAM guard and the memory profile for a V4.1 load.** The guard now sizes the working set (sequence state, prefill
  activations at the 16,384-token cap, small widened tensors, Engram row caches, decode windows) plus a 4 GiB margin, from the
  config and headers alone (`DeepSeekV41WorkingMemory`), and `TextMemoryProfile` reports that figure as the phase's activation
  bytes instead of zero. `DeepSeekV41HostModelLoader.Load` itself does not run the guard, so callers that bypass `TextService`
  must size a load first.
- **Added: `DeepSeekV41LoadOptions.Residency` and `MaxLayers`.** `MaxLayers` loads only the first layers of a checkpoint, for
  checking real weights without the whole model.
- **Added: real-weight tests, skipped unless the weights are present and failing instead under `HARTSY_REQUIRE_REAL_WEIGHTS=1`.**
  Ten row windows of real tensors decode bit-identically to an independent torch decode
  (`tests/python-reference/deepseek_v41/dump_real_slices.py`); three real layers (dense, Engram, compressor and indexer) give finite
  hidden states and logits. Nothing is compared to upstream on the real weights yet, so no support-matrix row is Verified and the
  catalog entry stays Structural.
- **Changed: BF16 weights are widened in parallel by row** (a 129,280-row head window took 21 s single-threaded for one token).


## alpha.279

- **Added: a DeepSeek-V4.1 checkpoint directory now loads and generates through `TextService`.** `HfTextDirectoryLoader` no longer
  stops at "model class not wired": it builds a `DeepSeekV41TextModel` (the host reference model behind its `IGenerationModel`
  adapter, the byte-level BPE tokenizer read from the checkpoint's `tokenizer.json` with begin and end of sentence checked against
  `config.json`, and `DeepSeekV41Encoder` as the chat template), and the slot's shared `TextGenerationPipeline` drives it for
  `GenerateAsync`, `StreamAsync` and `CountTokens`. The model always loads on the CPU backend (a GPU device request logs that it is
  ignored), accepts at most 16,384 prompt-plus-generated tokens per request, and a load first checks free host RAM against the F32
  size of the dense weights. A directory without `tokenizer.json`, or with one that is not byte-level BPE, is refused by name.
  Covered on the small upstream fixture checkpoint; nothing has run on the real weights, so no matrix row moves to Verified, and
  the catalog entry stays Structural and not CLI-drivable.
- **Changed: `HfTextDirectoryLoader.Load` takes the backend and returns the loaded model.** The `NotSupportedException` it threw for a
  valid V4.1 directory is gone.

## alpha.278

- **Added: the V4.1 host reference model as an `IGenerationModel`.** `DeepSeekV41GenerationModel` adapts a loaded model so the shared
  `TextGenerationPipeline` can drive it: `Prefill` (last row or every row, with the chunk required to start at the committed length),
  `DecodeBatch` (one token per sequence, validated for every sequence before any moves), `ProjectLogits`, a state-size estimate and host memory capacity. It owns the loaded model, takes
  text tokens only (image embeddings are refused) and runs no speculation. `DeepSeekV41GenerationState` is its `ISequenceState`: the
  compressor's partial group and the window ring cannot be truncated in place, so `Truncate` resets and replays the kept prefix from the stored
  token ids, correct but a prefill's cost. The output head's matrix product now runs across cores, one sequential dot product per output so
  results are unchanged. A real greedy run through `TextGenerationPipeline` produces upstream's first token and the same sequence as a manual
  loop. The fixture-checkpoint writer used by the loader tests is now a shared test helper. Not wired into `TextService`.

## alpha.277

- **Added: loading an opened DeepSeek-V4.1 checkpoint into the host reference model.** `DeepSeekV41HostModelLoader` reads every backbone
  layer by canonical key (attention, compressor, indexer, gate, shared expert, hyper-connection coefficients, norms, embedding and head),
  dequantizing each weight to F32 through its bound recipe (`WeightDequantizer`), with the config's own logical shapes checked. Routed experts
  are not loaded up front: one `DeepSeekV41ExpertCache` shared across all layers dequantizes an expert the first time a token routes to it,
  bounded by `DeepSeekV41LoadOptions.ExpertCacheCapacity`. Engram tables stay on disk behind `EngramTableStore` (storage backing, per-table
  budget). Two rope tables are built once and shared: the plain one and the compress-theta YaRN one. `DeepSeekV41LoadedModel` owns the
  checkpoint and stores for the model's lifetime. This is the reference path, so dense weights are held as F32, roughly 30 GiB of host memory
  for the official checkpoint; draft, vision and DSpark tensors are not read. A real safetensors directory written from the model fixture,
  opened through `DeepSeekV41Checkpoint` with a full config, reproduces upstream's hidden states and logits through prefill and decode with a
  three-expert cache forcing evictions. The Engram branch is not covered end to end: its hash ids address the official 384-million-row tables.
  Not wired into `TextService`.

## alpha.276

- **Added: DeepSeek V4.1 block and whole-model forward on the host reference path.** `DeepSeekV41Block` composes the optional Engram
  lookup, attention and the routed feed-forward layer, each between hyper-connection mixing, with upstream's hand-off of coefficients
  (attention collapses with the previous block's feed-forward coefficients, the feed-forward layer with the ones attention just derived).
  `DeepSeekV41HostModel` adds the embedding, the per-copy stream, the final collapse and norm, and the output head. It prefills from
  position 0 as one chunk and runs a later multi-token call one token at a time, which attention requires. `DeepSeekV41SequenceState` holds
  each layer's attention cache, the slots shared between layers and the Engram hash history. A six-layer model covering every attention mode
  matches the unmodified upstream `Transformer` through an 11-token prefill and six decode steps (`dump_model_fixture.py`), to 1e-3.
  The attention and model fixtures now run upstream with an exact-softmax `sparse_attn` instead of the harness port, which rounds
  probabilities to bf16 like the real kernel; that rounding made the stack drift by up to a few percent and hid any real defect, and the
  attention test tolerance drops from 1e-2 to 1e-3. Engram is checked per component and for its hash-id wiring, not in the model fixture.
  Not wired into `TextService`.

## alpha.275

- **Added: DeepSeek V4.1 attention layer and hyper-connection wrapper on the host reference path.** `DeepSeekV41Attention`
  follows upstream `Attention.forward`: low-rank queries, the shared key/value latent with its rope and FP8 round trip, the
  sliding-window ring, the compressed positions (pooling `DeepSeekV41CompressorState`, the indexer with level-one candidate blocks
  and top-k via `DeepSeekV41IndexSelection`), `SparseLatentAttention` with the per-head sink, then the grouped `wo_a` and `wo_b`.
  It reuses the backend `ApplyRopeInterleaved`, `ActQuantDequantInPlace`, `BuildWindowIndices`, `IndexerScores` and
  `SparseLatentAttention` ops. Per-layer cache is `DeepSeekV41AttentionState`; `DeepSeekV41SharedAttention` carries the compressed
  KV, index keys, top-k indices and candidate mask between layers and, as upstream's runtime does, persists across forward passes.
  A call at position 0 starts a new sequence and clears the layer's state, later calls take one token, and cache capacity is checked
  before anything is mutated. `DeepSeekV41HyperConnection` is the mHC wrapper around a sublayer
  (`hc_mixes` projection with its flattened RMS statistic, then the shared `Hc*` ops). A six-layer stack covering window-only,
  source, reuse, candidate-source and candidate-restricted layers matches the unmodified upstream through an 11-token prefill and six
  decode steps (`dump_attention_fixture.py`), within 1e-2: the harness's `sparse_attn` rounds probabilities to bf16 like the real
  kernel, which this exact-softmax reference does not. Ties in indexer scores break toward the lower index; torch leaves their order
  unspecified, so the fixture keeps ties out of reach. Not wired into `TextService`.

## alpha.274

- **Added: DeepSeek V4.1 Engram module on the host reference path.** `DeepSeekV41EngramModule` looks up a position's
  `(max_ngram_size - 1) * n_heads` hash rows, projects them with `wkv` into one key per hyper-connection copy plus a shared
  value, and adds `gate * value` to each copy, where the gate is a sigmoid of the signed square root of a per-copy
  normalized dot product with its key; masked positions pass through untouched. The row lookup is injected as an
  `EngramRowGather`, the shape of `EngramTableStore.Gather`, so FP8 table decoding stays where it already lives. Checked
  against the unmodified upstream `Engram` with a bf16-rounded float table and float32 `wkv` (`dump_engram_module_fixture.py`);
  upstream's FP8 activation quantization before `wkv` is not modelled. Not wired into `TextService`.

## alpha.273

- **Added: DeepSeek V4.1 MoE layer on the host reference path.** `DeepSeekV41MoeLayer` runs the gate through the existing
  `IBackend.MoeRoute`, then `DeepSeekV41MoeExecutor` applies the routed SwiGLU experts (up clamped both ways, gate clamped
  from above, routed weight applied before the down projection, as upstream does) and the shared expert in F32, visiting
  experts in ascending order like upstream. Experts come from an `IDeepSeekV41ExpertSource`; `DeepSeekV41ExpertLoader`
  dequantizes one from the checkpoint with shape checks and `DeepSeekV41ExpertCache` keeps the hot ones. The new
  `WeightDequantizer` picks the host codec from a weight's bound recipe (FP8 block, MXFP4, NVFP4, MLX affine, EXL3) or
  widens unquantized BF16/F16, and refuses a packed dtype or a recipe-less descriptor rather than misreading it. The
  layer is checked end to end, gate included, against the unmodified upstream `MoE` with float32 weights. The loader's
  per-matrix read is tested on an in-memory BF16 expert and its refusals; reading a quantized expert from real shards is
  not, because the synthetic checkpoint's expert shapes do not match real widths. Not wired into `TextService`; the
  support matrix marks `core.moe.exec` InProgress with CPU implemented.

## alpha.272

- **Added: DeepSeek V4.1 host reference primitives for the CPU path.** `DeepSeekV41RopeTable` (interleaved cos/sin with
  upstream's YaRN, which floors and ceils the correction range and applies no mscale), `DeepSeekV41CompressorState`
  (softmax pooling of `compress_ratio` tokens with the partial group carried across prefill chunks and decode),
  `DeepSeekV41IndexSelection` (level-one candidate blocks and the indexer's sorted top-k) and
  `DeepSeekV41GroupedProjection` (block-diagonal `wo_a`). The first three are checked against fixtures dumped from the
  unmodified upstream `model.py`. The model class is still not wired, so V4.1 directories still refuse to load. The
  support matrix marks `core.spec` and `load.config` Implemented, which the code already was.

## alpha.271

- **Changed: the `kolibri1` catalog entry moves from Structural to ValidationPending.** A real-weight run on the swarm server (2026-10-07) loaded the Q2_K quant across a 4090 + 3060 and answered coherently for a few hundred tokens before degrading; the pinned Q4_K_M (44 GB) has not been run and needs an estimated 48 GB of VRAM because the text engine has no host offload. The catalog comment and the status tables record this, along with the live Swarm runs of ControlFoley (video, video + reference clip) and Clef-Flash.

## alpha.270

- **Fixed: the layer-split GGUF load uses the quantized-load RAM estimate, and Q2_K / Q3_K stay compressed on the GPU.**
  `LoadSharded` loads without dequantizing, like the single-device CUDA path, but its guard still demanded 2.5x the file,
  so a 26.7 GB Kolibri-1 Q2_K was refused on a 64 GB server ("need ~66.7 GB headroom"). It now needs 1.15x the file plus
  the F32 size of any tensor that is still expanded. The keep-compressed list (`GgufLanguageModel`) was missing Q2_K and
  Q3_K although the CUDA backend has dequantize and fused GEMV kernels for both, so those files were expanded to F32 on
  load (~290 GB for the Kolibri Q2_K). The tensor-parallel path keeps 2.5x. The layer-split sharding test now uses the
  production estimate for its RAM precondition.

## alpha.269

- **Fixed: the host-RAM guard no longer refuses a quantized GGUF a CUDA device loads without dequantizing.** `TextService`
  demanded 2.5x the file size free before every GGUF load, which is the cost of dequantizing onto host buffers. A backend
  that reads quantized weights (`SupportsQuantized`) keeps Q4_0/Q5_0/Q8_0/Q4_K/Q5_K/Q6_K compressed and reads them through
  the mmap, so it now needs 1.15x the file plus the F32 size of any other quantized tensor in it and of the token and
  per-layer embedding tables (those are always widened). Dequantizing backends, SSM models and the tensor-parallel and layer-split paths keep 2.5x. A 44 GB Kolibri-1
  Q4_K_M was refused on a 64 GB server with "need ~110 GB headroom".

## alpha.268

- **Fixed: a ControlFoley video or reference clip no longer needs a prompt.** `MusicService` refused a request with an empty
  prompt and genre before the model ran, but the official demo scores a video, or follows a reference clip, with no
  prompt at all. A ControlFoley request carrying `Video` or `ReferenceAudio` now passes that check; every other model still
  needs a prompt or genre.

## alpha.267

- **ControlFoley video-to-audio through the engine.** `MusicRequest` gains `Video` (an encoded clip), `MaskAwayClip` (the
  official `--mask_away_clip`) and `NegativePrompt`. A request with a video is decoded by the ffmpeg child-process
  decoder at the clip's native size and rate, decoded up to one second past the requested duration (the preprocessor drops frames beyond it), and conditions the CLIP, CAV-MAE-ST
  and Synchformer streams; the output is cut to the clip's usable length when that is shorter. The Synchformer and
  CAV-MAE-ST weights (`ext_weights/` of `YJX-Xiaomi/ControlFoley`) download on the first video request, so text-only use
  never fetches them. `FfmpegProcessDecoder` gets a `maxSeconds` limit (`-t`) for this.
- **ControlFoley reference audio through the engine.** `MusicRequest.ReferenceAudio` (WAV, the span
  `ReferenceStartSeconds`..`ReferenceEndSeconds`, mixed to mono) now conditions the generation on the clip's CLAP embedding
  and MusicGen-Style/MERT timbre, as `demo.py` does. The encoders load straight from the original checkpoints on first use
  (`music_speech_audioset_epoch_15_esc_89.98.pt` from the ControlFoley repo, `facebook/musicgen-style` `state_dict.bin`,
  `m-a-p/MERT-v1-95M`); no converted copies are needed.

## alpha.266

- **AuK review follow-ups.** A default (`--seed 0`) AuK run draws a fresh random seed and logs it instead of reusing one fixed noise, matching the documented "0 leaves it unset" contract. A Flash request that sets steps or CFG logs the discarded values. An explicit `--duration` is pinned as winning over the reference clip length, with tests, and `AukDuration.Frames` no longer rounds a float `0.6` (0.6000000238) up a frame. The Qwen2.5-Omni shard index sha256 is pinned. Checked against the real Hugging Face headers: every pinned sha and size matches, and every key and shape the AuK, VAE and Qwen loaders require exists.

## alpha.265

- **Added: ControlFoley video and reference-audio conditioning (Audio layer).** `ControlFoleyPipeline` takes an optional decoded
  source video (CLIP frames, CAV-MAE-ST visual tokens, Synchformer sync tokens, with the official frame sampling and a
  bit-exact bicubic resize) and an optional reference clip (CLAP audio embedding and MusicGen-Style timbre feature, with a
  julius-compatible resampler). Each encoder matches the official python on tiny random instances (~1e-6 to 1e-4) and on real
  weights (Synchformer ~1.4e-6 relative, CAV-MAE 1e-6, CLAP embedding 1.5e-7, timbre 6e-8). The official CAV-MAE-ST load is a no-op
  (all 649 checkpoint keys carry a `module.` prefix and `strict=False` ignores them, so official inference runs that branch
  with random weights); the port strips the prefix and loads the real weights. Not wired into the engine or Swarm: `MusicRequest`
  has no video field, mp4 decoding is not connected, and no pipeline-level run has been done (it needs the 11 GB network).
  Version bump: alpha.264 -> alpha.265.

## alpha.264

- **Added: ControlFoley text-to-audio (`controlfoley`, `hartsy music -m controlfoley`).** Native port of Xiaomi's flow-matching
  generator (54-block DiT with classifier-free guidance and the euler sampler), the DFN5B CLIP text/image encoder and
  tokenizer, and the 44.1 kHz VAE + BigVGAN v2 decoder. Each part matches the official python on tiny random checkpoints
  (~1e-6) and on real weights (decoder 7e-5 on a fixed latent, CLIP 6e-7, network 8.8e-5 velocity on a truncated real
  network). Matrices and convolution kernels can stay bf16 (`controlfoley_bf16.safetensors`, converted with
  `tools/controlfoley/convert_network_bf16.py`), which lets the 2.8B-parameter network run in 16 GB. Full-depth CPU run: 4 s
  of audio, 10 steps, 9 min, finite and non-silent. Video and reference-audio conditioning are not wired yet. Weights are
  CC-BY-NC-4.0. Version bump: alpha.263 -> alpha.264.

## alpha.263

- **Added: Cloudflare Clef-Flash typed decisions (`clef-flash`, `POST /v1/systemone`).** The joint schema head, a Qwen3.5
  trunk built from HuggingFace-keyed weights (`Qwen35Model.FromHuggingFace`, all-position hidden states), the record
  encoder and the Jev / SystemOne response shape. Text input only; images and video are rejected. The head and trunk
  match the official modules on tiny random checkpoints, and the encoder matches `encode_record` with the real
  tokenizer. Real weights, CPU: the README invoice and support examples answer correctly (overdue 0.97, technical 0.96,
  outage 0.84). New `Modality.Decision`, `IDecisionService`. Version bump: alpha.262 -> alpha.263.

## alpha.262

- **Added: Breeze TTS 2 (`breeze`) runs natively.** T5Gemma2 text encoder, Qwen3 backbone with audio-frame embedding
  sums, CSM-style depth decoder, classifier-free guidance against the template negative prompts, the Qwen3-TTS 12 Hz
  vocoder, and reference-clip encoding with the audio tokenizer's own Mimi encoder (`Mimi.LoadEncoderWeights`). Parity
  tests compare the encoder and the backbone/depth stack with the official modules on tiny random checkpoints. Real
  weights, CPU: plain and voice-design (cfg 4) runs transcribe word-exact, and a cloned-voice run transcribes
  word-exact. The weights are research / non-commercial. Version bump: alpha.261 -> alpha.262.

## alpha.261

IndexTTS-2 accuracy pass. Every stage of the 2.0 pipeline is now compared numerically with a dump of the PyTorch
reference's own intermediates (`IndexTts2V20PythonParityTests`, real weights, env-gated), and greedy decoding is
compared token for token. Earlier checks only listened to the output, so these divergences went unseen:

- **Fixed: the w2v-bert reference features were computed from unscaled audio.** The reference feature extractor feeds the
  fbank 16-bit-scaled samples; with `[-1, 1]` audio, quiet frames and weak bins clamped to the log floor. Feature error
  against the reference went from 8.7% to 5e-6 and `hidden_states[17]` from 16% to 1e-5. Affects 2.0 and 2.5 (speaker,
  emotion and codec inputs all derive from it).
- **Fixed: the IndexTTS Conformer used the wrong positional encoding.** IndexTTS's `RelPositionalEncoding` hands its
  attention the plain forward sinusoid `pe[0:T]` and the attention has `rel_shift` removed; the port used the
  Transformer-XL `2T-1` relative table. 2.0 speaker latents went from 11% to 1e-4 error. The same encoder class serves
  the emotion encoder of 2.0 and 2.5 and IndexTTS-1.5's speaker encoder, so all three change.
- **Fixed: generation used the wrong mel position embeddings.** The reference's `GPT2InferenceModel` looks up
  `attention_mask.shape[1] - mel_len`, giving the start token position 0 but the first code position 2 (position 1 is
  skipped). Greedy decode now reproduces the reference's codes exactly, with and without repetition penalty.
- **Fixed (2.0): the S2Mel prompt condition was built from the raw w2v-bert feature**, which is the 2.5 flow; 2.0 feeds
  the semantic codec's quantized embedding (`S_ref`).
- **Fixed: reference clips are resampled like the reference.** `SincResampler` reproduces
  `torchaudio.functional.resample` exactly, applied in the reference's order (file → 22.05 kHz → 16 kHz), and clips are
  capped at 15 s. The 16 kHz signal now matches to 6e-5 (was 3.4%).
- **Fixed: the repetition penalty did not include the reference's filler token** (`input_ids` are all ones plus the
  start token), and `remove_long_silence`, which the reference never calls, was applied. Both now match.
- **Behaviour changes to note:** output differs from alpha.260 for the same seed (the fixes above change the model's inputs
  and decoding); explicit emotion vectors are no longer normalized by default; `remove_long_silence` is gone; 2.0 inserts
  200 ms of silence between text segments. Version bump: `Directory.Build.props` `VersionSuffix` alpha.260 → alpha.261.
- **Changed: explicit emotion vectors are used as given** (the reference's library path and the Qwen text path never
  normalize them; only its WebUI does). `IndexTts2Options.NormalizeEmoVector` opts into the WebUI's bias + 0.8 cap.
- **Added (2.0): exact reference text handling.** Token-level segment splitting (`split_segments`, golden-tested against
  the real tokenizer's output, including `quick_streaming_tokens`), 200 ms of silence between segments, and the English
  spoken-form normalization the reference runs before tokenizing (contractions, numbers, years, decimals, percent, money,
  ordinals, times, titles, punctuation map). `"2.0"` is now read "two point oh" instead of being passed as digits.
- **Changed: the S2Mel DiT and `VitsWaveNet` run entirely on backend ops.** The DiT's AdaLN, SwiGLU product, residual adds,
  concats and transposes, and the WaveNet's conditioning add, gate, split and skip accumulation were host loops that read
  every activation through `DataPointer` — on a GPU backend each is a device-to-host round trip, dozens per DiT block per
  flow step (the live server needed about two minutes for six seconds of audio). `cond_x_merge_linear` is also split so
  the per-frame style broadcast concat is a single per-step vector. Numerics are unchanged (parity test and the VITS
  tests pass); speed on CUDA is measured after deploy from the new per-generation timing log.
- **Added: `IndexTts2Reference` / `PrepareReference` / `SynthesizeStream`.** The reference clip's front end (w2v-bert, mel,
  CAM++, conditioning) is computed once and reused, and synthesis can stream one text segment at a time. The engine's
  `indextts2` runner now implements `IStreamingTtsRunner`, caches the most recent reference clip and logs per-stage
  timings (`IndexTts2Timings`) for every generation.
## alpha.260

- **Fish Audio S2 voice cloning.** `ModifiedDacEncoder` ports the other half of the codec: the causal strided encoder
  (windowed transformer in its last block), the downsampling ConvNeXt stack, the pre-module transformer and the residual
  quantizer. It matches the official encoder on a tiny random checkpoint (every stage and the exact codes) and, on the
  real `codec.pth`, reproduced 690 of 690 codes of a generated clip. A request's reference clip plus its exact
  transcript now becomes the cloning system turn; a reference without a transcript is rejected. The conv/transformer
  helpers the encoder and decoder share moved into `DacOps`.
- The tiny Fish Audio S2 test checkpoints (about 2.4 MB) are now tracked: `.gitignore` re-includes
  `Fixtures/FishAudioS2/*.safetensors`. They were ignored with the other weights, so alpha.259's parity tests had no
  fixtures to load on a fresh checkout.

## alpha.259

- **Fish Audio S2 Dual-AR model** (`FishAudioS2DualAr`; not yet registered as a TTS model — the codec and pipeline are
  still to come). The 36-layer slow transformer and 4-layer fast transformer run on the shared `GenericTransformer`
  (per-head Q/K norm, interleaved RoPE, tied text embedding) with BF16 weights kept in their stored dtype. The frame
  step follows fish-speech's `decode_one_token_ar`: the slow head is constrained to the semantic range plus
  `<|im_end|>`, Repetition Aware Sampling re-draws a repeated semantic token, code 0 is derived from the main token,
  and the fast model takes the post-norm slow hidden state. Sampling filters on the untempered distribution as upstream
  does. Slow logits, post-norm hidden states and fast logits match the official `DualARTransformer` on a tiny random
  checkpoint to 2e-6 (`tools/fish_audio/s2_dual_ar_reference.py`).
- **Fish Audio S2 codec decoder** (`ModifiedDacDecoder`): the code-to-waveform half of fish-speech's ModifiedDAC — the
  semantic + nine residual codebooks, the 8-layer window-limited causal transformer, the causal ConvTranspose/ConvNeXt
  upsampler and the causal Snake/residual-unit decoder, 2048 samples per frame at 44.1 kHz. Every stage matches the
  official implementation to about 1e-7 on a tiny random checkpoint, and decoding with the real released `codec.pth`
  matches to 3e-6 (`tools/fish_audio/modded_dac_reference.py`, `ModifiedDacRealWeightTests`). Reference-audio encoding
  is not ported yet.
- **Fish Audio S2 Pro text-to-speech** (`fishaudio`, status ValidationPending). `FishAudioS2Prompt` builds fish-speech's
  conversation prompt (system / user / open assistant turn, `<|speaker:N|>` batching, assistant turns carrying earlier
  batches' codes), `FishAudioS2Pipeline` runs the Dual-AR loop to `<|im_end|>` and decodes with `ModifiedDacDecoder`,
  and `FishAudioS2Model` registers it in the speech catalog (`fishaudio/s2-pro`: tokenizer, `codec.pth`, sharded
  weights). A real-weight CPU run (seed 7) produced 3.2 s that Whisper base.en transcribed word-exact. Reference-voice
  cloning is refused with a clear message until the codec encoder is ported. The weights are non-commercial
  (Fish Audio Research License).

## alpha.258

- Add the `kolibri1` text-model catalog entry (`Hob-forge/Kolibri-1-GGUF`, Q4_K_M, SHA-256 pinned). Status stays
  Structural: no real checkpoint has been run yet (the smallest quant is 28.6 GB).
- **Fixed: Kolibri-1 tokenization.** Its `tokenizer.ggml.pre` is `kolibri1` (llama.cpp's Qwen2 pre-tokenizer), which
  splits digits one at a time and matches contractions case-insensitively. The engine used the GPT-2 default, which
  groups digit runs and so produced different token ids for any prompt containing numbers.

## alpha.257

- Add Kolibri-1 native GGUF support: key mapping for sandwich norms, Q/K norm, MoE tensors and the router bias;
  local/global attention with NoPE; and Kolibri's sigmoid-logit-add expert routing, which requires the correction bias.
- **Fixed: Kolibri-1 GGUFs failed to load.** The published converter emits only the expert FFN lengths, but the loader
  required `kolibri1.feed_forward_length`. It now falls back to `expert_feed_forward_length`, then
  `expert_shared_feed_forward_length`, and still throws if none is present.

## alpha.256

- Add the Breeze TTS 2 architecture and checkpoint configuration contract. Its Mimi codec is the shared
  `MimiConfig.Mimi24kHzDsm` preset, and `MimiConfig.FrameRateHz` reports the true 12.5 Hz output rate.
- Keep Fish Audio S2's 4096-entry fast-decoder vocabulary distinct from ModifiedDAC's 1024-entry residual
  codebooks.

## alpha.255

- **IndexTTS-2.0 (`indextts2:2.0`).** The 2.0 checkpoint (`IndexTeam/IndexTTS-2`) now runs next to 2.5, which stays the
  default. It differs from 2.5 in the GPT's speaker conditioning (a Conformer+Perceiver over the w2v-bert feature with
  real `speed_emb` slots, instead of CAM++), the tokenizer (SentencePiece), the semantic codec (MaskGCT's RepCodec,
  from `amphion/MaskGCT`) and the handoff to the flow-matching stage: a second GPT pass through `s2mel.gpt_layer`,
  summed with the codec's `vq2emb`. `IndexTts2Version` picks the branch and the 2.5 path is unchanged. A real-weight
  generation transcribes through Whisper-base as its input text.
- **Emotion control is exposed for both versions.** `SpeechRequest` gains `EmotionReference` (a clip whose emotion,
  not voice, the speech adopts), `EmotionAlpha`, `EmotionText` and `EmotionFromText`; the existing `Emotion` vector
  is read in IndexTTS-2's own order (happy, angry, sad, afraid, disgusted, melancholic, surprised, calm — not
  Zonos's). `hartsy speak` gets `--emotion`, `--emotion-reference`, `--emotion-alpha`, `--emotion-text` and
  `--emotion-from-text`. The QwenEmotion classifier is loaded on first use, so the default VRAM footprint is unchanged.
  The model's file list now includes `feat1.pt`, `feat2.pt` and the four classifier files (about 1.2 GB), so an
  existing 2.5 install downloads them on its next load. The classifier files are optional: if they are missing,
  free-text emotion is disabled and the model still loads. Emotion weights outside 0–1.2 are rejected.
- **Fixed: the explicit emotion-vector mode threw `ObjectDisposedException`.** The pipeline disposed the
  `feat1.pt`/`feat2.pt` loaders right after building the lookup, but an F32 tensor is returned as-is rather than
  copied, so the exemplar banks it still read were freed. The loaders now live as long as the pipeline. Nothing
  reached this path before, because the engine never loaded the banks.
- **Fixed: `EmoAlpha` was ignored for an explicit or text emotion vector.** The reference scales the vector by the
  clamped alpha (truncated to four decimals); the port now does too.

## alpha.254

- **Kokoro-82M: Japanese and Mandarin voices.** `j` and `z` voices, which raised an error in alpha.252, now read their
  text the way misaki does for the official pipeline. Both are pure C#, with no Python, MeCab or native library.
  - `j` (Japanese) ports misaki's `JAG2P` in its default cutlet mode. A MeCab tokenizer reads full UniDic 3.1.0, the
    dictionary fugashi uses under misaki, and agrees with fugashi on every token of 474 test sentences. The kana
    reading, number reading (num2kana), width folding and misaki's kana-to-phoneme table follow. Its output matches
    `JAG2P()` exactly on all 474 sentences (`KokoroJapaneseParityTests`).
  - `z` (Mandarin) ports misaki's `ZHG2P`:
    - cn2an number normalisation;
    - jieba 0.42.1 segmentation, including its HMM for unknown words;
    - pypinyin 0.55.0 readings with its phrase matching;
    - misaki's pinyin-to-IPA conversion and tone marks.

    Its output matches `ZHG2P()` exactly on all 397 test sentences (`KokoroMandarinParityTests`).
  - Their data is fetched on first use, SHA-256 checked:
    - Japanese: UniDic 3.1.0 (a 501 MB archive, of which only the five files MeCab reads are kept, about 690 MB
      installed; BSD licence) and misaki's `ja_words.txt` from the misaki commit the English dictionaries use.
    - Mandarin: jieba's dictionary and HMM table, and pypinyin's two dictionaries (all MIT, about 10 MB).
- **Kokoro-82M: voice blending.** A voice may name several packs, as `KPipeline.load_voice` allows: `af_bella,af_sky`
  averages them row by row, matching `torch.mean` to 6e-8. A part may carry a weight: `af_bella:0.7,af_sky:0.3` or
  `af_bella(2)+af_sky(1)`. Voice names are checked before they reach a file path.

## alpha.253

- **IndexTTS-2.5: emotion-controllable zero-shot voice cloning** (`indextts2` catalog entry; zero-shot
  cloning wired end to end, emotion control modes not yet exposed through `TtsJob`/CLI/HTTP). Shares the
  GPT-2 T2S decoder shape with IndexTTS-1.5 but conditions on CAM++ speaker embeddings instead of a
  Conformer-Perceiver, and generates semantic-codec codes rather than mel frames directly. New pieces:
  - `Wav2Vec2Bert` (new `Models/Wav2Vec2Bert/` folder, not IndexTTS-scoped): a 24-layer, 1024-hidden
    Conformer with Shaw-style relative-key position bias (`facebook/w2v-bert-2.0`), taking hidden-state
    layer 17 of 24 — not the final layer, confirmed from the real `infer_v2.py`.
  - `VocosFactorizedCodec`/`VocosFactorizedCodecConfig` (generic Amphion/MaskGCT-family semantic codec:
    `VocosBackbone` encoder → `in_project` conv → L2-normalized single-codebook nearest-neighbor lookup
    `out_project` conv, with the decoder half built but never invoked on the real inference path).
  - `VocosBackbone`: the first config-driven, shared copy of the ConvNeXt-based Vocos backbone shape this
    codebase had already copy-pasted five times under one-off names.
  - `IndexTts2T2sDecoder`: extends the GPT-2 T2S decoder with a second, smaller Conformer+Perceiver for
    emotion conditioning (`IndexTtsPerceiver.NumLatents` generalized from a hardcoded 32 to a constructor
    parameter so both uses share one class), the real `merge_emovec` lerp, and an explicit emotion-vector
    lookup table (`IndexTts2EmotionVectorLookup`, cosine-similarity nearest-exemplar per category from
    the checkpoint's `feat1.pt`/`feat2.pt`). Also replicates the checkpoint's two zero-initialized
    "duration slot" conditioning rows for shape fidelity — confirmed inert in the real source
    (`speed_emb` is never exposed via the public `infer()` API); not a working duration control.
  - `IndexTts2Dit`/`IndexTts2DitBlock`: the S2Mel flow-matching stage — a 13-layer U-ViT-skip DiT (RoPE
    attention, SwiGLU FFN, AdaLN norms) with a WaveNet final stage, solved via `ConditionalCfm` (widened
    to accept any `ICfmEstimator` shape, not just CosyVoice's) over 25 Euler steps with `(1+rate)*cond -
    rate*uncond` CFG. `VitsWaveNet.LoadWeights` gained an injectable key suffix so VITS and this S2Mel
    stage share one WaveNet implementation. `InterpolateLengthRegulator` is a new, generic
    explicit-target-length nearest-neighbor regulator (distinct from `VitsLengthRegulator`'s
    duration-predictor shape).
  - `IndexTts2BigVganGenerator`: a slimmer top-level BigVGAN-22kHz assembly (no embedded speaker encoder,
    unlike IndexTTS-1.5's) reusing `AntiAliasedSnake`/`IndexTtsBigVganResBlock` verbatim.
  - `IndexTts2QwenEmotion`: free-text emotion classification via a bundled Qwen3-0.6B fine-tune, loaded
    through `HartsyInference.LLM`'s existing `GenericTransformer`/`TextGenerationPipeline`/
    `JinjaChatTemplate`/JSON-grammar-constrained sampling — no new LLM infrastructure, just a new
    `Qwen3HfConfigReader` (`config.json` → `TransformerConfig`, placed as `GgufConfigFactory`'s sibling,
    not IndexTTS-scoped) and this model's own fixed system prompt/label vocabulary.
  - `IndexTts2TiktokenTokenizer`: the 2.5 repo's tiktoken-format text tokenizer (`multilingual_zh_ja_
    yue_char_del.tiktoken`, 58,836 ranks + 1,673 specials = 60,509, matching `number_text_tokens`
    exactly), built on the existing `TiktokenConverter`/`HfTokenizerJson` rank-file parsing rather than a
    new parser.
  - IndexTTS-2.0 is not wired (shares every config value with 2.5 except `number_text_tokens`/tokenizer,
    but needs its own GPT `conformer_perceiver` speaker-conditioning path that doesn't exist yet).
  - Real end-to-end generation verified against the real checkpoints: 2.47s of finite, non-silent 22050 Hz
    PCM (RMS 0.225) from a real reference clip.

## alpha.252

- **Kokoro-82M: all its stock languages, each read the way the official pipeline reads it.** The voice's first letter
  picks the front-end, as `KPipeline` does. Before this, every voice went through the American G2P, so British
  voices spoke American English and Spanish, French, Hindi, Italian and Portuguese voices read their text as if it
  were English.
  - `b` voices (British) use misaki's British mode: the `gb_gold`/`gb_silver` dictionaries (same pinned commit,
    SHA-256 checked), British -s/-ed/-ing endings, no flap, and espeak en-gb for words misaki lacks. 98.6% word
    agreement with misaki `G2P(british=True)` on 300 sentences, the same as American.
  - `e`/`f`/`h`/`i`/`p` voices (Spanish, French, Hindi, Italian, Brazilian Portuguese) go through the new
    `KokoroEspeakG2P`, a port of misaki's `EspeakG2P` over the pure-C# espeak. Word agreement with misaki:
    es 99.9%, fr-fr 99.6%, it 100%, pt-br 100%, hi 99.5% (`KokoroEspeakParityTests`, 475 sentences).
  - `j`/`z` voices (Japanese, Mandarin) now fail with a clear error instead of reading their text as English.
  - Kokoro installs espeak-ng data into the model cache on first use. This is the data misaki phonemizes with,
    espeak-ng 1.52, taken from the pinned `espeakng_loader` 0.2.4 wheel (about 9 MB, SHA-256 checked, GPL-3.0).
    The other espeak models find the same copy.
- **espeak port: the language-dependent parts of espeak-ng 1.52**, which only English had before. Other languages
  missed the stress, numbers and dictionary choices these govern:
  - Per-language stress rules and flags (`SetWordStress`, every stress rule).
  - Letter groups.
  - Numbers read as words (`numbers.c`: cardinals, thousands, decimals, ordinal suffixes, lakh grouping).
  - Conditional dictionary entries (`$atend`, `$atstart`, `$noun`, `$verb`, `$past`, `$only`, `$capital`) and
    multi-word entries.
  - `$text` replacements, `.replace` character tables, `$alt`/`$alt2` vowel quality, and `$pause`/`$brk` pauses.
  - Prefix and suffix stripping as `TranslateWord3` does it.
  - Rules can now see the neighbouring words in the clause.
  - Doubled consonants, symbols (% ° & +), and hyphen-joined words written without a space.
- **espeak port: bug fixes that also affect English (Piper, StyleTTS 2, Zonos, NeuTTS, ZipVoice):**
  - A phoneme program's `NextVowelStarts` block was read one word short. This inserted a stray `l` after every
    English `r` ("θɹlˈuː").
  - A phoneme's IPA name kept the bytes after its terminator.
  - A word starting with a pause lost its leading space.
  - Words with a suffix at the very start of a sentence could throw `IndexOutOfRangeException`.
  - Quote marks were read as part of the word ("'I" read as the letter i).
  - Function words marked `$u+` were stressed mid-sentence.
  - One phonemizer shared by concurrent requests could mix their words. The rule matcher kept its per-word vowel
    counts, the phoneme interpreter its render-pass flag, and the IPA renderer its scratch buffer on the shared
    instance. They are now held per call, and a test checks that parallel reads match serial ones.

  English sentences matching espeak-ng 1.52 exactly: 59 → 377 of 400 (`EspeakSentenceParityTests`). Single words:
  385 of 400 (`EspeakParityTests`, floor raised to 95%; its fixture is regenerated from espeak-ng 1.52 by
  `tools/kokoro/espeak_parity_reference.py`).

## alpha.251

- Add Breeze TTS 2, Kolibri-1, Clef, ControlFoley, and Fish Audio S2 model contracts.

## alpha.250

- Add the Fish Audio S2 architecture contract and research baseline for the new Dual-AR TTS family.
- Add the Kolibri-1 checkpoint contract for the upcoming MoE and FP8 runtime path.

## alpha.249

- **Kokoro-82M: output now matches the official model.** Reported as "works but sounds bad"; three faults, all
  backend-independent:
  - The F0/energy predictor read the length-regulated PLBERT features instead of the length-regulated
    DurationEncoder output the reference feeds it (`model.py`: `en = d @ pred_aln_trg`). Same shape, so nothing
    failed, but pitch was off by ~74 Hz on average and a fifth of frames had the wrong voicing. The voice-pack
    style row was also two rows late (`pack[len(ps)-1]` counts phonemes, not BOS/EOS). Against the official
    PyTorch `KModel` on the same phonemes, durations are now identical and F0 is within 0.001 Hz
    (`KokoroProsodyParityTests`, reference from `tools/kokoro/prosody_reference.py`). StyleTTS 2 shares the fix.
  - The English G2P was a CMUdict mapping that agreed with misaki (Kokoro's training phonemizer) on 44% of words:
    every monosyllable stressed, no flap, curly apostrophes splitting words ("don’t" → "don tee"), numbers and
    currency misread. `EnglishG2P` is now a port of misaki's English G2P over its gold/silver dictionaries
    (Apache-2.0, fetched once, SHA-256 pinned) with context-dependent function words, -s/-ed/-ing morphology,
    acronyms, numbers, years, currency and quotes; CMUdict, the espeak port and letter rules cover words misaki
    lacks. 98.7% word agreement with misaki on 527 sentences (1 → 419 exact). The CMUdict-only
    `EnglishG2P(string)`/`(Stream)` constructors remain, obsolete; their output changes too (context-dependent
    "the"/"to", numbers read through the fallback), since they now run the same front-end over an empty lexicon.
  - Non-streaming synthesis (CLI, HTTP, Wyoming) handed PLBERT the whole text and threw past 512 phonemes. Input
    is now split on newlines and chunked at 510 phonemes on the strongest pause, as `KPipeline` does.
- `IndexTtsConfigValuesTests` used `var`, which failed the Audio test project's build under code-style enforcement
  (IDE0008); now explicitly typed.

## alpha.248

- **IndexTTS-1.5: nucleus (top-p) sampling and tail fade-out, found by auditing against the real `index-tts`
  Python source.** `IndexTtsT2sDecoder`'s AR sampler only applied top-k + temperature; the reference CLI's
  default decode stack also applies top-p=0.8 nucleus filtering after top-k (it decodes through HF
  `generate()` with both `top_k=30` and `top_p=0.8`). Swapped to the shared `NucleusSampler.Draw` (already used
  by CosyVoice/Spark-TTS) so the same temperature → top-k → top-p → multinomial-draw pipeline applies here too,
  layered under the existing CTRL-style repetition penalty. Also: `IndexTtsPipeline.Synthesize` now applies a
  20 ms raised-cosine fade-out to the very end of the generated waveform, matching the reference's
  `fade_out_tail` (`utils/common.py`) — decoding is stochastic, so the stop token occasionally samples early and
  the last PCM sample lands far from zero, producing an audible click (and sometimes a burst of noise, since
  BigVGAN's receptive field is incomplete right at that boundary); on a normal generation ending in silence the
  fade is a no-op. The reference's other default, `num_beams=3` (beam search combined with sampling via HF
  `generate()`), remains unimplemented — a materially bigger change (parallel beams/caches, score tracking),
  tracked as a known Phase-2-scale gap, not fixed here.

## alpha.247

- **IndexTTS-1.5: fix a load crash on the real checkpoint's integer buffers.** `IndexTtsPipeline.LoadAsync`'s
  `ToF32` helper (added in alpha.246 to centrally fix BF16-checkpoint leaks) was unconditionally casting every
  tensor in each loaded weight dictionary to F32, including non-floating-point buffers the real `gpt.pth`
  checkpoint carries (GPT-2-style integer/boolean position-id and causal-mask buffers) — `EnsureF32` has no I64
  source case, so the pipeline failed to load with "Unsupported dtype conversion: I64 -> F32." the first time it
  ran against the real checkpoint end to end (via a live deployment, not the unit test suite's all-F32 synthetic
  weights). `ToF32` now passes non-floating-point tensors through unchanged, matching every downstream consumer's
  assumption that those buffers are never read as weights.

## alpha.246

- **IndexTTS-1.5 speech model, structural port (Phase 1: zero-shot cloning only).** `hartsy speak -m indextts
  --reference <wav> "text"` clones a voice from a reference clip: a Conformer-Perceiver speech-conditioning
  encoder feeds a standard biased HF GPT-2 text-to-speech decoder (sampled autoregressively, two-pass — generate
  codes, then re-extract the matching final-layer hidden states), and a custom 24 kHz BigVGAN-v2 vocodes those
  hidden states directly, conditioned by an embedded ECAPA-TDNN speaker d-vector. No emotion or duration control
  (IndexTTS-2, not yet implemented) and no codec decode step — the shipped `dvae.pth` is training-only, confirmed
  unused in the reference `infer()`. Not numerically verified against the reference yet; see
  `MODEL_STATUS_AUDIO.md` and `docs/Research/INDEX_TTS_ARCHITECTURE.md`'s implementation-notes addendum for open
  risks (exact text-frontend CJK handling, sampling defaults).
- `GptBackbone`/`GptBlock` now load bias-or-zero for every projection (previously hardcoded bias-free), so a
  standard biased HF GPT-2 checkpoint loads correctly alongside Bark's bias-free one; `GptBackbone.Forward`/
  `ForwardStep` take an optional `positionsApplied` flag and `LoadWeights`' `posKey` is now nullable, for
  checkpoints (like IndexTTS) with per-segment position tables or none at all rather than one table spanning the
  whole sequence.

## alpha.245

- **AuK: skip per-call weight eviction when VRAM has room.** `AukOptions.SequentialResidency` defaults to `true`,
  and every stage (audio tower, thinker, DiT, VAE) freed its device weights at the end of every single call
  regardless of free VRAM, so back-to-back generations each paid a full host->device re-upload of every stage.
  Measured on a live 4090, this was ~3s of fixed overhead per generation. `Generate` now only evicts a stage when
  the device doesn't clearly have room to hold every stage the call touches resident at once; a backend that can't
  report VRAM (CPU) is unaffected. Output is unchanged either way.

## alpha.244

- **AuK and AuK-Flash (Tencent) speech model, structural port.** `auk:flash` (4 fixed steps, no guidance) and
  `auk:base` (32 steps, CFG 2.0, sway) run zero-shot voice cloning and instruction-described voices through
  `hartsy speak -m auk:flash` (`--reference`, `--instruction`, `--duration`) and the speech API. The pipeline is
  task-agnostic: an instruction plus a source clip drives the same DiT for editing, enhancement and separation, but
  only TTS is exposed by the engine in this release. Conditioning comes from the Qwen2.5-Omni-3B thinker (text LM
  plus audio tower; only shards 1 and 2 are downloaded) fused over all 36 hidden states; audio is decoded by the
  24 kHz BigVGAN-flow VAE. Not numerically verified against the reference yet (see `MODEL_STATUS_AUDIO.md`).
  The Qwen2.5-Omni encoder is under the Qwen Research License; the AuK weights are MIT.
- `GenericTransformer.ForwardEmbeds` and `Qwen2Model.ForwardEmbeds` take an optional per-layer tap; behavior is
  unchanged when it is null.
- `SpeechRequest` gains optional `Instruction` and `DurationSeconds`.
- bf16 repack recipes `auk-base`, `auk-flash` and `auk-vae` for `CheckpointRepacker`; nothing is uploaded.

## alpha.243

- **Fixed: a CUDA op runs in its own backend's context even when another copy of the engine left a different one
  bound to the thread.** SwarmUI loads a private copy of the engine per extension (AudioLab, LLMAssistant, the image
  backend), all on one thread pool. `CudaContext.EnsureCurrent` remembered each thread's binding in a `[ThreadStatic]`
  field, which each copy keeps separately, while the driver's binding belongs to the thread: once LLMAssistant bound
  the 3060 on a pool thread, AudioLab's copy trusted its own stale note and ran Whisper with the 3060 current. It read
  the 3060's free VRAM as the 4090's (and evicted on that reading), allocated Whisper's resident weights in the 3060's
  context, failed an upload into one with `CUDA_ERROR_INVALID_VALUE`, and the next request hit
  `CUDA_ERROR_ILLEGAL_ADDRESS` (700). That error is sticky on the 4090's context, which every copy shares, so image
  generation failed until a restart. `EnsureCurrent` and `EnsureRetainedCurrent` now confirm the binding with
  `cuCtxGetCurrent`, a thread-local driver read, before trusting the cache. Every extension has to take this version:
  a fixed copy binds its own context, but an unfixed one still runs in whatever context it finds.
- **Fixed: a host's setting can no longer lose to the settings file.** The file is loaded on the first knob read, and
  that load let other threads through as soon as it started. So a `KnobStore.Set` made on another thread mid-load was
  overwritten by the rest of the file, and a `Set` made before any read was overwritten by the load that the first
  read triggered. Hosts worked around it by reading a knob before setting one. Now a read waits for a load in progress
  to finish. `KnobStore.Set`, `KnobStore.Clear`, `KnobFile.Apply` and `KnobFile.Save` load the file first, so a
  host's value lands on top in any order. `Save` also stores the value it was given, coerced, instead of reading the
  knob back. A first-in-process `Save` could otherwise write the file's old value again. A missing explicit
  settings file or a malformed one now surfaces from the first `Set` as well as from the first read. This was the
  order-dependent `ModelFolderCaseTests` failure: a temp models root was replaced by the settings file's
  `paths.modelsRoot`.
- **Fixed: LTX two-stage refinement is refused on distilled checkpoints older than 2.5, even when
  `numerics.ltx2TwoStage` asks for it.** The x2 latent upsampler is an LTX-2.5 model. Making two-stage opt-in had
  moved the decision to the knob, and the knob applied to any distilled checkpoint, so the 2.5-only check that the
  distilled contract used to make was lost. `LtxVideo2Recipe.TwoStageRefusal` makes it again.
- **The CPU test lane passes on a machine with no GPU.**
  - Tagged `GpuIntegration`: the ten test files that construct a CUDA backend with no device check. Seven are in
    Cuda.Tests (51 cases); the other three are `WanAnimate2*` and `HunyuanImageVaeEncoderRealWeightTests`, which is
    also tagged `Integration`.
  - `TestTierLintTests` no longer accepts the `PtxDir()` helper's `Directory.Exists` as a GPU guard, so it now catches
    tests like these.
  - `Glm4SyntheticParityTests` is tagged `Integration` and skips unless its gitignored tensors are present.
  - The YuE Stage-1 prompt tests now expect the `split_lyrics` format that the tokenizer was moved to.
  - `KnobFileTests` no longer depends on what ran before it.
  - Two flaky tests were stabilized. `LatencyHistogram`'s allocation check takes the least of three passes. The
    `TextStreamPump` timing tests run warmed up, on their own.

## alpha.242

- **Fixed: the last raw thread-pool fan-outs on host paths now obey the CPU thread cap too.** `FluxRope`'s host
  Q/K rotation, `Nvfp4Linear`'s BF16 dequant, `VideoRgbFrames.ExtractAllFrames`, the CUDA backend's host W8A8 weight
  quantization and GPT-OSS's CPU-backend expert loop used raw `Parallel.For` / `Parallel.ForEach`, ignoring
  `numerics.cpuThreads` and `CpuParallel.InlineScope`. They now go through `CpuParallel`, `FluxRope` in ranges of 1024
  vectors through `CpuParallel.ForRanges`. The GPT-OSS loop runs a few lanes (at most half the cores, 8, and the cap),
  each pulling experts from a shared counter and reusing one pair of dequant slices allocated on its first expert, so
  slice memory stays bounded as before and the dequant and GEMM inside each expert nest on the same capped scheduler.
  Outputs are byte-identical at any cap, and no raw `Parallel` loop remains outside `CpuParallel` itself.
- **Bounded the opt-in prefix-KV reuse (`TextRequest.PrefixCacheKey`) for GPUs shared with other models.** A
  retained KV cache too small for the next request now grows by copying its reusable prefix on device instead of
  being dropped and prefilled again, which also stops a tool round with a large result from re-prefilling the rest
  of its turn. What a request retains is copied down to its length plus `vram.prefixCacheHeadroomTokens` (new,
  default 256) instead of keeping the whole allocation, and a sequence whose kept size would exceed
  `vram.prefixCacheMaxBytes` is freed after its request rather than retained; the store refuses one too, so the
  cap is hard. `vram.prefixCacheMaxBytes` now defaults to 1.5 GiB (was 512 MiB, which the newest entry could
  exceed): now that it also bounds each entry, it has to hold one voice call at its history ceiling on Qwen3-4B —
  the measured 609-token system-and-tools prefix, the 3,000-token history budget and a 201-token reply allocation,
  about 3,750 tokens at 288 KiB of F32 KV per token (~1.03 GiB); 512 MiB would drop that call's cache mid-call
  past ~1,800 tokens. `IBackend.ScatterSeqHeadMajor` gains a row-count overload and copies any byte-addressable
  dtype on CPU, CUDA and Vulkan; `FixedKvCache.CopyWithCapacity` and `IGenerationModel.ResizeSequenceState` expose
  the resize. `PrefixCacheCapacityHint` now only sizes a sequence's first allocation, so the voice session no longer
  passes one.
- **Fixed: host weight conversions, Mimi's RVQ encode and UnivNet's LVC gate no longer bypass the process CPU
  thread cap.** Large `Tensor.CastTo` and `DequantFp8E4M3ScaledToF16` casts, the fp8 quantizer's absmax, scale and
  stochastic-round passes, the NVFP4, MXFP4, FP8-block, affine, EXL3 and INT8-ConvRot host codecs,
  `LoraBaker.MatMulFma`, the MiniMax-H3 rebasers, Mimi's split-RVQ encode and Resemble-Enhance's UnivNet each fanned
  out with a raw `Parallel.For` on the shared thread pool, ignoring `numerics.cpuThreads` and
  `CpuParallel.InlineScope`. Row and tile loops now go through `CpuParallel.For`, with per-row scratch rented from
  `ArrayPool`, and the range passes through a new `CpuParallel.ForRanges`, whose ranges depend only on the length, so
  every output is byte-identical at any cap. On a host that lowers the cap (the voice host's unit runs with
  `engine.cpuThreadCap: 14`) these now use at most that many threads, so checkpoint conversion and LoRA baking there,
  and Mimi's RVQ encode (Kyutai STT, CSM) and the UnivNet vocoder at inference, can take longer on a machine with more
  cores than the cap.
- **Audio model switches now size the incoming model before deciding whether to evict.** `AudioRuntime` used to
  unload the other resident audio models on a switch only when free VRAM was under a fixed 3 GiB, so a 6-7 GB model
  arriving with 3-6 GB free evicted nothing: Dia then failed in `PreloadWeights` with `OutOfVramException`, and
  Orpheus loaded with most of its weights left host-side and streamed them every step. A switch now evicts when free
  VRAM is under the incoming model's need plus room beside it (a fifth more, and never less than
  `vram.autopromoteHeadroomMb`, because a weight that would leave less than that free is streamed instead of made
  resident), with the new `vram.audioEvictFreeVramFloorMb` (default 3072) as the floor. The need is the larger of
  what the model's latest load in this process left in use and the size of its weight files on disk, with F16/BF16
  tensors counted at F32 for a runner that widens them (Dia declares it). A model that is already loaded needs only
  the floor; same-model repeats still never evict, and pinned runners are still never evicted. If a run still throws
  `OutOfVramException`, the runtime unloads every other unpinned audio model, releases device memory, logs what it
  dropped, and retries once; a stream retries only if it has not yielded anything yet. Replaying the sweep on an RTX
  4090 so that Dia arrives with 3.9 GiB free: before this change Dia failed with the same driver refusal and Orpheus
  took 292 s to generate 6.2 s of audio against 7.5 s on a free card; with it the switch to Dia unloads the five
  earlier models, Dia succeeds and Orpheus takes 7.6 s, and every model's audio is byte-identical.
- **A cancelled prompt prefill now frees the GPU within about two transformer layers.** `IBackend` gains a fence
  pair — `RecordFence` / `WaitFence`, plus `ReleaseFence` — with no-op defaults: CUDA records a pooled event on the
  compute stream, Vulkan submits the batch recorded so far and hands back the timeline tick it signals, and the CPU
  backend has nothing to wait for. While the request's token can be cancelled, `GenericTransformer.ForwardEmbeds`
  waits, before issuing layer *k*, on the fence recorded after layer *k − 2* (`DeviceRunAhead`). The device always
  has the next layer queued behind the running one, and a stop leaves at most two layers to drain instead of
  everything the host had queued. Math, kernels and stream order are unchanged; a forward with a token that cannot be
  cancelled takes no fences at all.
- **Fixed: a large `NativeBuffer`'s zero-fill no longer bypasses the process CPU thread cap.** Buffers of 8 MB and
  up were zeroed with a raw `Parallel.For` sized by `Environment.ProcessorCount` on the shared thread pool, ignoring
  `numerics.cpuThreads` and `CpuParallel.InlineScope`, so one big allocation on any thread could take every core
  from the voice front end's real-time audio thread. The fill now goes through `CpuParallel.For` in fixed 2 MB
  chunks whose count depends only on the size: inside an `InlineScope` it runs on the calling thread, and the cap
  bounds how many threads clear at once. Smaller buffers still clear inline, and the memory is zeroed exactly as
  before.
- **Kokoro stops a cancelled synthesis at its next stage instead of finishing the sentence.**
  `KokoroPipeline.Synthesize`/`SynthesizeFromStyle` and `KokoroIStftNetDecoder.Forward` take a `CancellationToken`
  and check it at twelve stage boundaries — before any device work, after PLBERT, the text encoder, the duration
  predictor, the length regulator and F0/N, after the decoder's encode and decode blocks, after the harmonic source,
  after each upsample stage and before the iSTFT head — disposing the tensors that stage still holds before throwing.
  The token reaches Kokoro on every path: sentence streaming, `SpeechService.SynthesizeAsync` (through `TtsJob`), and a
  new `ISynthesizerLease.Synthesize(text, options, cancel)` overload, which the voice session's GPU thread now calls
  with the turn's token, so a barge-in stops issuing the sentence's work at the next boundary (on CUDA, kernels already
  queued still finish). The overload is a default interface method that checks only before the call, so other
  implementers keep compiling. Output is byte-identical when not cancelled.
- **The prompt prefill now observes the request's cancellation token between transformer layers.**
  `TextGenerationPipeline` hands the token to the first prefill through a new
  `IGenerationModel.Prefill(chunk, state, cancel)` overload (a default interface method that checks only before the
  call); `GenericTransformerModel` implements it by checking between layers in `GenericTransformer.Forward`,
  `ForwardEmbeds` and the layer-split `ForwardEmbedsStaged`, which all take an optional token. The math and the
  kernel shapes are those of an uncancelled call, so output is identical when nothing is cancelled. A stopped prefill
  commits nothing: the cache length is not advanced, and the rows the finished layers wrote past it are never read
  and are overwritten next time. As before, a request with prefix-cache reuse that is cancelled during the prompt
  prefill drops its retained sequence instead of keeping one the cache never received. On the CPU backend the stop
  lands at the next layer. On CUDA the host queues layers far ahead of the GPU, so the check stops issuing work only
  when the cancel lands early, and the layers already queued still run. Measured on an RTX 3060 (Qwen3-4B Q4_K_M,
  695-token prompt, 1.02 s prefill): a cancel 10 % in throws after 7 ms instead of 913 ms, but the card stays busy
  for 788 ms instead of 913 ms; from 30 % in, every layer is already queued and nothing changes. Freeing the card
  early needs a bound on how far the host runs ahead, which is open work.
- **Added `ToolCallFormats.TryDetectFromTemplate`**, which reads a model's own GGUF `tokenizer.chat_template`
  instead of guessing the tool-call format from its name: true only when the template references the
  caller-supplied `tools` variable AND literally instructs one of the four supported envelopes (Hermes JSON,
  Llama-3 `<|python_tag|>`, Mistral `[TOOL_CALLS]`, Gemma `<|tool_call>`). Verified against the real template
  of eight local GGUFs — three detect correctly (Qwen3-4B/Qwen2.5-1.5B → Hermes, gemma-4-E2B-it → Gemma) and
  five correctly detect as unsupported for family-specific reasons, see
  `tests/HartsyInference.Tools.Tests/Fixtures/ChatTemplates/README.md`.
- **Fixed:** `ToolCallFormats.Detect`'s name-hint heuristic no longer maps `"glm"`/`"deepseek"` to Hermes —
  neither family's real wire format matches it (GLM uses XML arguments, DeepSeek never renders `tools` for
  new calls). Both still reach the documented unknown-family Hermes fallback, so `Detect`'s observable
  behavior for a bare name hint is unchanged.
- **`WakeService.Claim`/`Release`**: an opt-in, per-device host handoff for the wake listener. A host can claim
  one connected satellite's turns (typically from a `Detected` handler) and receive its decoded inbound audio
  (16 kHz mono float, post-denoise when noise suppression is on, normalized from the wake path's internal
  int16 scale) through `WakeDeviceClaim.OnFrame` instead of the service's own wake scoring, end-of-speech
  capture and transcription, which are suspended for that device only. The connection, ping/pong keepalive and
  outbound audio path (`BeginAudio`/`SendAudioAsync`) are unaffected. `Release` returns the device to normal
  listening; a disconnect while claimed auto-releases and calls `WakeDeviceClaim.OnDisconnected` once. Zero
  change for a device nothing has claimed — `WakeSession.Claim` defaults to null and the existing
  scoring/VAD branch is reached exactly as before; proven against the full existing wake suite
  (`WakeTransportTests` and the rest) with real backbone/head/denoiser weights, not just by inspection. This is
  the engine-side requirement for `SwarmUI-AudioLab`'s satellite voice-agent Session mode, which could not
  otherwise get continuous raw audio for a device past its own wake detection.
- **Fixed two races in `WakeService.Claim`/`Release`'s disconnect path, found by review before this shipped.**
  A device's reconnect (`WakeSession.OnReconnected`, from its new connection) could land while its old
  connection was still unwinding; the old connection's `finally` then unconditionally cleared the new
  connection's `Codec`, reset `State` to `Handshake` (silently pausing the worker for that device, since
  `WakeWorker.Run` skips a session in `Handshake`), cleared the new connection's claim, and fired a spurious
  `OnDisconnected` for a device that was, in fact, still connected. `WakeSession.Codec` is now a field (like
  `Claim` already was) so the disconnect path can clear it with a CAS keyed to the specific codec that
  connection installed; a superseded connection's teardown now does nothing instead. A throwing
  `OnDisconnected` is also now caught and logged (`Logs.Error`) rather than propagating out of the `finally`,
  where it could otherwise mask whatever exception actually ended the connection. Same treatment for a
  throwing `OnFrame` in `WakeWorker`, caught separately from the pipeline/denoiser/VAD reset path so a
  persistently-throwing host callback doesn't flood the log with pointless resets of state a claimed device
  never reads. `WakeService.Claim` also now withdraws (and returns null for) a claim whose connection died in
  the gap between its own liveness check and installing the claim, so that race can no longer leave a host
  holding a claim that will never call `OnDisconnected`. A reconnect ends the device's previous claim too —
  cleared AND notified before `OnReconnected` publishes the new codec, not after, so neither step can land on
  a claim a host thread installed against that new codec, and so that notification is one of the writes the
  new codec's own publish carries to anyone who observes it (see the next entry): letting it silently carry
  over to the new connection's audio with no signal the old one is gone was the bug; running the notification
  after the publish, rather than before, turned out to be a second, narrower version of the same bug.
- **`WakeSession.Codec` is `volatile`.** `WakeListener`'s disconnect CAS and reconnect handling, and
  `WakeService`'s outbound audio path, all read or write it across threads without a lock. A release (the
  volatile write in `OnReconnected`, or the CAS in the disconnect path) only carries a thread's earlier writes
  forward to whoever observes it — never later ones — so anything a reconnect needs a reader to see has to
  happen before the codec publish, not after. A new regression test polling this field with a plain read, then
  immediately checking a side effect of the reconnect, caught both the missing `volatile` and, after adding
  it, that the notification above was still ordered on the wrong side of the publish: flaky under full
  test-suite parallel load either way, just far less often with only the field fixed. 16 consecutive clean
  full-suite runs once both were corrected, after failures inside the first 3-8 runs at each earlier stage.
- **`WakeWorker.Run` clears its shared `detections` list before deciding whether there is anything to score
  this iteration**, not only inside the unclaimed branch's own `Pipeline.Push`. Before, a denoiser that held
  an iteration's audio back entirely (so `toProcess` was empty) skipped scoring but left whatever detection
  the list held from an earlier iteration in place, and the dispatch loop below re-fired it a second time.
- **Orpheus TTS no longer re-prompts to download its SNAC codec on every generation.** `ModelCatalog`'s
  SNAC asset (`hubertsiuzdak/snac_24khz`) listed `RepoPath = "model.safetensors"`; the real downloaded
  file is `pytorch_model.bin`, so `ModelAcquisition`'s presence check always reported it missing. The CLI
  REPL calls `EnsurePresent` once per generation (not once per load), so every Orpheus prompt hit the
  false-missing path and its interactive `Download these now?` prompt. Fixed the `RepoPath`, and added a
  per-process confirmed-present cache so the audio-asset check only runs until it first succeeds for a
  given catalog id.
- **`hartsy transcribe --timestamps` now prints the actual timestamps.** The flag only ever changed a
  "segments N" count in the footer; the transcript text itself was identical with or without it, since
  `TranscribeAsync` never read anything from `TranscriptResult.Words` but its `Count`. It now renders one
  `[start --> end]  text` line per word/segment (word- or segment-granularity, whichever the model
  produced) when timestamps were requested and the pipeline returned any; the plain-text path is
  unchanged byte-for-byte otherwise.
- **The phone gateway and voice host moved out of this repo**, to a separate app:
  [HartsyAI/HartsyPhone](https://github.com/HartsyAI/HartsyPhone). `HartsyInference.PhoneGateway`,
  `HartsyInference.VoiceHost` and `HartsyInference.PhoneLink` (source, tests, the systemd units and install
  scripts under `deploy/`, and their research docs) are removed from this repo; the code and its git history
  continue in the new one, copied as of this engine's `main` at `40c84b69` (alpha.238). `HartsyInference.Voice`
  and `HartsyInference.Tools` are unaffected and stay here (AudioLab and other consumers still use `Voice`
  directly, with no phone dependency). **`HartsyInference.PhoneLink` stops publishing to NuGet as of this
  change**; `publish-nuget.yml`'s EXPECTED package list no longer includes it. SIPSorcery is no longer a
  dependency of this repo. No version bump for this change alone — see whichever numbered section above
  is first to ship after it for the actual release this landed in.
- **Dia TTS: a doomed-to-fail short prompt now fails in seconds instead of tens of seconds.**
  `DiaTtsModel.Session` already auto-tags untagged text with `[S1]`, but a one-sentence prompt (tagged or
  not) still ran the full 1720-frame default budget producing non-speech throughout (confirmed: Whisper
  transcribed the result as `[Music]`; energy stayed high for the full 20s rather than trailing into
  quiet). `maxTokens` is now capped to the text's own length (20 frames/char, floored at 200), which only
  applies when the caller left `TtsJob.MaxTokens` unset -- an explicit value is used as-is, uncapped,
  since a deliberate request isn't the runaway case this exists for. The factor comes from measuring
  natural (uncapped) EOS behavior on 8 real prompts spanning single-speaker sentences, `[S1]`/`[S2]`
  dialogues, a `(laughs)`/pauses case and a non-ASCII case: 4 fired a genuine EOS, the other 4 hit the
  1720 ceiling and were confirmed degenerate by what Whisper actually heard ("[Music]", "[ Silence ]", a
  one-word fragment) versus the non-ASCII prompt's real EOS at 1142 frames (13.126 frames/char, the
  max observed -- confirmed as genuine full-duration speech via a language-correct Whisper pass that
  transcribed it back verbatim). 1.5x that margin sets the factor at 20; it only binds (produces less
  than the 1720 default) for text under ~86 chars. **Confirmed model behaviour, not a port bug**: ran
  upstream nari-labs Dia-1.6B-0626 itself (local weights, no download) on the identical sentence,
  `[S1]`-tagged, same seed and the same 1720 cap. Upstream also never fires EOS -- its own
  `finished_step_Bx` accounting shows the sequence forced to the cap at step ~1704 -- and Whisper
  transcribes its output as `[Music]` too. Same inputs, same failure, in the reference implementation;
  nothing in the C# port's conditioning, CFG, delay pattern, or EOS rule is implicated.
- **cuDNN backend-graph engines marked `CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC` are now skipped by
  default (`numerics.cudnnDeterministic`, default ON).** Root cause of issue #20 (Chatterbox/CosyVoice2/
  Piper producing different HiFT-vocoder output on identical repeated calls): cuDNN's heuristic search
  picks engine 25 on this box for the last `ConvTranspose1d` upsample stage, an atomic-accumulation
  reduction whose float summation order varies run to run. `CudnnPlanSearch.BuildExecutionPlan` now reads
  each candidate's numerical note and skips a nondeterministic one for the next, falling through to the
  existing direct-kernel fallback if every candidate is nondeterministic. Measured zero cost and full
  reproducibility on Chatterbox/CosyVoice2/Piper/Kokoro (3060) and on Krea2-Turbo/Z-Image-Turbo (4090,
  cross-arm byte-identical output on both) — see the PR for both tables. The nondeterminism log line and
  its dedup key now carry a caller-supplied shape signature (op/Cin/Cout/kernel/stride/dtype for conv,
  op/batch/heads/seqlen/headdim/dtype for SDPA) instead of just the engine index.
- **Fixed the `krea2` and `zimage` catalog entries' local target paths.** Both always reported "needs 1
  file(s) not on disk" even though the real checkpoint was present, because `TargetSubdir`/`TargetName`
  didn't match where the file actually landed: `krea2` was missing a `/Turbo` segment, and `zimage`
  pointed at the HF repo's own filename under `Stable-Diffusion/ZImage/` instead of the locally-renamed
  `Stable-Diffusion/z-image-turbo.safetensors`. `hartsy image -m krea2`/`-m zimage` only ever worked via
  an explicit `--model-path` that bypassed the catalog check. `Repo`/`RepoPath`/`Sha256` were already
  correct on both (confirmed against the real HF repos); only the local target path was wrong.

## alpha.241

- **Fixed an intermittent `CUDA_ERROR_INVALID_VALUE` crash on the first `RmsNorm` call of a prefill.**
  `GpuTransferHelper.UploadTo` and `State.FreeDevice`'s async-free branch read `State.StreamHandle` directly, with
  nothing checking that the backend hadn't been retired (zeroing the handle) since the caller resolved that
  `State` — passing stream `0` into `cuMemcpyHtoDAsync`/`cuMemFreeAsync` is legal CUDA usage but wrong for a
  destination the stream-ordered pool allocated on the real compute stream. `UploadTo` now resolves the handle once
  through `State.RequireLiveStream` and throws a clear `ObjectDisposedException` instead of passing zero through;
  `FreeDevice`, which runs from cleanup paths, logs and skips the stream-ordered free instead of throwing.
- **Fixed ~144 redundant PCIe re-uploads per LLM forward call (prefill and decode alike), cutting 3060 cache
  misses from 146/step to the 3 that are actually unavoidable.** `GenericTransformer` builds the RoPE cos/sin
  table (and, for layer 0, reads the embedding lookup) once per forward call and reuses those host tensors
  across every layer, but the residency cache only persists a tensor that became some op's OUTPUT — a plain
  input that misses the cache is uploaded, read, and freed in that call's own cleanup, every single read. Each
  of the three tensors (cos, sin, embedding) now uploads once via a cheap `Scale(_, _, 1f)` identity op before
  the layer loop, so all 36-layer rereads hit instead of re-uploading. Under accurate (non-overlapped) timing
  this is a real, modest TTFT reduction (~4-5%); prefill GEMM time dominates the 3060's TTFT and is unaffected
  by this fix (a separate efficiency project is being scoped for that).
- **Fixed: HeartMuLa's quantized GGUF cache ignored `modelsRoot`/`ModelCacheRoot` entirely.** Every other
  audio model resolves its cache location under `AudioModelCache.CacheRoot` (`modelsRoot/audio` when
  `EngineKnobs.ModelsRoot` is configured, honoring the `EngineKnobs.ModelCacheRoot` override too).
  `HeartMulaMusicModel`'s Q8/Q4 on-disk GGUF cache was the one exception, always writing under
  `~/.cache/hartsyinference/heartmula/` regardless of either knob — so moving model storage (e.g. onto a
  RAID array, via `ModelsRoot`) silently left HeartMuLa's quantized weights on the OS disk instead.

  `HeartMulaMusicModel.ResolveQuantCachePath` now resolves under the shared audio root
  (`modelsRoot/audio/music/heartmula/`), falling back to the legacy `~/.cache/hartsyinference/heartmula/`
  location only when a file already exists there, so an existing install is not silently orphaned.
- **`IBackend.ApplyRope`'s combined q+k overload no longer corrupts K under GQA.** CUDA, Vulkan and the CPU
  default sized K's rotation from Q's head count: harmless while every caller was MHA, an out-of-bounds device
  read/write for a GQA caller (found through Dia's resident decode, #224: 16 query heads, 4 KV heads). All three now
  check that q and k share batch, seqLen and headDim (head counts may differ) and that cos/sin match headDim, throw
  otherwise, and run the per-tensor `ApplyRopeSingle` once per tensor. Same split-half formula: the Ernie A/B on CUDA
  is digest-identical; on Vulkan, dtypes other than F32/F16 now take `ApplyRopeSingle`'s reference fallback. Tests:
  three in `DitGlueKernelTests` and `ApplyRope_QK_Matches_The_Cpu_For_Mha_And_Gqa` on every GPU backend.
- **Opt-in prefix-KV reuse for `TextGenerationPipeline`.** A new `Generate` overload takes a `RetainedSequence`
  (`HartsyInference.LLM.Generation`): it reuses the longest common token-id prefix between the retained cache and
  the current prompt, `Truncate`s the divergent tail and prefills only the diverging suffix (always leaving the
  final prompt token to be prefilled fresh, so sampling always has a real logits row), instead of prefilling the
  whole prompt from an empty cache every call. `RetainedSequence`/`RetainedSequenceStore` are generic (count- and
  byte-bounded, LRU, checkout/checkin so a second concurrent request on a busy key runs uncached) and live
  entirely in the engine, independent of any caller. `TextRequest.PrefixCacheKey` (null by default) opts a caller
  in; `TextRequest.PrefixCacheCapacityHint` sizes a fresh retained sequence once instead of letting it be
  reallocated turn over turn. `TextService` keeps one store per device slot (`vram.prefixCacheMaxEntries`/
  `vram.prefixCacheMaxBytes`), disposed before `FreeAllDeviceMemory` on every unload/reload path. Capacity
  growth past a retained sequence's own cache reallocates transparently; the tensor-parallel path is
  unaffected (no `ISequenceState` to retain). `GenerationResult.ReusedPromptTokens` reports how much of a
  call's prompt came from the cache. The tensor-parallel path and `DynamicBatchScheduler`/`PagedKvPool` are
  untouched — this is the single-sequence `TextGenerationPipeline` path only.
- **Voice turns reuse their call's own prefix.** `VoiceAgentSession` gives each call a key (carried through
  `BuildRequest` and so through every `ToolLoop` round too, since each round's request is `request with {...}`) and
  `EnablePrefixCache` (default true) controls it. `StartAsync` fires a one-token priming request for the
  system+tools prefix so turn 1 is warm as well, racing the greeting instead of delaying it.
- **Voice LLM VRAM.** `TextRequest.CacheWeightCasts` (null = backend default) overrides the device backend's
  `CacheWeightCasts` when its slot's backend is first created; `VoiceAgentOptions.CacheWeightCasts` defaults to
  `false`. Measured on Qwen3-4B-Q4_K_M/4090: the backend's own default (cache a dequantized F16/BF16 copy of
  every quantized weight) costs ~7.3 GB resident once warm on top of the ~5.5 GB the weights themselves take —
  the dominant share of the model's footprint; off, weights stay compressed with a transient per-GEMM dequant,
  for a fixed ~50 ms tax per prefill call (prompt-length independent; decode's quantized GEMV path and tokens/sec
  are unaffected either way). `ITextService.TrimMemoryPool` (default no-op, so an existing implementation keeps
  compiling) is `TextService`'s per-slot, best-effort, non-blocking equivalent of `Unload` that returns pool slack
  to the driver without unloading weights; the voice turn loop awaits it at the same idle point
  `VoiceGpuWorker.RequestTrim` already uses for the audio backend, and `VoiceModelSet.WarmAsync` and the session's
  priming request each trim their own transient usage once done with it.
- **Voice LLM VRAM: no redundant weight-split preload.** `TextService.LoadInto`'s single-device weight
  upload called `GenericTransformer.EnumerateWeights()` with its default (`includeRedundantSplits: true`),
  unlike `LoadSharded` and `GenericTransformerModel.PreloadDecodeWeights`, which already pass `false`. On
  Qwen3-4B-Q4_K_M the load-time-fused Q/K/V and gate/up projections leave their pre-fusion split originals
  resident too (tensor-level measurement: 1.21 GiB of the 3.53 GiB a default single-device load uploads, vs.
  2.32 GiB deduplicated — matching the 2.33 GiB file almost exactly); nothing on `TextGenerationPipeline`'s
  single-sequence decode/prefill path reads them (only the batch scheduler's mixed-dtype split-projection
  path does, via its own lazy auto-promotion, unaffected by this change). Checked and ruled out as explanations
  for the same gap: `AutoPromoteWeights` (a residency-pinning decision only — it uploads a weight already
  twice-missed in its native dtype, never a dequantized copy) and the tied embedding/`lm_head` (kept quantized
  at ~304 MiB via the existing `_lmHeadQuant` fused-GEMV path, not dequantized to F16/F32 as its size alone
  would suggest). `TextRequest.PreloadRedundantWeightSplits` (null = unchanged default) opts a request out;
  `VoiceAgentOptions.PreloadRedundantWeightSplits` defaults to `false` and is wired through warm-up, the
  call's priming request and every turn, identically to `CacheWeightCasts`.
- **Voice gate, measured on real weights (Qwen3-4B-Q4_K_M/4090, Whisper small.en + Kokoro/3060, the host's real
  7-tool set, 10-turn growing history):** `voice.llm.ttft_ms` 85.7-105.4 ms and flat regardless of history length
  (gate ≤ 150 ms; was 200-302 ms and growing), `voice.llm.first_sentence_ms` 139.1-158.1 ms (gate ≤ 200 ms),
  `voice.turn.total_ms` 1095.8-1225.1 ms (gate ≤ 1300 ms), decode 89-102 tok/s. VRAM primed and flat for all ten
  turns at **5.81 GB, under the ≤ 6 GB target** (was ~13.8 GB). A smaller `PrefixCacheCapacityHint` and
  `vram.kvF16` stay unused.
- **Dia TTS decode self-attention is GPU-resident; warm generation is ~3.6x faster with bit-identical
  output.** `DiaAttention.SelfForwardFlash` (gated on `IBackend.FlashDecodeSupported`) replaces the
  per-step host reshape/RoPE/GQA-repeat/memcpy attention path Dia shared with pre-fix Zonos, mirroring
  Zonos's own resident decode (`ForwardResident`, `FixedKvCache`). Split-half (NeoX) RoPE is applied via
  two per-tensor `IBackend.ApplyRopeSingle` calls (q and k separately -- Dia's self-attention is GQA, 16
  query heads / 4 KV heads, so the combined q+k call corrupted K before #228 fixed it).
  Verified same seed, same prompt, baseline vs fixed: identical sha256 and duration across 6 reps each;
  warm median 91.92s -> 25.39s, RTF ~10.9 -> ~3.0. CPU/Vulkan paths are untouched.
- **Audio regression triage, alpha.183 -> alpha.238: no regression found.** Orpheus, CSM, Qwen3-TTS,
  Chatterbox and Moonshine (STT) A/B'd on the RTX 4090, same text/voice/seed: all flat to noise, Qwen3-TTS
  ~24% faster (not chased further). The 176 shared-backend files (Cuda/Core/Gpu) that changed in that
  window did not slow any of the five models down.

## alpha.240

- **Fixed: an unparameterized Piper request 404'd fetching `piper.onnx`.** `AudioModelSelector.Parse` falls
  back to the bare catalog token (e.g. `"piper"`) for `Variant` whenever the request token has no `':'` —
  correct for a descriptor that treats a bare id as its own repo/model identifier, but Piper's weights ARE
  the voice (one `.onnx` per voice), so that bare token is not a voice at all. `SpeechService.ResolveTarget`
  used to pass it straight through as the load variant for any `VoiceSelectsWeights` descriptor whenever no
  separate named voice was given, with no way to tell "the catalog id leaked through" from "the caller
  genuinely asked for a voice named `piper`" — so a request with no voice and no `:variant` 404'd fetching
  `rhasspy/piper-voices/piper.onnx` (no such file exists; every real Piper voice lives at
  `<lang>/<lang_REGION>/<name>/<quality>/<id>.onnx`) instead of falling back to Piper's own default voice.

  Fixed in `SpeechService.ResolveVariant` (extracted from `ResolveTarget`): detects the bare-token-fallback
  shape by comparing `AudioModelSelector.Variant` against `AudioModelSelector.Id`, not by hardcoding
  Piper's name, so the fix covers any other `VoiceSelectsWeights` model with this same shape, not just
  Piper. `AudioModelSelector.Parse` itself and `PiperModel.LoadAsync`'s own `"default"`/empty sentinel check
  are both unchanged — the first is shared, load-bearing logic for every modality's selector, the second
  already did the right thing once actually given one of those values.

## alpha.239

- **Masked inpaint pastes its result back through one engine-level, hard-threshold step, as SwarmUI does.** The
  pipelines each blended the decoded image over the source with the same soft mask they used inside the denoise.
  `MaskRecomposite` now does the paste after generation: any mask value above 0.001 takes the new pixel, so Mask
  Blur only softens the in-denoise blend (and, because the grown and blurred mask is thresholded, widens the pasted
  region by about the blur radius), unless `ImageRequest.MaskCompositeUnthresholded` asks for the soft paste.
  `Inpaint.RecompositeMask` turns the full-canvas paste off (Init Image Recomposite Mask); the crop and segment paths
  always paste. `RecipeImg2ImgBinder` switches the pipelines' own paste off whenever a mask is present, so a caller
  driving a recipe pipeline directly with a mask gets no paste and should go through `IImagesService`.
  `MaskCompositeUnthresholded` and `RecompositeMask` are request fields for library callers (the SwarmUI extension);
  the CLI and HTTP API do not expose them yet.
- A declined "inpaint only masked" crop (empty mask, or a crop covering the whole canvas) now clears the crop request
  before the full-canvas run; before, the mask resolver's guard threw.
- **Audio: fixed the vocab-sized delegate-sort allocation anti-pattern in the TTS samplers** — the same
  pattern PR #215 fixed in the LLM package's `TopPStep`, independently present in several places in
  `src/HartsyInference.Audio`:
  - `NucleusSampler.Draw`'s unbounded fallback (hit whenever `topK<=0`, or `topK` exceeds the bounded fast
    path's 1024 cap): Chatterbox, Zonos, Dia (its own `Draw` call), Bark's coarse/fine/generic stages, and
    FishSpeech's codebook pass all pass `topK=0` today. A fresh `float[count]` + `int[count]` plus a full
    `Array.Sort` with a `Comparison<int>` delegate, every token.
  - `LogitSampling.SampleTopK` (Kyutai's text-token sampling and its depformer, up to 32 calls per audio
    frame): the same shape, run unconditionally regardless of how small `k` was against the vocabulary.
  - Two bespoke duplicate sorters outside `NucleusSampler` entirely: `DiaPipeline.SampleDiaChannel`'s
    CFG-window selection and `BarkCausalStage.SampleTopPWithProb`'s semantic-stage top-p cut (up to 768
    calls per generation).
  - `Yue2LogitProcessor.ApplyNucleus`: the allocation half of the same anti-pattern (fresh `List<int>`/
    `List<float>` plus collection-expression array copies every call) without the delegate-sort half — its
    sort was already the primitive two-array `Array.Sort` overload.

  Fix, mirrored from #215 rather than imported across the package boundary (Audio's `Draw` walks a SORTED
  sequence for its multinomial draw — tie order and float-summation order are part of the observable output
  — unlike LLM's `SamplerChain`, which always draws in natural index order): new
  `src/HartsyInference.Audio/Sampling/SortHelpers.cs` holds a primitive `Array.Sort(float[], int[])`
  (negate / sort ascending / negate back) in place of the delegate comparer, and every fixed call site keeps
  a `[ThreadStatic]` scratch buffer pair, sized once and grown (never shrunk) on demand — no allocation after
  the first call on a given thread. Every full-sort site keeps its EXACT pre-existing algorithm; only the
  allocation and the sort mechanism changed. `NucleusSampler`'s existing bounded top-k fast path (every
  other caller: CSM, SparkTTS, Qwen3-TTS, CosyVoice, GptSoVits, NeuTTS, MusicGen, MiniMax, the original YuE)
  was already allocation-free and delegate-free and is untouched.

  **Verified byte-identical**, not assumed: `SortHelpersTests` proves the new primitive sort reproduces the
  OLD delegate sort's exact permutation — including exact ties at every rank — across random, all-equal,
  low-cardinality, sorted, organ-pipe and median-of-three-killer inputs from 1 to 102,048 elements (the one
  documented divergence is NaN placement, pinned by its own test rather than left unchecked: real model
  logits are never NaN without sampling already being broken upstream). The failure mode shifts, though: the
  old delegate sort put a NaN logit at the back (low-ranked, rarely drawn); the new primitive sort's
  floating-point pre-pass puts it at the front, so `probs[0]`/`vals[0]`/`wSort[0]` — read as the running max —
  can itself be NaN, poisoning the whole softmax rather than one token's probability. Acceptable (both are
  "undefined behavior on already-corrupt input," never relied on either way), but worth knowing if a NaN
  logit ever appears in production. Five more test files keep each fixed
  call site's pre-fix algorithm verbatim as a reference oracle and compare it against the real (fixed) code
  across random inputs, exact ties at several boundary shapes, tiny/huge top-p, temperature=0, minP-only,
  and masked tokens — 87 tests total, all passing. `BarkCausalStage.SampleTopPWithProb` and
  `Yue2LogitProcessor.ApplyNucleus` move from `private` to `internal` so the test project can reach them
  directly (the same `InternalsVisibleTo` pattern already used elsewhere in this package).

  **Measured on the 3060** (`hartsy speak`, same text, seed 0 — the TTS default — base vs. this branch,
  interleaved): sampler-alone cost (steady-state ms/call, CPU) and end-to-end wall clock, output digest
  identical unless noted:

  | Model | sampler ms/call before→after | end-to-end wall before→after | output |
  |---|---:|---:|---|
  | Kyutai TTS | 0.226→0.089 (depformer) / 0.93→0.42 (text) | 25.39s → 9.41s (−62.9%) | digest identical |
  | FishSpeech | 12.02→7.39 (slow/text pass) / 0.056→0.025 (fast/codebook pass) | 11.20s → 7.10s (−36.6%) | digest identical |
  | Bark | 0.058→0.027 (coarse/fine); bespoke semantic sort fixed, same technique | 13.63s → 12.80s (−6.1%) | digest identical |
  | Zonos | 0.056→0.024 (×9 DAC channels/frame) | 29.10s → 26.15s (−10.1%) | digest identical |
  | Chatterbox | 0.78→0.52 (×1/token) | 5.62s → 5.64s (noise) | **not comparable** — Chatterbox's S3Gen/HiFTNet stage is not bit-reproducible even on the SAME unmodified build run twice (confirmed); the CPU reference-oracle tests are the byte-identical proof for this model |
  | Orpheus | 2.79→1.80 (sampleCount=28,683, its real post-`CodeStart` slice) | ~0.4% of total (estimate: 639 tokens × 0.99ms saved ÷ 156.5s at the documented ~245ms/token) | not independently re-measured end to end — the existing `OrpheusPipeline` perf-pass comment already measured "sampler restriction... flat": its decode is GPU-compute-bound on the F32 backbone, not the sampler |
  | Dia | 0.057→0.024 (×9 DAC channels/frame; CPU-isolated) | not cleanly measured end-to-end | see below |
  | CSM, SparkTTS, Qwen3-TTS, CosyVoice, GptSoVits, NeuTTS, MusicGen, MiniMax, YuE (v1) | unaffected — already on `NucleusSampler`'s bounded fast path (`topK` small and `<1024`) | unaffected | regression-guarded by `BoundedFastPath_Unaffected_MatchesReference` |

  **Dia**: a clean before/after end-to-end run was not obtained in this PR. The CLI confirmed it loaded the
  correct `nari-labs/Dia-1.6B-0626` checkpoint (not the broken base release this repo already root-caused a
  different symptom to), so that is not the explanation. A short (one-sentence) prompt reproduces Dia's own
  documented "prompt is very short... tends to produce silence" pathology and never reaches EOS; a proper
  multi-turn `[S1]/[S2]` dialogue prompt still did not complete in 180s on either build. `nvidia-smi dmon`
  sampled ~8-28s into one such run showed 0% SM utilization for most of that window with a utilization spike
  right at the end — most likely still the 1.6B F32 checkpoint's load/deserialize/H2D-upload phase finishing,
  not a stall mid-generation (an earlier, separate probe took 38s before an unrelated OOM, consistent with a
  long load); this sample did not reach far enough into actual decoding to say where DECODE time goes. A
  concurrent agent's GPU test suite was also independently confirmed running on the same card during testing,
  so even the timing that was captured isn't fully trustworthy. What IS established: the sampler's own
  contribution is negligible (CPU-isolated, ~0.03ms/channel saved × 9 channels/frame) and output-identical
  (`DiaSampleChannelIdentityTests`, including exact ties at the CFG-window boundary) — Dia's real cost,
  whatever it is, is elsewhere, and needs a dedicated, contention-free re-run (with load and decode timed
  separately) to profile. Not fixed in this PR. zipvoice (named in the original slow-model list) has no
  token-sampling path at
  all — a flow-matching model, out of scope for this fix entirely.

## alpha.238

- **Voice host exe (`src/HartsyInference.VoiceHost`, not packaged).** The phone-call voice agent's model process: a
  generic host (from the ASP.NET Core shared framework; built empty, so nothing reads the environment or an appsettings
  file) that builds the engine on the audio card (`cuda:1`) with `ToolCalling.Install`, loads and warms the voice model
  set, and serves the PhoneLink socket the gateway dials. One `VoiceAgentSession` per call; the models outlive calls;
  every language-model request names `LlmDevice` (`cuda:0`).
- **Boot warm-up offers the host's real tool set.** `VoiceHostTools.WarmDefinitions` builds the same
  `ToolDefinition`s a real call's registry would (the six telephony tools plus `get_time`, for whichever of them
  `tools.enabled` names), with request/hang-up delegates that throw if ever invoked — warm-up only offers tools to
  the chat template and the tool-call grammar/stream filter (`VoiceModelSet.WarmAsync` streams and discards), it
  never dispatches one. `VoiceHostService.StartAsync` passes them to `WarmAsync`, so production warm-up takes the
  tool-aware `WarmToolMaxTokens`-token path instead of staying on the cold one-token path a tool-less warm-up takes.
- Typed options from `/etc/hartsyinference/voice.json` (`link`, `models`, `agent`, `tools`, `engine`, `logging`; `{}` is
  valid, an unknown key fails the start, every error names its JSON path). The link token comes from a secret file
  (`LoadCredential=`), refused when group or others can read it and never logged; `engine.cpuThreadCap` is
  `numerics.cpuThreads` for the host's life.
- Link: `Hello` checked first (version, 16 kHz, token in constant time), else `Error` and close; at most four
  connections in their handshake at once; per-frame sequence check; a new connection replaces the current one only
  after its own `Hello`. A dedicated sender thread on absolute
  20 ms deadlines writes control frames first, then each call's reply audio in 20 ms frames tagged with the producing
  turn, with a 40 ms prebuffer per burst and catch-up frames; its audio path, the session's read included, allocates
  nothing once warm. A barge-in
  becomes `Flush(turnId)` and nothing of that turn follows it; `OutboundEnd` closes each turn that played. Session
  events go out as `Event` frames (state, final transcript, turn latency).
- Telephony tools (`send_dtmf`, `transfer`, `hold`, `unhold`, `play_prompt`) are `ToolRequest`/`ToolResult` round trips
  with a timeout; `get_time` is answered on the host; `hangup` goes to the gateway only once the goodbye of the turn
  that called it has drained to the link (its `OutboundEnd` written, or the turn flushed) plus 200 ms for the audio
  still downstream, capped at the audio the session still held plus 1 s and at 10 s, so the caller hears the goodbye
  and a stuck drain cannot keep the call open; a barge-in on the goodbye cuts it short but still ends the call. A
  resumed call gets a fresh session and an apology line; any session failure ends only
  its call with `CallEnd(Failed)`; SIGTERM ends calls with `CallEnd(LocalHangup)`. Workstation concurrent GC,
  `SustainedLowLatency` while calls are up, pool floor `numerics.cpuThreads + 8`.
- **`HartsyInference.Voice`: `ReadOutbound(Span<float>, out int turnId)`** returns one turn's audio at a time and names
  the turn that wrote it (each turn publishes a mark before its first sample), so a remote player can drop a flushed
  turn by id. The untagged read is unchanged. `OutboundQueuedSamples` reports the reply audio still queued.
- **`SecretFile` moved to Core** (`HartsyInference.Core.Configuration`), with the caller's exception factory; the gateway
  reads its secrets through it unchanged.
- **Deployment:** `deploy/systemd/hartsyinference-voice-host.service` (`Restart=always`, `RuntimeDirectory`,
  `AllowedCPUs=0-6,8-14`, token credential) and `hartsyinference-phone-gateway.service` (`Requires=` the host,
  `LimitRTPRIO=50`, `Nice=-10`, `AllowedCPUs=7,15`, a 64 MB gen0 budget, three credentials, no service-wide FIFO);
  `AllowedCPUs=0-6,8-14` on the API unit, whose start-limit settings now sit in `[Unit]`, where systemd reads them.
  Runbook: `docs/Checklists/VOICE_AGENT_VERIFICATION.md`.
- **Host tuning:** `deploy/install-host-tuning.sh` installs two things. One is `hartsyinference-cpu-performance.service`,
  a oneshot unit that sets every CPU's frequency governor to `performance` at boot through sysfs. The other is
  `/etc/security/limits.d/hartsy-rt.conf` (`hartsy - rtprio 50`, for runs without systemd). The script is a dry run by
  default, with `--apply`, `--revert` and `--dry-run --revert`. `--revert` puts every CPU on `performance` back on
  `schedutil`, leaves other governors alone, and fails if a CPU cannot go back. The script is idempotent, needs root
  only to change anything, refuses to write through a symlink, and prints its sources' SHA-256 so an apply can be
  checked against the dry run. These are optimizations: under `schedutil`, the idle
  gaps between turns cost Kokoro about 33 ms per sentence and the paced front end about 0.7 ms per frame, but every
  gate passes either way.
- Tests (`tests/HartsyInference.VoiceHost.Tests`): 87 unit tests against a fake gateway on a temporary socket with
  scripted sessions (handshake refusals, call lifecycle and faults, PCM16 scale and sequence checks, flush and stale-turn
  rules, tool round trips and timeouts, hangup after the goodbye's last frame, its cap and a barge-in on the goodbye,
  config and token file, 1200-frame sender cadence with zero allocation, zero allocation while the sender's reads wake a
  producer waiting for playback, as the real session's do, and the boot warm-up's tool definitions against a fake
  `VoiceModelSet`), and `[Slow]` loopback calls (sipsorcery
  softphone → real gateway → real host with Kokoro and Whisper small.en on the RTX 3060), including silence on the line
  between the goodbye and the BYE, and a host killed with SIGKILL mid-call (`tests/HartsyInference.VoiceHost.TestHost`).
  Six new Voice unit tests cover the tagged read.
- **Rebased onto `main` (#202, alpha.237) and re-run on the RTX 3060 with `Denoise` at its new default (on, int8).**
  `LoopbackAssets.All()` named Silero, Whisper, Kokoro and the JFK clip but not RNNoise; added its weights and int8
  tables, since `VoiceModelSet.LoadFrontEnd` now needs them and never substitutes raw audio for a missing denoiser (the
  weights were present on this box, so this did not change the result below, only the gate's correctness on a host that
  lacks them). Quiet window on the 3060 only (the scripted-LLM loopback calls need no 4090), `--verify-since` clean
  after. `LoopbackSipCallWithHostTests` 3/3, `LoopbackHostKillTests` 1/1, same as the `Denoise`-off run:
  - Barge-in: `Flush` reached the gateway 8.0 ms after the VAD decision; the cancelled reply's last audible frame left
    the gateway 6.3 ms after the decision (was 6.4 ms off `Denoise`); 0 frames of the flushed turn after `Flush`.
  - Agent hangup: 106 audible goodbye frames, then 24 quiet frames before the BYE (identical frame counts to the
    `Denoise`-off run); the host asked 225 ms after the turn ended.
  - Host killed with SIGKILL mid-call: the phone got the BYE 1186 ms after the 3 s outage period (was 1.2 s off
    `Denoise`); the restarted host replaced the stale socket and took the next call.
  - `voice.endpoint.ms` now reads 776.00 (was 736.00 off `Denoise` — the 40 ms RNNoise lag `VoiceTurnMetrics` picked up
    from #202); every other stage is unaffected by denoising, as expected.
  - **Allocation (open in #202, confirmed fixed here):** the link-close line reads `audioPathAllocated=0B` on both the
    long-lived call host (1333 ticks) and the SIGKILL test's two short-lived hosts, where the earlier `Denoise`-off run
    measured 19 528 B.
  - Sender: lateness p50/p99 within the 200 µs bucket, max 29.1 ms, 1 catch-up frame, 0 resyncs over 1333 ticks (was
    max 25.4 ms, 2 catch-up frames, 0 resyncs over 1339 ticks off `Denoise` — noise at this precision, same shape).

## alpha.237

- **`HartsyInference.Voice`: opt-in per-call voice agent.** New packable library (references Audio, Engine and Tools;
  not in the meta package). `VoiceModelSet.LoadAsync(engine, options)` opens Whisper and Kokoro runner leases on an
  engine built on `AudioDevice` (checked) and loads the per-session front-end weights from the wake models' `vad` and
  `denoise` folders: Silero is required, and RNNoise is required when `Denoise` is on, failing at load rather than
  passing raw audio through. `WarmAsync` runs five syntheses and one recognition of a second of silence on the model
  set's GPU thread, plus a generation on `LlmDevice`: a one-token request without tools, or, when the caller passes
  its tool set, `WarmToolMaxTokens` (8) real tokens with tools offered, through `StreamAsync` rather than
  `GenerateAsync` so the tool-call grammar, its stream filter and parser, and the template's tools branch are all hot
  before the first caller — see the Qwen3 re-measurement below. The five texts cover Kokoro's
  power-of-two frame-length buckets from 32 to 512, so a reply's first sentence finds its bucket's convolution plans
  already built. `VoiceAgentSession(models, ITextService, ToolRegistry, options)` is one call: `StartAsync`,
  `PushInbound` (never blocks; 30 s, oldest dropped and counted), `ReadOutbound` (never blocks; zero-fills),
  `SpeakAsync`, `PushDtmf` (`[DTMF n]` user turns), `Transcript`, `EventRaised` (in order, on a pool thread),
  `EndAsync`. States: Created, Warming, Listening, Thinking, Speaking, ToolRunning, Ended.
- Threads:
  - The audio thread is dedicated, runs in `CpuParallel`'s inline scope with its own `CpuBackend`, sleeps on a
    doorbell and allocates nothing per 20 ms frame (asserted over 1000 frames). It runs RNNoise at int16 scale, then
    Silero endpointing (700 ms silence, 15 s cut), then barge-in (after a 300 ms hold-off, 200 ms at probability
    0.6 or above; it flushes the outbound queue and `CancelAsync`s the turn).
  - The GPU thread (owned by the model set) holds the device gate for one job at a time, and frees each job's
    activations with the pool's reservation kept (`FreeActivations(trimPool: false)`). It queues one pool trim per
    turn when the session returns to listening, skipped if the next turn's work is already queued.
  - The turn loop runs the token-trimmed conversation through `ToolLoop` with thinking off and `Device` on every
    request, then streams sentences to the GPU thread, resamples and queues them.
- The outbound flush is consumer-applied (the ring allows discards only on the reader): the audio thread bumps an
  epoch, the reader discards, the writer re-bumps when a frame lands after the discard, and a new turn waits until
  the reader has applied every flush.
- `ReadOutbound` allocates nothing, also while a turn waits for playback, space or a flush.
  - The producer's waiter carries its target, and the reader completes it once, on the read that reaches it.
  - The token completes the waiter through `UnsafeRegister`, so the wake queues only the producer's continuation.
  - A full queue wakes its writer once per quarter of the ring.
- An utterance whose speech lies within the reply (plus a hold-off-long echo tail) and never barged in is not
  answered, and is counted. Two user or two plain assistant messages in a row merge, so turns stay alternating.
- A revoked lease (engine free-memory, backend switch) is reopened once outside the gate and the job retried.
- Per-turn `VoiceTurnMetrics` are logged as `[Voice] turn N: …` with every `voice.*` stage.
- Tests (`tests/HartsyInference.Voice.Tests`, 83 unit tests on fakes): endpointing, barge-in, the flush protocol
  (with a concurrent-flush ordering stress), GPU-thread jobs (every job keeps the pool, one trim per turn),
  conversation trimming, model-set load contract, the warm-up (one synthesis per length bucket), whole turns, lease
  revocation, and zero allocation on the audio thread and on the reader (0 B over 1000 reads with a cancellable
  wait pending, one wake at its position); a missing int8 RNNoise table fails the load loudly and names the table,
  even when the F32 weights are present.
- Integration tests (CPU, real weights):
  - JFK endpointing: Silero 0.57 ms per 20 ms frame.
  - Whisper tiny through the session: 100 % JFK content-word recall, 91 % narrowband.
  - RNNoise (int8) + Silero per 20 ms frame, back-to-back over the JFK clip: p50/p99 against the redefined gate
    (3 / 5 ms) — see the RNNoise entry below.
- GPU classes pass on the RTX 3060 (measured with `Denoise` off, its default at the time; a rerun with the new
  int8-on-by-default follows):
  - Session end to end, scripted LLM: Whisper small.en 89-156 ms per utterance (gate 350 ms). Kokoro takes 195 ms
    median for a 15-word sentence (gate 250 ms) and 241 ms for a turn's first sentence (18 words, first synthesis),
    within the 250 ms budget, with alpha.232's length-bucketed conv plans and the bucket warm-up. The first turn
    totals 1195 ms from end of speech to first reply audio (budget 1.3 s). Recall is JFK 11/11, reply 8/9.
  - Barge-in: no reply audio read after the decision, flush applied 4.1 ms later (gate 100 ms).
  - Pinned runners: the leases survive memory-pressure switches with no reopen.
  - A Slow Qwen3 class runs only with an explicit 4090 grant.
- **RNNoise now loads at `RnnoisePrecision.Int8` for the voice front end, and `Denoise` defaults to true.** The
  front-end gate (redefined in the RNNoise int8 PR) was only met at int8: quiet p50 1.07-1.08 ms, p99 1.20-1.59 ms
  against a 3 / 5 ms budget (`WakeModelSet.LoadDenoiser(RnnoisePrecision)` overload; the wake stack's own
  `LoadDenoiser()` call keeps loading Float, so wake scoring is unaffected). A missing int8 table fails the load the
  same way a missing weights file always has — loudly, never a passthrough. RNNoise adds 640 samples (40 ms) of
  algorithmic delay ahead of both the endpoint and barge-in decisions. Design: `docs/Research/VOICE_AGENT_SESSION.md`.
- **GPU re-run with `Denoise` on (RTX 3060, quiet window verified clean):** `VoiceSessionEndToEndTests` passed;
  turn 1 `voice.frontend.ms` p50/p99/max 1.00/2.00/1.11 ms (int8 RNNoise + Silero, live call), STT 133.26 ms,
  Kokoro median 195.4 ms, `voice.endpoint.ms` **unchanged at 736.00 ms** (a sample-counted interval is blind to
  RNNoise's fixed delay — it is paid once before the count starts, so the caller's real wait is nearer 776 ms),
  turn total 1209.39 ms (≤ 1.3 s budget; ~117 ms left for a real model's first sentence). Recall 11/11 caller,
  8/9 reply. `VoiceSessionQwen3EndToEndTests` (real Qwen3-4B on the 4090, audio on the 3060, both cards visible)
  also passed: `qwen3` now resolves (#205's case fix), `voice.llm.ttft_ms` 357.36 ms and `first_sentence_ms`
  949.65 ms on this cold first live turn (tools installed, real prompt — not the plan's isolated, pre-warmed
  152 ms/151 tok/s probe), turn total **1974.72 ms, over the 1.3 s budget**, almost entirely from the LLM stage.
  Recall 100 % both directions.
- **LLM voice-turn latency bring-up — budget met on every stage, every turn.** Two causes, both fixed now:
  1. `WarmAsync` warmed the LLM with a one-token, no-tools request, so the tool-call grammar, `ToolCallStreamFilter`/
     `ToolCallParser`, the Jinja template's tools branch, and every decode step past the first were cold on the real
     first turn. Fixed above (`WarmAsync` accepts the host's tool definitions and runs `WarmToolMaxTokens` tokens
     through `StreamAsync` when given any).
  2. The first re-measurement (357 → 65-78 ms TTFT, but `first_sentence` still 90-120 ms over its 200 ms budget)
     pointed at decode throughput: 38-40 tok/s in the session against the plan's isolated probe's 151 tok/s with the
     same grammar armed. That pointed at session-path transport at the time, but the real cause, found and fixed on
     `main` (alpha.236, `perf/llm-short-reply-decode`, #215): the probe sampled greedy; the voice session's
     `TextRequest` defaults to `Temperature=0.7, TopP=0.95, Greedy=false`, and `TopPStep`'s nucleus filter argsorted
     the full ~152K-token vocabulary through a `Comparison<int>` delegate and allocated three vocab-sized arrays —
     every decode step, only on that non-greedy path. Fixed there (reused scratch buffers, a delegate-free sort,
     sorting only the candidate subset when it suffices); this PR just rebased onto it.

  `VoiceSessionQwen3EndToEndTests` drives 4 turns of the same utterance (conversation history grows each turn) with
  two independent timelines — `RecordingDiagnostics` (raw per-token events via `EngineOptions.Diagnostics`) and a
  `TimestampingTextService` decorator (the chunks the session streams, after the tool-call filter) — plus a
  best-effort read of `CudaBackend.LtGemmPlanStats` after warm-up and every turn. Re-measured after the rebase
  (RTX 3060 + 4090, both cards visible, clean quiet window, verified clean afterwards):

  | Turn | `llm.ttft_ms` (≤ 150) | `llm.first_sentence_ms` (≤ 200) | decode tok/s | `turn.total_ms` (≤ 1.3 s) |
  |---|---:|---:|---:|---:|
  | 1 (cold) | 63.7 | 152.2 | 101.5 | 1226.6 |
  | 2 | 46.4 | 120.3 | 121.7 | 1092.4 |
  | 3 | 62.4 | 148.7 | 102.5 | 1124.0 |
  | 4 | 81.9 | 172.1 | 99.8 | 1133.5 |

  **Every stage met its budget on every turn, cold or warm, this run** (one run; tightest margins are turn 1's
  total, 73 ms, and turn 4's first sentence, 28 ms — needs the engine at alpha.236+ for #215's sampler fix, or the
  real number is back to ~290 ms regardless of this PR). TTFT 357 → 46-82 ms; `first_sentence` 950 → 120-172
  ms (was still 291-319 ms, over budget, before the sampler fix); `turn.total` 1975 → 1092-1227 ms. Decode is 99.8-
  121.7 tok/s (was 37-40), matching the sampler fix's own cited 94-106 tok/s for a realistic `ToolLoop` turn.
  `LtGemmPlanStats` stayed `(0,0,0,0)` throughout both re-measurements — this GGUF-quantized model's decode never
  reaches cuBLASLt's fused-epilogue GEMM path, confirming the plan-cache counter was never going to explain either
  gap. `voice.endpoint.ms` reads **776.00 on every turn = 736 + RNNoise's 40 ms lag**, confirming the endpoint-metric
  fix below in a real session. Recall 100 % both directions on turn 1, no `<think>` text in any turn.
- `voice.endpoint.ms` now adds the denoiser's algorithmic lag (`RnnoiseStream.LatencySamples`, exposed as
  `VoiceAudioFrontend.DenoiserLatencySamples`, 0 when `Denoise` is off) to the sample-counted hangover, so the metric
  (and `turn.total_ms`, which folds it in) reports the caller's real wall-clock wait instead of being silently short
  by 40 ms whenever denoising runs — see the live-session confirmation above. Pinned by a CPU test: a real Silero VAD
  + a real RNNoise instance over the JFK clip, one turn through a full `VoiceAgentSession` (fake speech/LLM), reports
  `EndpointMs` 736.00 with `Denoise` off and 776.00 on — exactly the 40 ms lag, every run.
- **Still open for PR9 (`feat/voice-host-and-deploy`, stacked on this branch):** `VoiceHost`'s boot-time
  `WarmAsync` call must pass its tool set too, or production warm-up stays on the cold one-token path this fix
  addresses only when a caller opts in.

## alpha.236

- **Non-greedy sampling (temperature/top-p) no longer sorts the whole vocabulary from scratch every token.**
  The phone voice agent's short replies were measured streaming at 37-40 tok/s on the 4090 against an isolated
  probe's 151 tok/s for what looked like the same request. Neither named suspect (per-generation CUDA/KV-cache
  startup cost; `StreamAsync`/`ToolLoop`'s detokenizer/filter/channel transport) held up when measured directly
  — both are flat/negligible. The probe measured only greedy decoding; the voice session's `TextRequest`
  defaults to `Temperature=0.7, TopP=0.95, Greedy=false`, and `TopPStep`'s nucleus filter allocated three
  vocab-sized arrays and argsorted the full ~152K-token vocabulary through a `Comparison<int>` delegate — every
  decode step, only on the non-greedy path.
  - **Fix.** `TopPStep`/`TopKStep`/`MinPStep`/`SamplerChain` reuse per-generation scratch buffers instead of
    allocating fresh ones every token. `SamplerMath.SortDescendingByValue` sorts via `Array.Sort(float[], int[])`
    (no delegate dispatch) instead of a comparer. `TopPStep` additionally sorts only the candidate tokens whose
    probability already clears a tiny floor when their own total reaches `p` (provably identical to sorting the
    whole vocabulary whenever it applies — see the type's remarks), falling back to a full sort otherwise.
  - **Numbers (Qwen3-4B Q4_K_M, 4090):** session-default sampling's first-10-token latency: 25.2 ms/token before
    (39.7 tok/s) → 7.6 ms/token after (131.6 tok/s); the realistic voice-turn bench (`ToolLoop.RunAsync` →
    `StreamAsync`) moves from 27-38 tok/s to 94-106 tok/s. Greedy decoding (unaffected code path) holds steady at
    110-160 tok/s across every measurement pass. Reply text and token-ids are byte-identical to the pre-fix
    output in every case measured. [Results](benchmarks/results/2026-10-01_llm_short_reply_decode.md).
- `TopPSortRefactorIdentityTests` keeps the pre-fix sampler algorithm verbatim as a reference and checks the new
  code masks identical logits across peaked/near-flat/pre-masked-tail/fallback-forcing distributions up to
  151,936 (Qwen3's vocabulary) and 2,000,000 elements — `regression-ab.sh`'s identical-output arms are greedy
  and never reach this code at all.

## alpha.235

- **Whisper's GELU is now the exact erf form Whisper uses.** OpenAI's Whisper applies `F.gelu` after both conv-stem
  convolutions and `nn.GELU()` in every MLP, and HF's configs say `activation_function: "gelu"`. `WhisperEncoder`
  and `WhisperDecoder` called `backend.Gelu`, the tanh approximation, which differs by up to 4.7e-4 per activation;
  they now call `backend.GeluErf` (CUDA `gelu_erf_f32`, Vulkan `gelu_exact`, the `IBackend` default on CPU).
  - **Tokens:** with the alpha.233 log-mel, CPU greedy decodes against `WhisperForConditionalGeneration` on the
    reference features, with the engine's decoding rule, are identical on all 144 cases — tiny, base, small.en,
    medium, distil-large-v3 and v3.5, 12 clips, with and without timestamps — where the tanh form left 4 near-ties
    (reference margins 0.0037–0.032 logits) resolved the other way.
  - **Latency (RTX 3060, small.en):** unchanged — 2 / 5 / 10 s in 111–122 / 112–115 / 187–189 ms median on the old
    512-point log-mel and 106–119 / 106–110 / 177–189 ms on alpha.233's, gate met; GPU tokens identical to the
    respective previous build on every bench case. [Results](benchmarks/results/2026-10-01_whisper_erf_gelu_3060.md).
- `WhisperExactGeluTests` reduces a tiny synthetic Whisper to its GELUs and layer norms and checks the encoder and
  the decoder against the exact form in double precision; the tanh form at any one of the four sites fails it.
- `WhisperBenchTests` exempts exactly the 2 s slice from the narrowband recall gate, with the measured reason beside
  the constant. It used to exempt every slice under 5 s, so a shorter slice added later would have gone ungated
  unnoticed.

## alpha.234

- **RNNoise has an opt-in int8 precision, and with it the voice front end meets its gate.**
  `RnnoisePrecision.Int8` runs conv2 and the three GRUs on the int8 tables that upstream's default C build compiles
  in. conv1 and the two heads stay F32, as they do there. A 10 ms frame reads 3.2 MB of weights instead of 11.5 MB,
  and a paired 20 ms frame 4.5 MB instead of 16.9 MB.
  - **Tables.** `RnnoiseInt8Tables` reads them out of `src/rnnoise_data.c` in the pinned xiph tarball into
    `rnnoise_int8.safetensors`. All 24 arrays are byte-equal to what gcc compiles from that file (a SHA-256 per array
    in `RnnoiseInt8TablesTests`). The GRU index lists are dense, and the conversion checks that.
  - **Numerics.** Upstream's default `./configure` build is its SSE2 path, not AVX2: `--enable-x86-rtcd` defaults to
    off and nothing adds `-march`. It codes activations as `127 + floor(0.5 + fl(127·x))` and sums the uint8 × int8
    products exactly in int32. This follows that build. The AVX2 path differs: its `maddubs` saturates each pair at
    int16, and it rounds `fma(x, 127, 127)` half-to-even.
  - **Op.** `IBackend.QuantizeActivationsU8` and `IBackend.LinearI8U8`, implemented on the CPU backend in
    `Int8GemvKernels`. AVX2 widens the bytes and multiplies with `pmaddwd`; the scalar path gives the same bits.
    Two activation rows share each weight load, so the paired 20 ms path still reads its shared weights once.
  - **Selection.** `RnnoiseWeights.LoadFile(path, precision)`. F32 is the default, and the wake stack loads F32,
    which a test checks. `RnnoiseInstaller` installs the tables when asked for Int8.
- **Parity with upstream's default build** (48 kHz jfk.wav, per-frame error, clean / 5 dB noise): median
  0.034 % / 0.053 %, p99 0.46 % / 0.35 %, max 1.57 % / 0.73 %. For scale, the F32 port against upstream's float build
  is 0.017 % / 0.019 %, and upstream's two builds differ from each other by 0.131 % / 0.163 %. The int8 sums are exact,
  so what remains is what separates the F32 port from upstream's float build: summation order in conv1 and the heads,
  exact against approximated tanh and sigmoid, and the front end. A slightly different activation can move a code by
  one step.
- **Quality, int8 against F32.**
  - SNR goes from 5.0 to 11.1 dB at both precisions. Noise between words drops 33.0 dB at int8 and 32.6 dB at F32,
    and speech moves 0.4 dB at both.
  - Whisper small.en's content-word recall through the full chain is 100 % at both precisions: clean and at 5 dB SNR,
    at 16 kHz and narrowband.
- **The gate, measured at int8** (2026-10-01; i7-6900K, CPU 0, `schedutil`, spinning 20 ms clock, 2,000 frames per
  run, interleaved, every run checked for foreign builds, tests and SwarmUI requests):
  - Quiet: p50 1.07–1.08 ms, p99 1.20–1.59 ms (gate: 3 / 5 ms).
  - 4 streaming threads: p50 1.80–2.11 ms, p99 3.56–4.25 ms, none late (gate: none late, p99 10 ms).
  - 8 streaming threads, not gated: p50 2.66–3.24 ms, p99 4.34–6.17 ms, none late.
  - Every run allocated 0 bytes and ran no GC. F32 measured p50 6.6–7.3 ms and p99 10.5–11.1 ms under 4 threads the
    same morning, which misses.
- **Sleeping clock.** A new bench mode, `HARTSY_VOICE_FRONTEND_BENCH_PACED=sleep`, sleeps to each tick as the voice
  host's audio thread does.
  - Under `schedutil` the core then idles between frames and runs at 1.3–1.7 GHz mean instead of 3.5 GHz.
  - Quiet: p50 2.93–2.94 ms, p99 3.91–4.05 ms.
  - 4 streaming threads: p50 3.63–4.12 ms, p99 5.84–6.90 ms, none late.
  - Wake-up delay after the tick: p50 0.08–0.15 ms.
- **Bench.** `VoiceFrontendBenchTests` now asserts the redefined gate on the spinning clock. It also takes
  `HARTSY_VOICE_FRONTEND_BENCH_PRECISION=int8`.
- **Tests.**
  - Kernel: exact sums, the epilogue, AVX2 against scalar, and row independence, plus the tie rule of the codes.
  - Table conversion: the gcc digests, and refusal of sparse or misshapen arrays.
  - Weights: gate order and conv2's column order.
  - Pairing, allocation and real speech, now at both precisions.
  - Parity against the default build, the wake stack's F32, and recall. `LinearI8U8` refuses LoRA adjuncts.

## alpha.233

- **Whisper now sees the log-mel it was trained on.** The front end zero-padded the 400-sample window into a
  512-point FFT (257 bins at 31.25 Hz instead of 201 at 40 Hz) and did not center the STFT, so a 30 s window was
  2997 frames and 1499 encoder positions instead of 3000 and 1500. It now runs the 400-point STFT with reflect
  padding of 200 and drops the last frame, as OpenAI's `log_mel_spectrogram` and HF's `WhisperFeatureExtractor` do.
  - **Features:** against `WhisperFeatureExtractor` (transformers 5.15, CPU) on 12 clips — JFK at 16 kHz and
    8 → 16 kHz narrowband, whole and cut to 2 / 5 / 10 s, 5 s of silence, 5 s of white noise, 30 s and 44 s of
    speech — the largest difference on the normalized log-mel is 1.6e-5 at 80 mels and 2.6e-5 at 128 (mean
    ≤ 1.1e-7), the size of the reference's own torch-versus-numpy spread (1.3e-5 and 1.4e-5).
  - **Tokens:** greedy decodes, with and without timestamps, against `WhisperForConditionalGeneration` run on the
    reference's features with the engine's decoding rule, CPU backend: 70 of 72 identical for tiny, base and
    small.en (58 of 72 before), 24 of 24 for medium and distil-large-v3.5, 22 of 24 for distil-large-v3. The
    four that differ are near-ties the reference itself wins by 0.0037–0.032 logits, and the engine picks the
    reference's runner-up. Whisper uses the exact (erf) GELU where the engine uses the tanh form; with the exact
    form tiny, base and small.en are identical on all 72. That swap is not in this release.
  - **Output moves by design:** transcripts change where the old input tipped a decision. small.en on the 2 s
    narrowband slice now ends "And so my fellow Ameri-", as the reference does on the same samples; JFK stays
    11/11 at 16 kHz and narrowband.
  - **Latency (RTX 3060, small.en, in-process, two runs per build):** 2 / 5 / 10 s utterances in 95–116 /
    105–119 / 173–178 ms median against 97–125 / 110–114 / 185–194 ms for the previous build; gate (≤ 350 ms) met
    at median and p95 (≤ 140 ms). The mel stage itself fell from 4.1 / 8.9 / 16.8 ms to 1.8–2.6 / 3.8–3.9 /
    3.9–5.2 ms. On the GPU at default precision the decodes equal the HF reference's on all 21 cases the bench and
    the parity set share (19 before). [Results](benchmarks/results/2026-10-01_whisper_logmel_3060.md).
- `MelSpectrogramExtractor.Config.ExactFftSize` transforms at exactly `NFft` points, through the allocation-free
  mixed-radix `FftPlan` when that is not a power of two (400 = 4·4·5·5), never Bluestein. Every other preset keeps
  its output bit for bit. `WhisperLegacyPow2Config` keeps the old layout for the S3 speech-tokenizer front end, so
  CosyVoice 2 and Chatterbox do not move with Whisper.
- `ComputeZeroPadded` takes centered presets, reading the reflect padding at both edges of the zero-padded window
  through index arithmetic; `Compute` no longer builds the reflect-padded copy. Frames fan out through
  `CpuParallel.For` in blocks sized by the FFT alone, each renting its scratch, so a warm extractor allocates
  nothing; `FftPlan.ForwardReal` takes a caller-owned work buffer so blocks share one plan.
- `WhisperBenchTests` evaluates the narrowband recall gate (≥ the 16 kHz baseline − 10 pts) on the full clip and on
  slices of 5 s and longer, and prints a gate column. The 2 s slice is still timed, but its recall is informational:
  its cut lands inside "Americans", where the reference model also drops the word's end on the narrowband samples.

## alpha.232

- **Kokoro's first synthesis of new text is no longer slower than a repeat.** Each new sentence length used to build
  about 39 cuDNN convolution plans, and building one is ~97 % heuristic query (~2 ms; finalize ~0.01 ms), so new
  text paid 60-80 ms that a repeat did not.
  - A 1D cuDNN conv now takes its engine configuration (engine and knobs) from its conv family's power-of-two length
    bucket. The heuristic runs once per family and bucket, at the bucket's top length minus one, and each new length
    only finalizes a plan from that configuration (~0.1 ms).
  - The reference is odd so the chosen configuration handles a ragged final tile and fits every length in the
    bucket; at a power-of-two reference the heuristic picked edge-free tiles that refused unaligned lengths. A
    configuration that still does not finalize for a length falls back to that length's own heuristic.
  - The choice depends on the length alone, so the audio does not depend on which lengths came first: the same 20
    sentences in reverse order give byte-identical audio.
  - Applies to every `groups == 1` `Conv1d` / `ConvTranspose1d` on CUDA (vocoders and codecs); 2D and 3D convs are
    unchanged. `numerics.audioConvLengthBuckets=false` is the kill switch back to the per-length heuristic.
- **Measured on the RTX 3060** (20 new 15-word sentences, then the same 20 again; p50 / p95):
  - Back to back: 212.9 / 255.6 → 180.1 / 220.3 ms. With 2-5 s idle gaps: 245.4 / 296.1 → 195.3 / 227.9 ms. The
    250 ms gate now holds for new text in both modes.
  - Repeats are unchanged (175.5 → 176.3 ms). Conv plan building per new sentence fell from 61 ms to 6 ms, with no
    fallbacks. A cold bucket still pays one heuristic round (about 70 ms) once per process, which a session warm-up
    across the 32-512-frame buckets removes.
- **Bounded, not byte-identical.** The per-length heuristic already picked different engines (46 / 55 / 56) for
  different lengths of the same conv, so no single choice per bucket reproduces it.
  - Kokoro: log-spectral correlation ≥ 0.99995 and max-abs ≤ 5.2e-3 against the previous build, same lengths, same
    Whisper recall. With the buckets switched off the build is byte-identical to the previous one.
  - Piper (VITS): same lengths; log-spectral correlation 1.000000 on four of six sentences, 0.99993 and 0.99497 on
    the other two.
- **Per-shape setup counters on `CudaBackend`:** `CudnnConvPlanStats` (plans built and how many came from a bucket,
  reference builds, fallbacks, build time split into graph, heuristic and finalize), `DescribeCudnnConvPlanFamilies`,
  `LtGemmPlanStats` and `GetMemPoolUsage`.
- Tests:
  - `CudnnConvPlanStatsTests` (CUDA): bucket and reference-length math, one plan per length from its bucket, the
    per-length heuristic with buckets off, byte-identical output whichever order two lengths of a bucket arrive in, and
    forward and transposed bucketed output within TF32 of the per-length heuristic.
  - `KokoroFirstSynthesisBenchTests`: the opt-in 3060 first-synthesis bench (back to back or with gaps, forward or
    reverse order, against reference audio). `ConvLengthBucketDigestTests`: the Piper A/B. `TtsConvBucketSpotCheckTests`
    and `TtsConvBucketSpotCheckCompareTests`: per-model arms on the 3060, scored on the CPU with Whisper.

## alpha.231

- **RNNoise runs the two 10 ms frames of a 20 ms voice frame layer by layer, bit for bit.** Per 20 ms frame it now
  reads 16.9 MB of weights instead of 23.1 MB.
  - `RnnoiseModel.ProcessPair` puts both frames' conv windows through each conv in one product, both frames'
    GRU input projections `W·x` through one pass over `W`, and both rows through the dense heads. Only the
    recurrent `U·h` stays one frame at a time.
  - That reads the conv weights (0.69 MB), the three input-side `W` (5.32 MB) and the heads (0.20 MB) once per 20 ms
    instead of twice. The recurrent `U` (5.32 MB, read twice) is 63 % of what remains.
  - `RnnoiseDenoiser.ProcessPair` analyzes both frames before synthesizing either. When either frame is silent, the
    other runs alone.
  - `RnnoiseStream` pairs two whole frames when one call holds them. It never holds one back, so latency is
    unchanged.
- **Measured against the previous build** (2026-10-01, i7-6900K, pinned to CPU 0). The builds were interleaved
  A/B, under the bench lock, with a separate check of running processes and of the SwarmUI journal before and after
  every run.
  - Quiet, back to back: p50 1.43–1.52 → 1.32 ms, p99 2.46–2.73 → 1.95–1.98 ms.
  - Quiet, at the live 20 ms cadence: p50 2.03–2.36 → 2.02–2.10 ms, p99 2.99–3.93 → 2.44–2.85 ms. The gate still
    misses here.
  - With 4 / 8 threads streaming memory on other cores: p50 6.1–6.2 → 4.3 ms and 9.5–10.0 → 6.7–7.0 ms. The
    RNNoise stage's time fell by about a third, a little more than its weight traffic.
  - Zero allocation and no GC in every run.
- Tests:
  - `RnnoisePairTests` runs a stream that takes one frame per call against one fed two frames, one and a half, two
    and a half, or random sizes per call. At 16 and 48 kHz it compares every output sample, and the speech
    probability after every call, bit for bit, and asserts that pairs with either frame silent occurred. The
    synthetic-weight cases run in the unit lane; the real-weight cases, on speech, are Integration.
  - `LinearTransBIdentityTests` checks that each row of a two-row product has the bits of that row alone, at the
    paired path's shapes, inline and fanned out.
  - Both also pass with `DOTNET_EnableAVX2=0`, which forces the scalar kernels.

## alpha.230

- **Mask Grow and the tight "inpaint only masked" crop now match SwarmUI.** Mask Grow expands the mask by
  `(grow + 1) / 2` pixels per side, as `SwarmMaskGrow` does; it used the full amount, so every inpaint and
  `<segment:>` mask was twice as wide as the same setting in SwarmUI. `Inpaint.CropToMask` and
  `Regional.ExactMaskOversize` let a caller ask for a crop at padding 0, which `ShrinkGrow` and `MaskOversize`
  could not express because 0 means off and default there. Both fields keep their meaning, so no published
  signature changes.

## alpha.229

- **The voice front end is about 3× faster at p50 and allocation-free, and still misses its 2 ms budget on this
  box.** Silero VAD plus RNNoise per 20 ms frame, on one pinned core with frames back to back
  (`VoiceFrontendBenchTests`, six runs on the final head): p50 1.57–1.70 ms, p99 3.72–5.72 ms, max 4.3–7.3 ms. In
  alpha.226 it was p50 5.1 ms, p99 7.0 ms and max 10.9 ms; the voice plan allows 2 ms. The thread's own CPU time
  matches wall time within 0.01 ms, so the slow frames are not preemption. Allocation fell from 102 KB per frame to
  none, and no GC ran while timed. Most of what remains is RNNoise's six F32 GRU products per 10 ms (1152×384 each),
  which are bound by memory bandwidth. F16 weights would halve that and int8 would quarter it; that is a precision
  decision, and it is left open here.
- **Off the quiet bench, weight traffic decides the budget.** Measured 2026-10-01 on the i7-6900K (20 MB L3), pinned
  to CPU 0.
  - **Only clean runs count.** Each run kept no foreign dotnet build or test before or after it, and no SwarmUI
    request landed during it, as checked against the SwarmUI journal (`tests/swarm-quiet-window.sh --verify-since`
    for the latest runs). Runs that overlapped a SwarmUI generation were discarded.
  - **At the live 20 ms cadence with no extra load**, p50 is 2.61–2.64 ms in three runs and p99 4.2–5.2 ms, so p50
    misses too. Between frames, other work on the box evicts the weights that the back-to-back bench keeps hot by
    re-reading them every 1.6 ms.
  - **With 4 threads streaming over large buffers on other cores**, p50 is 10.3–10.7 ms over two runs; with 8 it is
    15.5–16.4 ms, with p99 up to 18.6 and 28.2 ms. That load takes 19–22 GB/s of this box's DRAM bandwidth. The
    thread's CPU time rises with its wall time, so these are memory stalls, and the clock stayed at 3.5 GHz.
  - **At the live cadence against 8 streaming threads**, 632 of 2,000 frames started late, so the front end barely
    keeps up with real time.
  - **The same 8 load threads on L1-resident buffers change nothing**: p50 1.54 ms.
  - **Under load, each stage's time is its weight bytes over the bench thread's share of DRAM bandwidth.** That share
    is 2.1–2.3 GB/s against 4 threads and 1.4–1.6 GB/s against 8, and RNNoise and Silero agree on it within 3–11 %
    in every run. RNNoise reads its 11.5 MB of F32 weights twice per frame; Silero reads its 1.2 MB once.
  - By that arithmetic, F16 or int8 GRU weights shrink the stall in proportion, but neither gets under 2 ms against
    saturating load unless the weights also stay in L3.
- `FftPlan` (Audio, `Preprocessing`): a planned mixed-radix complex FFT ported from the kiss_fft RNNoise vendors —
  radix 2, 3, 4 and 5, twiddles and input permutation computed once, nothing allocated per call, upstream's
  operation order and twiddle table. `StreamingStft`, `StreamingIstft` and RNNoise's pitch transform use it only
  where `Fft` would take Bluestein: a size of 64 or more that is not a power of two. At RNNoise's 960 points Bluestein
  ran two padded 2048-point transforms and allocated 16 KB per call. Every other size stays on `Fft`, whose
  output is unchanged. `FftPlan` matches a double-precision DFT to 1e-6 relative at fourteen sizes.
- RNNoise's two convolutions each produce a single step, so they now run as `Linear` over flattened views of the same
  weights rather than through the generic `Conv1d` kernel, whose per-tap bookkeeping dominated at that length.
- `SileroVad` runs its STFT as one matrix product over the four hop-spaced windows, and each encoder convolution as
  an unfold followed by a matrix product, with activations kept time-major and the weights only viewed. Against the
  onnxruntime reference its per-chunk probabilities now differ by at most 8.9e-7 over 343 chunks of jfk.wav (4.77e-6
  before); against the previous forward, by at most 4.35e-6 over 686 chunks clean and noisy, with the same 233 chunks
  scoring as speech.
- `Resampler.ResampleRange` computes only the requested slice of outputs, running the interior ones as a vector dot
  product over reversed taps, which matches `Resample` to float rounding. `StreamingResampler` uses it and no longer
  resamples the context padding it throws away. Its output therefore matches the previous output to float rounding
  rather than bit for bit, and the rounding follows the CPU's vector width. That applies in `RnnoiseStream` and in the
  phone gateway's outbound path, its two users. `Resample` itself is unchanged. Measured floor against the previous
  block-and-slice output, on jfk.wav in 20 ms frames (AVX2):
  - max abs 3.0–4.2e-7 and RMS 2.1–2.3e-8, 136 dB under the speech's 0.142 RMS;
  - for 8 ↔ 16 kHz, 16 ↔ 48 kHz and 8 ↔ 48 kHz in both directions, and the gateway's 22.05 and 24 kHz → 8 kHz;
  - pinned below 1e-5 by `StreamingResamplerTests.SliceOnlyPath_MatchesTheOldBlockAndSlicePath_ToFloatRounding`.
- `RnnoisePitchAnalyzer`'s coarse lag sweep runs a vector of lags at once. Each lane sums its own lag in the original
  order, so the denoiser's output is bit-identical (all 528,000 samples of a 48 kHz test clip).
- **The front end no longer allocates.** `CpuParallel.For` captured its body in a closure, and C# builds a captured
  parameter's closure on entry to the method, so every `LinearTransB` and `Conv1d` call allocated 64 B and 104 B even
  when it then ran inline. The new `CpuParallel.For<TState>` takes the loop state explicitly and keeps its fan-out
  lambda in a separate method; those two kernels pass their tile state and a delegate held in a static field. The loop
  bodies moved without changing. `RnnoiseStream` at 16 kHz and `SileroVad` now allocate nothing per frame inside
  `CpuParallel.EnterInline()`. Other `CpuParallel.For` callers are untouched.
- `LinearTransB` runs four weight rows at a time. Each row still performs exactly the single-row sequence (products,
  horizontal sum, scalar tail and per-tile accumulation), so every CPU `Linear` output is bit-identical. It is faster
  because one row's dependent FMA chain left the core mostly idle: a 1152×384 product went from ≈ 75 µs to ≈ 58 µs on
  one core. `LinearTransBIdentityTests` compares against the previous loop byte for byte.
- RNNoise parity with upstream's float build is unchanged: median 0.017 % clean and 0.019 % with noise, with the
  same p99 and max against all four references.
- Tests: `FftPlanTests` (including a bit-for-bit pin to upstream kiss_fft at 960 points and a comparison with the
  Bluestein path at 960 and 480), `StreamingStftTests.Frames_ComeFromThePlanOnlyAtBluesteinSizes` (bit for bit: 512
  stays on `Fft`, 960 plans), `ResamplerTests.ResampleRange_MatchesTheSameSliceOfResample`, the streaming floor
  above, `VoiceFrontendAllocationTests` (zero bytes over 1000 frames), `LinearTransBIdentityTests`, and three
  stateful-`For` cases in `CpuParallelInlineScopeTests`. `SileroVadParityTests` logs its maximum difference.
- `VoiceFrontendBenchTests` changes:
  - It reports thread CPU time per frame beside wall time, and picks the core whose hyperthread pair is idlest.
  - It logs the bench core's clock from cpufreq.
  - Three new opt-in knobs:
    - `HARTSY_VOICE_FRONTEND_BENCH_LOAD_THREADS=N` runs a STREAM triad on N other cores, pinned outside the bench core's
      hyperthread pair.
    - `HARTSY_VOICE_FRONTEND_BENCH_LOAD=l1` keeps that load's buffers in L1, as a control without memory traffic.
    - `HARTSY_VOICE_FRONTEND_BENCH_PACED=1` starts one frame every 20 ms, spinning in between.

## alpha.228

- **Folders under the models root are matched ignoring case when the engine's spelling is missing.** On a
  case-sensitive filesystem the catalog's `LLM/qwen3/Qwen3-4B-Q4_K_M.gguf` never found the SwarmUI store's
  `llm/qwen3/`, so `qwen3` resolved to nothing and a download would have created a second `LLM/` beside `llm/`.
  `CaseInsensitivePath` (`HartsyInference.Core.IO`: `ResolveFile`, `ResolveDirectory`, `ResolveEntry`) resolves a path
  under a root one segment at a time:
  - an existing exact path comes back exactly as `Path.Combine` builds it, which is always the case on Windows and
    macOS;
  - otherwise each segment takes the one sibling of the right kind whose name differs only in case; two or more are
    ambiguous, logged once, and keep the engine's spelling;
  - a segment that resolves to nothing keeps its spelling along with the rest of the path, so a download lands in the
    folders that already exist. Only a missing segment costs a listing of its parent.
- Matching only fills former misses. A lookup that searches several places tries the engine's spelling in all of them
  before any case variant, so whatever it found before is still what it finds:
  - `ModelDownloader.TargetPath`: canonical, then legacy names, exactly, then both ignoring case. This covers the
    catalog path `ModelResolver` checks first, every download target and `MissingAssets`;
  - `ModelResolver`: catalog file, then modality-folder guess, exactly, then both ignoring case;
  - `ModelFileLocator`: its whole search over the named folders, then over folders that exist only in another case.
    This covers the CAM++ `audio/speaker` lookup;
  - `VisionModelPaths.FindYolo` scans `yolov8` exactly before its name probe.
- Single-place lookups match case variants directly: the `VisionModelPaths` conventional folders, the LTX-2
  latent-upsampler scan, the `AudioModelCache` root and the downloader's audio stand-in check.
- Below the audio root nothing is matched in another case. `AudioModelRoot` and `AudioModelCache` resolve only the
  `audio` folder itself, so RVC, YuE2 and Demucs look exactly where they did. YuE reads its checkpoint folder from
  `TargetPath`, so the loader reads where a download writes.
- The wake model root is now `WakeService.DefaultModelRoot()`, shared by the service,
  `SpeakerProfileStore.DefaultDirectory` and `hartsy wake train`. Documented in `docs/SETTINGS.md`.

## alpha.227

- **Whisper transcription is GPU-resident; small.en meets the phone-agent STT gate on the RTX 3060.** Per utterance
  (in-process, warm): 2 / 5 / 10 s in 124 / 116 / 194 ms median at 16 kHz (narrowband 8 k → 16 k: 122 / 113 / 201 ms),
  from 1358 / 954 / 1747 ms; device→host syncs per call 658 / 877 / 2045 → 8 / 11 / 27, one per generated token.
  The output is unchanged: the same tokens as before on the 3060 for small.en (the JFK slices, the full clip at 16 kHz
  and narrowband, timestamped and streaming decodes) and for tiny, small, medium, distil-large-v3 and v3.5, and on the
  CPU backend for tiny, base and small.en.
  - Decoder: the logits were a scalar host loop over the whole vocabulary (51 864 × 768 multiply-adds per token for
    small.en, 37 ms a step, 61 % of a 10 s transcription). They are one backend `Linear` at full F32 for that call
    (`WhisperOps.LinearFullPrecision`), so only the summation order changed. The head split and merge, the
    self-attention cache append and its prefix copy, and the last-row slice are backend ops (`Permute0213`,
    `KvCacheAppend` into a device-allocated cache, `SliceTimeRange`, `SliceRows`) instead of host loops that read every
    activation back; attention sees the same contiguous prefix through the same kernel. One causal mask serves all
    layers of a step.
  - Encoder: head split and merge on `Permute0213`; the stem transposes the conv output directly (a `Reshape` view read
    it back to the host); the positional add is a backend `Add` against a row view of the table.
  - Weights: the encoder and decoder preload what their device ops read on every call (idempotent, and it undoes an
    eviction), so biases and norms no longer upload on every op. A card that cannot hold them warns once and keeps
    the previous per-use upload and headroom-gated promotion instead of failing the call. The token table, which is
    also the tied logits weight, is read on the host through a pointer taken at load, so the per-step embedding lookup
    never drops its device copy.
  - `MelSpectrogramExtractor` (every preset): each mel filter sums only its nonzero bins. `ComputeZeroPadded` computes
    a zero-padded window without building the pad and fills frames that lie wholly in the padding with their constant.
    Whisper's 30 s window for a 2 s utterance went from ~155 ms to 4 ms (10 s: 17 ms). Bit-exact:
    `MelSpectrogramExactnessTests` compare every preset against the dense form, and the zero-padded path against
    `Compute` on the padded buffer at the window boundaries.
- One `diagnostics.profile` stage timer for the Audio package (`Audio/Diagnostics/StageTimer`, replacing
  `KokoroStageTimer`): wall time and D2H syncs per stage, repeated sub-stages summed with their means, and noted
  minimums. Whisper marks mel, encoder stem and layers, cross K/V, prompt, and per step embed / layers / logits /
  argmax, with the smallest top-1/top-2 logit margin; Kokoro's report now carries sync counts too.
- `WhisperBenchTests` (GpuIntegration, opt-in `HARTSY_WHISPER_BENCH=1`, asserts a 3060): the Probe A/C cases of
  `VoiceTurnBenchTests`, per-case token dumps for comparing builds, a model list for regression rows, stage and per-op
  profiles, and a CPU mode. Results: `benchmarks/results/2026-09-30_whisper_3060_perf.md`.

## alpha.226

- **RNNoise installs from xiph's own release.** The wake stack and the voice front end load
  `{wake root}/denoise/rnnoise.safetensors`, and nothing produced it: no one hosts a converted copy, so the AudioLab
  extension shipped an empty `DenoiserUrl`. `RnnoiseInstaller` (Engine) fetches
  `rnnoise_data-0a8755f8e2d834eff6a54714ecc7d75f9932e845df35f8b59bc52a7cfe6e8b37.tar.gz` — the tarball xiph/rnnoise's
  `download_model.sh` fetches from media.xiph.org — refuses it unless its SHA-256 is the one upstream pins in
  `model_version`, and converts its `models/rnnoise10Ga_12.pth`, the checkpoint behind the C library's default
  `rnnoise_data.c`, in C# through `RnnoiseCheckpoint` (Audio) and the safe-subset pickle loader. `EnsureAsync` downloads
  when the file is missing; `InstallFromTarball` installs offline from a copy. Nothing is re-hosted; the weights are
  BSD-3-Clause. The tarball's other checkpoint, `rnnoise10Gb_15.pth`, backs the smaller `_little` tables and is unused.
- `tools/convert_rnnoise.py` is the offline reference conversion (torch with `weights_only`); `--check-c-tables`
  proves the checkpoint is the one the C library compiles in by matching the three float tables bit for bit. Its output
  and the C# installer's are tensor-identical.
- `RnnoiseWeights.Load` refuses a missing or mis-shaped tensor by name — a checkpoint with upstream's default 256-wide
  GRU used to load and fail on the first speech frame — and a load that throws part-way no longer leaks the tensors it
  had already bound. `WakeModelSet.LoadDenoiser` disposes them too, and its missing-file warning names the installer.
- `AudioFileFetcher` takes an optional SHA-256 and discards a download that does not match before it reaches its path.
- Fix: `RnnoiseStream.LatencySamples` counted the denoiser's delay as a window plus a frame. It is one window —
  overlap-add plus the gain lookahead, 20 ms, which is also upstream's delay — so at 16 kHz it now reports 640 samples,
  where the output measurably lands, instead of 800.
- Verified 2026-09-30 on CPU F32 over jfk.wav at 48 kHz, as per-frame error against the reference clip's RMS: against
  upstream's `--enable-dnn-debug-float` build, median 0.017 % clean and 0.019 % with white noise at 5 dB SNR (p99
  0.26 % / 0.08 %); against the stock build, whose conv2 and GRUs run on int8 copies, 0.137 % / 0.166 %, which is
  upstream's own int8-to-float gap (0.131 % / 0.163 %). At 16 kHz through `RnnoiseStream`, jfk with white noise at
  5 dB SNR: SNR 5.0 dB in, 11.1 dB out, noise in the gaps between words down 32.6 dB, speech level within 0.4 dB.
- **The voice front-end budget is missed.** The voice plan allows Silero VAD plus RNNoise 2 ms per 20 ms frame on one
  core; `VoiceFrontendBenchTests` measured p50 5.1 ms, p99 7.0 ms and max 10.9 ms over 2,000 frames on a pinned core
  (process CPU time over wall time 1.02, no other test run on the box), allocating 102 KB per frame. The time goes to
  RNNoise's three 960-point FFTs per 10 ms (Bluestein, allocating 16 KB each), its second convolution through the
  generic Conv1d kernel, Silero's convolutions, and the F32 GRU products. The fix follows in the next version.
- Tests: `RnnoiseCheckpointTests` (synthetic tarball in the unit lane, the real one as Integration),
  `RnnoiseInstallerTests` (pin refusal in the unit lane, the real download under `Network=Real`),
  `AudioFileFetcherTests` (the hash gate), `RnnoiseRealSpeechTests` (Integration), and `VoiceFrontendBenchTests`
  (Integration, opt-in with `HARTSY_VOICE_FRONTEND_BENCH=1`). `RnnoiseParityTests` now logs the distribution it
  measures, and its regeneration notes name the debug-float build as the like-for-like reference.

## alpha.225

- **Phone gateway exe (`src/HartsyInference.PhoneGateway`, not packaged).** The SIP/RTP leg of the phone-call voice
  agent, on sipsorcery 10.0.16 (pinned with its license note) plus the repo's own media path: `ClockedAudioSource` is
  the RTP clock (one foreground thread per call on absolute `MonotonicClock` deadlines, `SCHED_FIFO` when `LimitRTPRIO`
  allows it and a 150 µs spin tail otherwise, `SpscRing` drain → G.711 encode → `SendAudio`, comfort silence on every
  tick, up to five catch-up frames then a resync, `Flush` applied on the next tick, zero allocation after its on-thread
  warm-up); `RtpJitterBuffer` (eight 20 ms slots, three-frame pre-buffer, repeat-once-then-silence concealment,
  reorder/late/duplicate/reset counters, depth trimming) is fed from `OnRtpPacketReceived` with one copy and drained by
  a pump thread that decodes, resamples 8→16 kHz and queues on the link; `EngineLink` dials the voice host's Unix socket
  (backoff with full jitter, Hello/HelloAck, ping, a liveness watchdog that also catches a blocked write, a control lane
  whose senders wait for room while the reader thread never does, a wedged lane dropping the frame into
  `link_control_lane_dropped_total` and restarting the connection, audio lane drop-oldest, flush-epoch drop rule);
  `LinkOutageGuard` holds a caller through a host outage with an embedded prompt, re-announces the call on reconnect and
  hangs up after a timeout; `CallController` is the one-call state machine (greeting, RFC 4733 DTMF both ways,
  `hangup`/`send_dtmf`/`transfer`/`hold`/`unhold`/`play_prompt` tools; the dial plan fails closed: outbound calls
  (`POST /calls`) and transfers need `sip.destinationPrefixes`, which allows only matching numbers through the
  registrar, refuses a destination with a port, URI parameters (`maddr`, `transport`), headers or another host, and
  dials `sip:<number>@<registrar>` rebuilt from the validated number, and with no prefixes every destination is refused
  (403) unless `sip.allowAnyDestination` is set for a LAN or development, which warns at start-up and cannot be combined
  with prefixes; a failed media setup answers 500 and leaves the gateway idle). New INVITEs are screened on the
  transport before sipsorcery's user agent sees them (486 while a call is up, 503 with the host down, 603 by policy),
  each refusal on its own server transaction and counted once per INVITE (Call-ID + top Via branch, 32 s). A fault on
  the RTP tick or inbound pump thread ends the call instead of leaving it half-dead: logged once,
  `calls_media_fault_total`, BYE and `CallEnd(Failed)`, or, before the call is announced, a BYE after the ACK and
  nothing sent to the host. `PhoneMediaSession` advertises the STUN/literal public address and latches on the first
  packet. Config is a JSON file; secrets are files it names (`sip.passwordFile`, `link.tokenFile`, `admin.tokenFile`),
  read once, refused when group or others can access them, never logged, and under systemd delivered with
  `LoadCredential=` (`docs/SETTINGS.md` notes it); `/health`, `/metrics` (Prometheus) and token-gated `POST /calls` on
  loopback; recording off by default. `HartsyInference.Phone.slnf` builds it without the GPU packages. Docs:
  `docs/Research/PHONE_GATEWAY.md`.

## alpha.224

- **Kokoro synthesis 6.7× faster on the 3060** (15-word sentence 1145 → 170 ms median in-process, 165–191 across nine
  runs; 5 words 778 → 82 ms, 30 words 2076 → 254 ms; 1552 → 9 device→host syncs per call). The synthesis graph stays
  device-resident: the AdaIN / AdaLN style splits, the length regulator, the style broadcast and channel concats,
  reflection pads, residual adds and the PLBERT head permutes are backend ops (`SliceLastDim`, `LayerNormModulate`,
  `RepeatTime`, grouped `ConvTranspose1d`, `Concat`, `GatherRows`, `Permute0213`, `Add`/`Scale`), and `KokoroPipeline`
  preloads its weights once per backend and keeps the two style halves resident for the call. Bounded, not
  bit-identical, by TF32 rounding only: log-magnitude-STFT correlation vs alpha.218 0.997 / 0.995 / 0.989 for 5 / 15 /
  30 words with identical lengths and transcripts — closer than alpha.218's own full-F32 output (0.994 / 0.986, and a
  flipped duration on 30 words) — and waveform-identical to alpha.218 when both run at full F32 (correlation
  ≥ 0.99999). StyleTTS 2, which shares the predictor, decoder blocks and `BiLstm`, is waveform-identical at full F32
  (correlation 1.000000) and 2–3× faster, and CosyVoice 2, which shares the NSF DSP, is unchanged (correlation
  1.000000). Evidence and the remaining levers in `benchmarks/results/2026-09-30_kokoro_3060_perf.md`.
- **`BiLstm` runs its recurrence on the host.** Both directions' input projections are one GEMM over the whole sequence
  (the two `W_ih` stacked at load), read back once; the sequential `h·W_hhᵀ` step is a SIMD dot per gate row
  (`LstmOps.RunSequence`), the two directions through `CpuParallel.For(2, …)`. `LstmCell.Step` and
  `LstmOps.GateAndUpdate` are unchanged for `UnidirectionalLstm` and `SileroVad`. Pinned to the per-step cell in
  `BiLstmTests`.
- **The NSF vocoder DSP is parallel and still bit-exact.** `Fft.DirectDft` (n < 64) read `Math.Cos/Sin` for every
  `k·t` — 2·n² transcendentals per frame, 650 ms of a Kokoro sentence at n_fft = 20 — and now indexes a cached per-size
  `[n, n]` table of exactly those values (`DirectDftTests`). `IStft.Apply`, `NsfVocoderDsp`'s STFT and iSTFT head, and
  the NSF harmonic source fan fixed blocks of frames out through `CpuParallel` (so they honour `numerics.cpuThreads`,
  `CpuParallel.EnterInline()` and the process-wide worker ceiling), with a block size that depends only on the per-frame
  cost and per-block scratch from `ArrayPool<float>.Shared`. The harmonic source records each block's starting phases
  in a sequential additions-only pass and jumps each block's noise state with the new `DeterministicRng.Advance` (the
  xorshift step as a GF(2) matrix power), so every piece — and chunked streaming of the source — is bit-for-bit the
  old single-threaded loop at any core count, any cap and inline (`NsfVocoderDspTests`).
- `KokoroBenchTests` (opt-in, `HARTSY_KOKORO_BENCH=1`) and `SharedBlockRegressionTests` (opt-in,
  `HARTSY_SHARED_BLOCK_REGRESSION=1`, StyleTTS 2 through its espeak `en-us` front-end and a clean 24 kHz reference,
  CosyVoice 2), both GpuIntegration on the 3060: latency, sync counts, PCM dumps, a reference comparison (max-abs,
  waveform and log-spectral correlation), Whisper recall, and a full-F32 switch (`…_EXACT=1`) for precision-free A/Bs;
  `KokoroStageTimer` reports per-stage wall time under `diagnostics.profile`.

## alpha.223

- **Runner leases on the speech services.** `ISpeechService.OpenSynthesizerAsync(spec)` and
  `ITranscribeService.OpenTranscriberAsync(spec)` load a model exactly as `SynthesizeAsync`/`RunAsync` do (same catalog,
  cache key and download) and return a lease that pins the runner on the engine's own audio backend until disposed:
  - `ISynthesizerLease`: `SampleRate`, `Synthesize(text, SpeechRequest options) → float[]`;
  - `ITranscriberLease`: `Transcribe(ReadOnlySpan<float> pcm, int sampleRate, AudioRequest options) → string`.
- Lease calls run synchronously on the caller's thread. They take neither the engine's audio generation lock nor the
  device gate: the caller holds `DeviceGate` for `InferenceEngine.ComputeBackend` around each call. The transcriber
  takes PCM with no WAV round trip and resamples only when the rate differs from the model's, through the resampler
  the service's decode uses. It refuses word timestamps and diarization, because the service answers those from a
  different decode. A Piper lease refuses a voice other than the one its weights were loaded for.
- Opening runs the service path (generation lock, device gate, memory-pressure sweep), so it waits for a generation in
  flight; never await it while holding the gate. Any engine release (`Dispose`, `FreeMemory`, `SetBackend`,
  `SetPlacement`) revokes open leases:
  - it first waits, within the existing 120 s release budget, for a lease call in flight;
  - later calls throw `ObjectDisposedException`, and nothing signals the revocation sooner;
  - an open that straddles a release is refused.
- To re-open, the holder disposes the revoked lease (a no-op) and opens a new one. After `FreeMemory`, `SetBackend`
  or `SetPlacement` the engine stays usable, and the new lease reloads the model on its current backend.
- `Dispose` releases the pin, is idempotent and waits for a call in flight. Both new service members default to
  `NotSupportedException`, so other implementations of the interfaces keep compiling.
- Consumers: the voice session (Whisper small.en and Kokoro resident on one card for a call) and AudioLab's
  keep-resident setting.
- A host with the LLM on one card and audio on another builds the engine on the audio card and sends each LLM request
  with `TextRequest.Device`. Text slots build their own backend and gate only their own ordinal, so the LLM loads
  nothing on the engine's device. On such an engine every `GenerateAsync`/`StreamAsync` must set `Device`, or the
  model loads on the audio card. `ITextService.CountTokens` takes no device: it never loads a model or touches a
  backend, and it counts with an idle loaded slot's tokenizer or falls back to an estimate. Documented on
  `TextRequest.Device`, `ITextService.CountTokens` and in `docs/MULTI_GPU.md`.
- `SpeechService` and `TranscribeService` now resolve descriptor, variant and cache key through one helper each, shared
  with the leases. The evaluation order is unchanged, so the service paths behave as before when no lease is open.
- Tests (`Diffusion.Tests`):
  - `AudioRunnerLeaseTests` (unit; fake runners seeded into a CPU engine's caches) cover:
    - pin and unpin;
    - ten forced-pressure STT↔TTS switches with pinned runners kept and an unpinned one evicted;
    - service calls on the leased runner;
    - lease calls proceeding while a service generation holds the lock;
    - revocation by `Dispose`, `FreeMemory` and `SetBackend`;
    - engine and lease Dispose waiting for a call in flight;
    - concurrent and double Dispose;
    - no pin left after a failed open or one cancelled while queued;
    - an open that straddles a release.
  - `AudioRunnerLeaseRealWeightTests` (Integration, CPU): a Kokoro lease encodes to the service's exact WAV bytes, and
    Whisper-tiny gives the service's exact JFK transcript at 16 kHz and through a 24 kHz clip.
  - `AudioRunnerLeaseGpuTests` (GpuIntegration) passes on the RTX 3060, with Kokoro and Whisper small.en leases driven
    under `DeviceGate`:
    - Whisper small.en through a lease transcribes the JFK clip 11/11 and equals the service transcript;
    - Kokoro through a lease is Whisper-verified 6/6;
    - both stay resident and correct through three forced-pressure switches to Whisper-tiny.
  - `AudioEvictionPressure.Relax()` keeps a box that is genuinely short of RAM from evicting fake runners.

## alpha.222

- **`HartsyInference.Tools`: opt-in LLM tool calling.** New packable library (references Engine only; not part of the
  `HartsyInference` meta package). `Parsing/ToolCallFormat` + `ToolCallFormats.RulesFor/Detect` hold the per-family
  wire formats: Hermes/Qwen `<tool_call>{…}</tool_call>` (closing tag optional, the call ends at the balanced object),
  Llama-3 `<|python_tag|>` + JSON and the bare `{"name": …, "parameters": …}` object, Gemma 4's
  `<|tool_call>call:name{k:v,s:<|"|>…<|"|>}<tool_call|>` block (converted to a JSON arguments object), and Mistral's
  `[TOOL_CALLS][{…}]` array, single object and `name{…}` forms. `ToolCallParser` is incremental (one `StringBuilder`
  per span, no allocation on the plain-text path), reads markers split across deltas, completes several calls per turn
  with ids `call_<n>`, consumes the whitespace and closing marker around a call, and forwards anything that does not
  resolve into a call (invalid JSON, no `name`, a closing tag before the value balanced, a span past 64 KiB, an
  unterminated span at end) as plain text. Because every local GGUF types the markers as CONTROL/USER_DEFINED tokens
  that the passthrough detokenizer drops, the bare forms (`{` at line start, `{"name"` anywhere, `[` at line start,
  `name{` at line start, `call:` anywhere) are first-class rules, not fallbacks; they are strict, though: a bare JSON
  span is released at its first key unless it is `"name"`, and with the offered tool names known (the installed
  filter passes `request.Tools`) a bare span naming any other tool is text (a bare `name{` is released at its brace
  unless the name is a complete offered one), while the tagged forms stay permissive.
  `ToolCallStreamFilter : ITextStreamFilter` emits each call as `TextChunkKind.NativeToolCall` and stops after the
  first by default (`StopAfterFirstCall`), deferring the stop until every call closed by the same delta has been
  emitted; `ToolCalling.Install(EngineOptions, format?)` sets `TextStreamFilterFactory` for requests that offer
  `Tools` and returns null for every other request. `ToolRegistry` / `IToolHandler` dispatch a `NativeToolCall`, returning an
  `{"error": …}` result for an unknown tool or a throwing handler; `ToolSchema.FromDelegate` builds the JSON schema
  from a C# delegate (`[Description]`, string/integer/number/boolean/enum, defaults and nullable types optional,
  `CancellationToken` skipped) and `ToolRegistry.Add(name, delegate)` binds the model's arguments to it by name.
  `ToolLoop.RunAsync(ITextService, ModelSpec, TextRequest, ToolRegistry, maxRounds, ct)` streams a turn, dispatches its
  calls, appends the assistant turn (`ToolCalls`) and one `TextRole.Tool` message per result (`ToolCallId`, `Name`), and
  re-runs; each round's `Result`/`ToolCall` stop is suppressed in favour of one final `Result` and stop. Tool results
  travel as `TextChunkKind.Status` chunks (`Text` = `tool_result:` + result, phase `tool_result`, `ToolCall` set) since
  no existing kind fits and Engine gains none; the round limit yields a `tool_loop:max_rounds=N` status, the result and
  `StopReason.ToolCall` without dispatching the last call. Engine's only change is `InternalsVisibleTo` for the new
  tests. Real-weight gate `Qwen3ToolCallEndToEndTests` (Qwen3-4B Q4_K_M on CUDA, `hang_up`, `EnableThinking=false`)
  asserts the checkpoint's template compiles to `JinjaChatTemplate`, renders `<tools>`, and streams a parsed call ending
  as `StopReason.ToolCall`.

## alpha.221

- **English-only Whisper checkpoints transcribe.** `openai/whisper-{tiny,base,small,medium}.en` and
  `distil-whisper/distil-{small,medium}.en` returned an empty transcript after walking to the token limit (~11 s per
  utterance on a 3060; 98.7 s on CPU for the 11 s JFK clip). Their vocabulary has 51864 entries and every special token
  sits one below the multilingual layout (EOT 50256, SOT 50257, `<|transcribe|>` 50358, `<|notimestamps|>` 50362,
  `<|0.00|>` 50363), and their decoder prompt is SOT + `<|notimestamps|>` with no language or task token. The tokenizer
  hardcoded the multilingual ids, so the prompt was `<|en|>, <|zh|>, <|transcribe|>, <|notimestamps|>` and the decode
  loop waited for an EOT (50257) the model never emits. `WhisperTokenizer` now resolves every id (`EotId`, `SotId`,
  `FirstLanguageId`, task, no-speech, no-timestamps, first timestamp) from the checkpoint's `added_tokens.json` and
  `vocab.json`, exposes `IsMultilingual` (OpenAI's `n_vocab >= 51865` rule) and builds the English-only prompt from it;
  `WhisperPipeline` stops on the checkpoint's EOT and suppresses from its SOT, and exposes `IsMultilingual`. The v2 and
  `.en` HF files name the no-speech token `<|nocaptions|>`, so the old `<|nospeech|>` lookup fell back to 50362 for
  them: right for the v2 layout, `<|notimestamps|>` on `.en`; the v3 files name it `<|nospeech|>` and resolved
  correctly before. Multilingual decodes (v2 and v3 layouts) are unchanged.
- `WhisperConfig.IsMultilingual` drives the config's special-token ids; the `.en` presets and the two distil `.en`
  presets carry `VocabSize` 51864 and pad 50256. `WhisperPipeline.LoadAsync` takes `vocab_size` from the checkpoint's
  `config.json` for an inferred preset, and `WhisperDecoder.LoadWeights` refuses an embedding or `proj_out` whose
  shape disagrees with the config — the old 51865 assumption read the tensor after the embedding as a phantom logit —
  and `LoadAsync` refuses a directory whose `config.json` and tokenizer files disagree about the layout.
- STT catalog: `whisper:tiny.en` / `base.en` / `small.en` / `medium.en` resolve to the English-only releases, and the
  catalog drops language and task for an English-only pipeline, so callers no longer need the `Language=""` trick
  (which still works).
- Tests: `WhisperTokenizerLayoutTests` (synthetic `.en` and multilingual layouts, unit lane), `.en` preset and
  `InferConfig` assertions in `WhisperConfigTests`, and `WhisperEnglishOnlyTests` (Integration + RealWeights, CPU):
  `whisper-small.en` transcribes the JFK clip 11/11 content words in 64 s, `whisper-tiny` unchanged and pinned.

## alpha.220

- **Text stream lifetime.** `TextStreamPump` now uses an unbounded channel, reversing alpha.206's bounded (256) one by
  decision: the sink runs on the decode thread inside the slot lock and the device gate, so a bounded channel with a
  blocking writer let a slow SSE client stall decode while holding the GPU. Memory stays bounded by `MaxTokens` deltas
  and abandonment still cancels the producer through the linked token. The `Error` stop chunk carries the exception
  message in `Text`; a pre-cancelled token and a load failure both end the stream within 100 ms (unit-tested).
- **Behaviour change:** `TextRequest.SystemPrompt` is no longer dropped when `Messages` is set: `PromptBuilder` prepends it
  as a system turn unless `Messages[0]` is already a system message (a host that folds its system text into the first
  message keeps its single copy), so the CLI's `--system` now takes effect. `GenerationRequest.EffectiveMessages()` is
  that single rule, and the output parser resolves its initial state from the same view the template renders.
- `IncrementalDetokenizer`'s fallback for tokenizers without `TokenBytes` decodes a short window from the last
  emitted boundary (the tokens emitted there stay as context) and holds a delta back while it ends in U+FFFD, instead
  of re-decoding the whole id list per token. Concatenated deltas equal a one-shot decode; the GGUF byte path is unchanged.
- **Generic tool-call plumbing (Engine/LLM only).** `TextMessage` and `ChatMessage` carry `ToolCalls`, `ToolCallId` and
  `Name`; `GenerationRequest.Tools` and a default-interface `IChatTemplate.Encode(..., tools)` overload pass tool
  schemas to the template. `JinjaChatTemplate` builds the OpenAI-shaped context (`tools` parsed from each schema, or
  null when none; per-message `tool_calls[].function.{name,arguments}`, `tool_call_id`, `name`, `reasoning_content`);
  `ChatMlTemplate` renders Qwen2.5's `# Tools` block, `<tool_call>` turns and `<tool_response>` results byte for byte
  with the real template; a conversation without tools or tool turns renders exactly as before, while a `tool`-role
  message now lands inside a user turn as `<tool_response>` (Qwen's format) instead of a bare `tool` turn. New `ITextStreamFilter` seam
  (`OnDelta`/`OnEnd` → `TextFilterResult{ForwardText, ToolCall, Stop}`) installed per request through
  `EngineOptions.TextStreamFilterFactory`: `RunText` routes every content delta through it, emits
  `TextChunkKind.NativeToolCall`, and a stop ends generation as `StopReason.ToolCall` with `TextResult.ToolCall`
  set. `TextRequest.Tools` still arms the `<tool_call>` sentinel grammar. Parsers, registries and the agent loop are
  not here; they arrive with the separate Tools package, and `ForceToolId` stays unimplemented (documented).
- OpenAI-compat `/v1/chat/completions` maps `tool_calls`, `tool_call_id` and `name` both ways; the two new
  `ChatMessageDto` fields are omitted when null so existing responses are byte-identical.

## alpha.219

- **Shared real-time helpers (Core).** `Runtime/MonotonicClock` (`CLOCK_MONOTONIC` reads and `clock_nanosleep` to an
  absolute deadline, EINTR retried, no allocation; Linux only, `Stopwatch`/`Thread.Sleep` fallback elsewhere),
  `Runtime/RealtimeScheduling` (`TryEnterFifo` after an `RLIMIT_RTPRIO` probe, `TryPinToCpu`; never throw, the reason
  names the `LimitRTPRIO=` / `limits.d` fix), `Runtime/SpscRing<T>` (lock-free single-producer/single-consumer ring,
  drop-newest with a dropped count, consumer-side `DiscardAll`), `Numerics/LatencyHistogram` (fixed µs buckets,
  p50/p99/max, allocation-free `Record`) and `CpuParallel.InlineScope` (a thread-local scope in which `CpuParallel.For`
  never fans out). `Thread.Priority` is a no-op on Linux, which is why the FIFO helper exists.
- **Sentence streaming is one shared helper (Audio).** `Streaming/SentenceChunkedSynthesis.StreamBySentence` /
  `StreamFromDeltas` turn any whole-utterance synth into a sentence stream with the streaming-codec producer/consumer
  discipline, `AudioStreamer` backpressure, a `run` delegate as the scheduler seam and cancellation before, inside and
  after every job. `Frontends/StreamingSentenceSplitter` applies `SentenceSplitter`'s rules to token deltas (never
  emits the open tail; a smaller first-sentence minimum), `SentenceSplitter.SplitClauses` cuts over-long sentences at
  clause marks, `Frontends/SpokenTextNormalizer` strips `<think>` blocks, markdown and emoji, `Io/G711` codes μ-law
  and A-law by table, and `Models/Wake/IVadModel` lets `SileroVadStream` be driven by any VAD (`SileroVad` implements it).
- **Piper streams through the shared helper** — its private sentence loop is gone. Streamed output is byte-identical
  to before (digest gate on `en_US-ryan-medium`). **Kokoro now streams by sentence** through the same helper: the
  voice pack is resolved once per job, G2P runs per sentence, sentences over 300 characters are clause-split so
  PLBERT's 512 positions hold. Whole-text `Synthesize` is unchanged (digest-identical to a direct pipeline call);
  the two-sentence stream is Whisper-verified at 10/10 content-word recall. Through `SpeechService.SynthesizeStreamAsync`
  the runtime now holds its generation lock for the whole Kokoro stream, where the old text-split loop released it
  between chunks.


## alpha.218

- **Audio eviction keeps the incoming model.** `AudioRuntime`'s memory-pressure sweep compared the prefixed job key
  (`tts:…`) against the caches' bare keys, so once free host RAM dipped under `vram.audioEvictBelowGb` (default 14 GB)
  every STT↔TTS switch evicted the runner about to run and reloaded it. Jobs now carry an `AudioJob` (cache + bare key)
  and the sweep keeps that key through its own cache. Demucs runners are keyed by the explicit local path or model name
  instead of the resolved on-disk path, which is the same identity.
- `AudioRunnerCache.Pin(key)` holds a runner resident through memory pressure (refcounted, name-level); engine release
  and backend switches still unload pinned runners. Groundwork for the voice-agent session and an always-resident setting.

## alpha.217

- **New package `HartsyInference.PhoneLink`** (phone-call voice agent PR4): the gateway-to-host wire protocol, depending on Core only and
  not part of the `HartsyInference` meta package. Unix-domain-socket transport, host listens and the gateway dials; 16-byte little-endian
  frame header (`u32 payloadLength | u8 type | u8 flags | u16 reserved | u32 callId | u32 sequence`) with a 1 MB payload cap rejected from
  the header alone. Audio is raw PCM16 (`InboundAudio` 320 samples at 16 kHz, `OutboundAudio` tagged with a turnId); handshake, call end,
  DTMF, ping/pong and the turn/request prefixes are binary; `CallStart`, `Event`, `ToolRequest`, `ToolResult` and `Error` bodies are JSON
  through a source-generated context. Spec with the flush/turnId epoch rule in `docs/Research/PHONE_LINK_PROTOCOL.md`.
- `LinkFrameWriter` stages header and payload in one pooled buffer and writes each frame with a single stream write; audio frames allocate
  nothing and a synchronously completed write never enters an async state machine. `LinkFrameReader` reassembles frames across any read
  boundary in one pooled buffer; `LinkFrame` exposes typed decoders that validate frame type and payload size. Single writer and single
  reader per direction, enforced.
- Tests (`tests/HartsyInference.PhoneLink.Tests`, unit lane): byte-exact header golden, a round trip for every message type, a two-frame
  stream reassembled at every chunk size and every split boundary, oversize rejected and the exact cap accepted, truncated versus clean end
  of stream, pinned JSON wire shapes, and zero allocation over 1000 audio writes and reads.

## alpha.216

- **LLM generation goes through one model contract.** `IGenerationModel` and `ISequenceState` (with cursor-only
  `Checkpoint`/`Rollback`) sit between the generation pipeline and the transformer, and `GenericTransformerModel` adapts
  the existing `GenericTransformer` to it. `TextGenerationPipeline` and `DynamicBatchScheduler` gain `IGenerationModel`
  constructors; the old constructors build the adapter, so public signatures are unchanged. Projection, RoPE-table and
  gated-FFN helpers moved into shared internal statics that `GenericTransformer` delegates to. No behaviour change is intended.

## alpha.215

- **EXL3 trellis decode** (DeepSeek-V4.1-Flash program PR 23b). `Exl3Codec` decodes the `sfxnz/DeepSeek-V4.1-Flash-EXL3` 2-bit MCG experts: 16x16 tail-biting trellis tiles
  through the MCG hash, a 128-point Hadamard per block and the `suh`/`svh` sign vectors. The trellis stage is bit-exact against exllamav3's own `reconstruct_tile` on a
  committed 256x384 fixture (`tests/HartsyInference.ModelAssets.Tests/Fixtures/Exl3/`). The full F32 result stays within 1.1e-7 of an fp64 dense-Hadamard oracle and within 1.2e-3
  of exllamav3's fused fp16 kernel, relative to max|W|; it is not bit-identical to that kernel, whose butterflies run in fp16.
- `QuantCompanionBinder` binds EXL3 experts from `.trellis` and refuses a missing `suh`, `svh` or `mcg`, bit widths other than 2 (naming the key and shape), geometry that is not a multiple of 128
  and wrong companion dtypes. Non-routed FP8 weights and the row-scaled `lm_head` bind through the FP8 path. New `DType.I16` and the safetensors `I16` mapping carry the trellis.
- `QuantRecipe` slicing: input-column windows on 128 boundaries slice the trellis and `suh`; a partial output-row window throws `NotSupportedException` naming the key and range.
- **CUDA.** `dequant_recipe_to_bf16` gains `dequant_exl3_2bit_to_bf16` (sm_80 PTX regenerated; the existing entries are unchanged). `CudaQuantWorkspace` dequantizes EXL3 matrices to BF16
  and `QuantExecutionPolicy` plans W2A16. The device result equals the host F32 rounded once to BF16, bit for bit, on an RTX 3060.
- Not done: no real-checkpoint tensor has been decoded, the BF16 output is coarser than exllamav3's fp16, output-row windows, a Vulkan path and `IBackend.Linear` wiring.

## alpha.214

- **Derivative quant formats** (DeepSeek-V4.1-Flash program PR 23). `ModelOptNvfp4Codec` decodes NVIDIA NVFP4 (low nibble first, E4M3 scale per 16,
  scalar F32 `weight_scale_2` multiplied into the scale first) and `AffineIntCodec` decodes MLX affine `q*scale + bias` per 64 at 4 or 8 bits
  (new `QuantEncoding.AffineInt8`; the V4.1 MLX checkpoint mixes both). AMD Quark MXFP4 reuses `Mxfp4E8M0Codec` with U8 scales. Codecs match an independent
  numpy decoder and `mx.dequantize` bit-exactly (`tests/python-reference/deepseek_v41/fixtures/derivative_quant_codecs.json`).
- `QuantCompanionBinder` binds the hybrid NVFP4 layout (FP8 attention with F8_E8M0 `.scale`), Quark FP8 attention scales and MLX 4/8-bit inferred from
  the scale shape, and refuses experts without `weight_scale_2`, non-U8 Quark scales, wrong packed widths, group sizes other than 64 and non-F32 MLX scales.
  `QuantRecipe`/`ExpertMatrix`/`ExpertWeights` carry the bias and global-scale companions so the expert cache uploads them.
- **CUDA.** `dequant_recipe_to_bf16` gains `dequant_nvfp4_modelopt_to_bf16` and `dequant_affine_to_bf16` (sm_80 PTX regenerated), and `CudaQuantWorkspace`
  dequantizes NVFP4, Quark and MLX expert matrices to BF16 bit-exactly against the host codecs. `QuantExecutionPolicy` plans the BF16 dequant and
  labels NVFP4 as W4A16 (`input_scale` is left unread; no W4A4 is claimed).
- Deferred: EXL3 trellis decode (PR 23b), the 32x32 block-FP8 native GEMM, native NVFP4/MXFP4 block-scaled GEMM on Blackwell, DwarfStar GGUF Engram row264.

## alpha.213

- **Vulkan paths for the DeepSeek-V4.1-Flash primitives** (program PR 24b). New compute shaders, each parity-tested against the CPU reference
  on an NVIDIA RTX 3060 through Vulkan: `MoeRoute`, `MoeBuildDispatch`, `MoeCombine`, `TopKLastDim`, `Softplus`; `HcSplitSinkhorn`,
  `HcPreMix`, `HcPostMix`; `SparseLatentAttention`, `IndexerScores`, `BuildWindowIndices`, `QuantizeLatentRows`, `ActQuantDequantInPlace`
  and the 5-argument `ApplyRopeInterleaved`. Integer outputs, quantizer bytes, window indices, rope and the HC mixes are exact; router
  weights are within 5e-6 relative, attention and indexer within 1e-5, Sinkhorn within 1e-6.
- Behaviour to know: `MoeCombine` and `QuantizeLatentRows` skip an out-of-range slot or destination row instead of throwing;
  `SparseLatentAttention` throws `NotSupportedException` for k above 3800 and `QuantizeLatentRows` for destination tensors that are not
  whole 32-bit words.
- `VulkanBackend.DequantRecipeToBf16` (shader `dequant_recipe_bf16`) widens MXFP4-E8M0 and block-FP8-E8M0 recipe weights to BF16,
  bit-identical to `Mxfp4E8M0Codec` and `Fp8BlockE8M0Codec` on synthetic layouts. It forms subnormal results on the BF16 grid directly because
  Vulkan devices flush float subnormals. `SupportsRecipeDequant` reports what it can decode. `SupportsResidentQuant(Tensor)` stays false on
  Vulkan: there is no block-scaled GEMM, so recipe weights are widened, not kept packed.
- The Vulkan expert cache is not implemented; experts are host-staged uploads. Native block-scaled GEMM on Vulkan stays `Unsupported`.
  No AMD hardware evidence has been collected; the NVIDIA-via-Vulkan runs are plumbing evidence only. Nothing here is wired into a model.

## alpha.212

- **Fix: DeepSeek-V4.1 checkpoint refused the real official Engram shards** (DeepSeek-V4.1-Flash program). Shards 47 and 48 hold each
  layer's embedding tables (`[rows,256]` F8_E4M3 + `[rows,8]` F8_E8M0) together with `engram.q_weight`, `engram.k_weight` and
  `engram.wkv.{weight,scale}` (verified against the `dba1be0a` index and shard 47 header), and `DeepSeekV41Checkpoint.Open` threw "Engram
  tables share shards with other weights". The tables stay pread-only and never mapped; the small Engram weights in the same shard are now
  pread into owned tensors by the new `ShardedSafeTensorSet.ReadTensor`, which `GetWeight` and `GetQuant` use for them. Any other weight
  sharing a table's shard is still refused. The tiny test checkpoint now mirrors the real shard layout.

## alpha.211

- **Engram constants, hasher and row store** (DeepSeek-V4.1-Flash program PR 13, minus the in-model module). The constants (compressed token
  map, multipliers, offsets, primes) are dumped from the unmodified upstream `engram.py` at checkpoint revision `dba1be0a`, committed under
  `DeepSeekV41/Engram/Constants/` (gitignore exception for `*.bin`), embedded, and SHA-256 checked on load; C# never re-derives them.
  `dump_engram_constants.py --cross-check` also compares them with the DwarfStar GGUF metadata (token map, primes, multipliers) and the MLX
  `engram_token_map.json`, recorded in the manifest.
- `EngramHasher` (XOR of compressed ids times multipliers over the lookbacks, modulo the per-head prime plus the cumulative offset; DEAD
  tokens and sequence-start lookbacks map to the pad id; history carries across prefill and decode) equals the upstream `NgramHashState`
  exactly on 15 cases.
- `EngramTableStore` in Core: `IEngramRowLayout` with the official FP8/E8M0, GGUF row264 and MLX affine layouts; `PrefetchAsync`, `Gather`,
  `Stats`; per-batch dedup and sort, reads merged per 4 KB page over `IWeightByteSource`, LRU slab row cache, owned-row range check,
  `Storage` and `HostResident` backings (`Device` throws `NotSupportedException`). `EngramTableStores.Open` builds one from a checkpoint table.
- Tests: layout decode vs Python dequant on edge, seeded synthetic and range-fetched real rows; store behavior on a synthetic shard file.
  A guarded Integration test gathers 10,000 random real rows from shard 47 and skips without the checkpoint; it did not run here, and
  neither did an RSS-bound or cold-latency measurement. `EngramModule` waits for PR 8.
- Known issue found, not fixed here: `DeepSeekV41Checkpoint.Open` rejects the real official checkpoint because shards 47 and 48 also hold
  `engram.q_weight`, `engram.k_weight` and `engram.wkv.*`, and pread-only shards cannot serve those tensors.

## alpha.210

- **Expert bank, expert cache and device dequant** (DeepSeek-V4.1-Flash program PR 12). `ExpertBank` builds each `ExpertWeights(W1, W2, W3)`
  once so the `Tensor` identities the weight caches key on stay stable. `IExpertCache` (`Acquire`/`Prefetch`/`Release`/`Stats`/`BudgetBytes`)
  is implemented by `CudaExpertCache` directly on `IStreamingWeightCache` and its pinned staging ring: a slot is one whole expert (one
  `BeginUploadAsync` of up to six tensors), in-flight uploads are shared, pinned entries are never evicted, victims come from a segmented
  LRU weighted by per-layer frequency, and `EvictAsync` runs after the last reader's event. The policy lives in `ExpertCacheBase`, is
  unit-tested on CPU against a fake, and `Dispose` cancels, waits, releases leases, evicts all and returns staging.
- `GpuTransferHelper.ExcludeFromAutoPromotion` keeps expert tensors out of the weight cache promotion (uploaded twice, at least 1 MB).
  `CudaExpertCache` forces `PinUploadSource` off so mmap-backed weights are never `cuMemHostRegister`ed. `ExpertRouting` reads
  the compact `[T, k]` I32 top-k from `MoeRoute`.
- **CUDA BF16 dequant for official recipes.** `dequant_recipe_to_bf16` (sm_80 PTX) unpacks MXFP4 with E8M0 scales per 32 and FP8 E4M3 with
  E8M0 blocks; `CudaQuantWorkspace` is a bounded ring of two BF16 slots (23,592,960 B, the largest routed-expert matrix) that bypasses
  `IBackend.Linear`'s F16 cast cache. `IBackend.Linear` does not use it yet (PR 23/24).
- Tests: 23 CPU policy tests, GpuIntegration tests on an RTX 3060 (hit/miss/dedup, prefetch sharing, pinning, promotion blocked, 1,000
  acquire/release cycles leak-free, bit-exact dequant vs the host codecs); an M1 test uploads layers.0 experts 0/191/383 and compares
  with the official fixtures, and skips without a populated shard 3 (it did not run against real bytes).

## alpha.209

- **DeepSeek-V4.1-Flash config, HF directory loader and catalog rows** (program PR 7). `HfCheckpointDirectory.TryProbe` recognises a
  config plus safetensors directory from `config.json` and the file listing alone, and `HfQuantFlavorDetector` names the producer
  (official, NVIDIA NVFP4, AMD Quark, EXL3, MLX) from `quantization_config`. `IHfKeyMapper` maps Official, MLX, EXL3 (`lm_head` to
  `head`) and DwarfStar GGUF names (`blk.N.*`, fused experts, row264 Engram) to the canonical names.
- `DeepSeekV41Config` parses `config.json`, requires `compress_ratios.Length == num_layers + 3` and derives per-layer plans;
  it has no model class. `DeepSeekV41Checkpoint` opens the shard set header-first (Engram shards pread-only) and exposes
  `GetWeight`, `ExpertBank(layer)` and `EngramTable(layer)` as borrowed views. A draft scan keeps `mtp` layers with missing experts
  from throwing at open and lets `RequireDraft()` refuse DSpark with the exact gap (MLX `mtp.2` lacks 14 of 128 experts).
- `ModelResolver` (Text) returns a directory that probes as a Hugging Face checkpoint and refuses a directory holding both a
  `.gguf` and a safetensors index; a single `.gguf` resolves exactly as before. `TextService` validates and opens a
  `deepseek_v41` directory, then throws `NotSupportedException` because the model class is not wired.
- `ModelCatalog` row `deepseek-v4.1-flash` with per-derivative `CatalogVariant` shard sets pinned to the inspected commits and
  `Components` flags (DwarfStar: no draft, separate vision file). It has no `Assets`, so the CLI offers no download.
- The text memory estimate reads headers only and reports `WeightBytesByClass` (dense, expert, Engram, embed, head, vision, draft)
  that sum to `metadata.total_size` (510,286,023,000 B on the official checkpoint); a new `MemoryComponent.LanguageModel` phase
  carries the resident bytes and the stream floor. `AssessAsync` still answers `Unknown` until the residency planner lands.
- Tests pin the real pinned `config.json` (fixture), folded real shard headers (12 KB template fixture) and, behind
  `HARTSY_DSV41_HEADER_REPLICA`, a sparse replica of all 48 real headers; probing, resolver, catalog, loader and estimate cases run on
  synthetic tiny checkpoints.

## alpha.208

- **Quant recipes and companion binder** (DeepSeek-V4.1-Flash program PR 5). `QuantRecipe` (`Core/Tensors/Quant/`) describes one
  quantized weight as encoding + `BlockGeometry` + scale tensor and slices scale-aware: `SliceRows` on block-row boundaries,
  `SliceCols` on block-column boundaries (byte aligned for 4-bit), otherwise `NotSupportedException` naming the key and range
  rather than pairing elements with another block's scale. `QuantWeightInfo.Recipe` carries it, and recipes use their own
  `Format` strings (`recipe-fp8-block-e8m0`, ...) so `BlockScaleFormats.FromQuantFormat` cannot route them into the Blackwell
  cuBLASLt path.
- `QuantCompanionBinder.Bind(inventory, QuantFlavor)` pairs each weight with its scale companions from headers alone
  (Official `.scale`/`.weight_scale_inv`, NVFP4, Quark, MLX affine gs64, EXL3) and infers a unique block geometry from the scale
  shape. One aggregated error lists unpaired weights, orphan companions, ambiguous or unmatched geometry and bad scale dtypes (the Official flavor decodes only F8_E8M0 or raw U8 scales).
- Host codecs `Fp8BlockE8M0Codec` (FP8 E4M3, 32x32 or 1x32 E8M0 scales) and `Mxfp4E8M0Codec` (E2M1, low nibble = even element)
  expose `DequantRows(packed, recipe, rowOffset, rowCount, dest)`; the scale is a multiplier `2^(e - 127)`, byte 255 is NaN.
- **Fix:** `ApplyFp8ScaledDequant` silently dropped a rank-2 fp8 `.weight_scale`, leaving the raw fp8 weight unscaled. It now
  throws for any non-scalar scale on an fp8 weight.
- Milestone M1 (`DeepSeekV41Shard3ParityTests`, `tests/python-reference/dump_deepseek_v41_shard3_ref.py`) compares the codecs to the
  official `convert.py` dequant on 64-row windows of shard 3 (`layers.0.*`: dense FP8 32x32, shared experts, routed experts 0/191/383
  MXFP4) plus a `Linear` check; it is skipped unless the shard and fixtures exist. Run so far against a sparse replica holding the
  real bytes of those windows only, so the matrix row stays InProgress.

## alpha.207

- **Single-latent sparse attention, indexer, hyper-connection, latent-quantization and window primitives are backend
  primitives** (`IBackend.SparseLatentAttention`, `IndexerScores`, `HcSplitSinkhorn`, `HcPreMix`, `HcPostMix`,
  `QuantizeLatentRows`, `ActQuantDequantInPlace`, `BuildWindowIndices`, and an `ApplyRopeInterleaved` overload with a
  `dimOffset`), DeepSeek-V4.1-Flash program PR 6. A `LatentSource` describes a row cache in F32, FP8 e4m3 + ue8m0/32,
  FP4 e2m1 + e4m3/16 or FP4 e2m1 + e8m0/32; attention treats one latent as both key and value, addresses a window ring
  then a main cache through one index space (-1 skipped), adds a per-head fp32 sink to the softmax denominator only, and
  dequantizes in-kernel. The CPU references live in Core and were checked against pure-torch dumps (quant bytes and
  act-quant results exact, attention within 1e-5, Sinkhorn within 1e-6). CUDA adds `latent_attention.ptx`,
  `latent_quant.ptx`, `hc_mix.ptx` and `latent_positions.ptx` (sm_80 baseline, F32 accumulation, drift-checked): attention
  and indexer agree with the CPU reference within 1e-5, quantization, act-quant, hyper-connection mixes, window indices and
  rope are bit-identical. Limits: attention needs k <= 12288, the indexer dim <= 512, hyper-connections hc <= 8, and
  `QuantizeLatentRows` skips (instead of throwing on) a destination row past the cache. Vulkan reports
  `NotSupportedException`. `MlaForward` and `MoeFeedForward` are unchanged; no model uses these yet.

## alpha.206

- **Streaming output parser** (DeepSeek-V4.1-Flash program PR 11). `IOutputParser` (`Push(tokenId)` / `Finish`) turns generated
  ids into `ParsedEvent`s (reasoning, content, tool-call begin/args/end/abort, stop, malformed). `DeepSeekV41OutputParser` matches
  `</think>`, EOS and the `\n\n<｜DSML｜ calls>` marker on the detokenized text with longest-suffix holdback, so a marker split
  across any token or UTF-8 boundary is still found, and streams each call's JSON arguments as they arrive with the reference
  grammar (`namespace::tool`, leading-space tags, `string="true|false"`). Every split point equals one-shot parsing and the
  upstream `encoding.py` parse on a committed 41-case fixture (also on the real tokenizer); malformed output raises
  `Malformed` events and never throws; `encode(parse(x))` reproduces the prompt. `PassthroughOutputParser` keeps every other
  template byte-identical.
- `IncrementalDetokenizer` replaces the O(n^2) re-decode for tokenizers that report `ILlmTokenizer.TokenBytes` (`GgufTokenizer`):
  bytes go through a stateful UTF-8 decoder that holds back an incomplete character. Other tokenizers keep the full re-decode.
- `TextChunkKind` gains `Reasoning`, `ToolCallDelta`, `ToolCallAbort` (a faulted call's deltas must be discarded) and `Usage`; `TextChunk` gains `ToolCallIndex`, `Status` and `Usage`
  (`TextStatus`, `TextUsage`, declared only: no Usage/Status chunk is emitted yet). A complete `NativeToolCall`
  (`call_{requestId}_{index}`, plus `Namespace`) still fires per call. `IConversationEncoder.CreateParser` wires the parser to
  the DeepSeek-V4.1 encoder.
- **`TextService.StreamAsync` lifetime fix** (`TextStreamPump`): a linked cancellation source, a bounded (256) channel and a
  `finally` that cancels and awaits the generation worker, so a consumer that stops early releases its slot in under a second
  instead of leaving generation running.

## alpha.205

- **Header-first sharded safetensors** (`ShardedSafeTensorSet`, DeepSeek-V4.1-Flash program PR 4). `OpenIndex` (driven by
  `model.safetensors.index.json`) and `OpenFiles` (odd names such as MLX's) read only each shard's 8+N header bytes with
  `RandomAccess.Read` and build a complete `name -> TensorLocation` inventory before anything is mapped: opening the
  official 475 GiB, 48-shard, 96,085-tensor checkpoint maps nothing. A shard is `mmap`ed on the first `GetTensor`, advised
  `MADV_RANDOM` (`MmapHandle.Advise`); shards listed in `PreadOnlyShards` are never mapped and are read through
  `IWeightByteSource` (`PreadByteSource`, batched async preads). The open fails with one aggregated, actionable error for
  a shard missing on disk, a key in the index but not its header (or the reverse or in another shard), a key duplicated
  across shards, or tensor bytes that do not sum to `metadata.total_size`.
- `SafeTensorHeaderReader` is the shared header validator (`SafeTensorsLoader.Load` and `CheckpointHeader` use it): oversized
  or truncated headers, overlapping or out-of-file tensors, and a `data_offsets` span that disagrees with dtype x shape are
  rejected up front. Offsets are 64-bit throughout (tested past 4 GiB with a sparse file). `SafeTensorsLoader.Load` is
  therefore stricter than before about a header that is truncated, overlaps or has a tensor outside the file; it keeps
  accepting data-less header stubs because it skips only the dtype x shape span check.
- New safetensors dtypes: `F8_E8M0`, `U16`, `U32`, `U64`, and `F8_E4M3FNUZ`/`F8_E5M2FNUZ`. The FNUZ pair parses (so an
  inventory can list it) but is refused with a clear error when a tensor is materialised, since decoding it as OCP fp8
  would halve every value.
- A split GGUF (`split.count` > 1) is detected on load and refused with a `llama-gguf-split --merge` hint instead of
  loading one part with tensors missing.

## alpha.204

- **Multi-stage pre-tokenizer and a structured conversation encoder (DeepSeek-V4.1 PR 10).** `HfTokenizerJson` read only the
  first `Split` of a `Sequence` pre-tokenizer; the new `PreTokenizerPipeline` runs every `Split` stage in order with HF
  behavior/invert semantics, then ByteLevel. Astral code points now tokenize correctly (the old single regex mangled surrogate
  pairs), and an unknown pre-tokenizer stage type or non-Regex `Split` pattern throws `NotSupportedException`
  instead of being ignored. `ChatMessage` gains blocks, tool calls, reasoning content, task and the `tool`/`latest_reminder`
  roles; `IConversationEncoder` returns `EncodedConversation` (ids, image spans, dead/vision-route masks, initial parser
  state). `DeepSeekV41Encoder` ports the upstream `encoding.py` render half; rendered text is byte-equal to the five
  upstream fixtures and ids equal HF `tokenizers`. It is not registered in the catalog yet.

## alpha.203

- **MoE routing, dispatch, combine, top-k and softplus are backend primitives** (`IBackend.MoeRoute`, `MoeBuildDispatch`,
  `MoeCombine`, `TopKLastDim`, `Softplus`). Routing covers softmax, sigmoid and sqrtsoftplus scoring with bias, per-token
  alternate bias, group-limited selection (including the HF masked-fill quirk), renormalization, scale and a logit divisor;
  ties take the lowest index. The CPU reference lives in Core and the CPU backend delegates to it. CUDA adds
  `moe_route.ptx`, `moe_dispatch.ptx` and `lm_topk_f32.ptx` (radix-select top-k up to k=2048 over vocab-sized rows), all
  deterministic with no global atomics. Vulkan reports `NotSupportedException`. `MoeFeedForward` is unchanged; the
  DeepSeek-V4.1-Flash executor adopts these in a later PR.

## alpha.202

- **The Comfy-Org HunyuanImage 2.1 repack loads and renders** (`hunyuanimage21-fp8-1_0-hunyuan-image-21-base-fp8.safetensors`,
  the `split_files/diffusion_models` file). Its ComfyUI module names (`img_attn.qkv`, `img_mod.lin`, `norm.query_norm.scale`,
  `in_layer`/`out_layer`, `model.model.` prefix) are renamed to the original-Tencent layout the converter already maps
  to diffusers. The repack also casts every bias and norm affine to raw F8_E4M3; those rank-0/1 tensors are now widened
  to F32 at conversion (`CheckpointConvertUtils.WidenFp8Vectors`). Left as fp8, the CUDA F32 `LayerNorm` read the token
  refiner's affine bytes as floats, and the output was prompt-free blobs.

## alpha.201

- Native CUDA 3-D convolution declines transient VRAM allocation failures without disabling the route for the session.

## alpha.200

- **Large convolutions run channels-last on CUDA.** cuDNN's tensor-core engines are built for NHWC: at VAE-decode sizes
  a BF16 3×3 conv ran ~40 TFLOPS over NCHW and ~155-165 over channels-last on a 4090. cuDNN convolutions whose input
  has at least 2²⁵ elements now transpose the activation and weight into channels-last scratch around the call (new
  tiled transpose kernel, `channels_last.ptx`) and fall back to NCHW when the scratch will not fit. Smaller UNet-scale
  convs stay NCHW, where the transposes cost more than they save. Kill switch: `numerics.convChannelsLast`.
- **Causal video-VAE 3-D convolutions run as one cuDNN 3-D convolution** (`IBackend.TryConv3DFrameMajor`) instead of
  one 2-D pass per temporal tap over every padded frame plus an accumulate pass. It reads the frame-major padded
  buffer `CausalConv3d` already builds. The Wan 2.2 VAE decode's convolution time at 1280×704×121 falls from ~69 s to
  ~16 s. `CausalConv3d` keeps its 5-D weight beside the per-tap slices (the op declines convs below the channels-last threshold), both built at construction.
- `CudnnConv` builds plans over any number of spatial dims.

## alpha.199

- **Ideogram 4 steps 12% faster (≈2.5 s per 1024² / 20-step generation on a 4090).** New `flash_attn_f16` kernel:
  F16 attention with F32 accumulation on the tensor cores, reading strided operands so one kernel serves head-major,
  token-major and fused-projection layouts. It serves head dim 256, where cuDNN offers a single engine at about half
  FlashAttention-2's throughput: 2.93 ms per call vs cuDNN's 4.2 ms at Ideogram 4's shape. Kill switch:
  `numerics.flashF16`.
- Ideogram 4 attention runs token-major, dropping four permutes per block. The CUDA token-major attention entry now
  also accepts the byte-identical `[1, S, heads, headDim]` layout.

## alpha.198

- **SDXL and SD1.5 render at sizes whose latent is not a multiple of 8** (1280×720, 720×1280, …) instead of returning a
  flat grey image reported as a success. The UNet's stride-2 downsample allocated `n/2` where the conv produces
  `ceil(n/2)`, and the up path doubled instead of resizing to the skip it concatenates. The concat now refuses a
  mismatched skip rather than blending misaligned memory. New `IBackend.UpsampleNearest2DToSize` (native on CUDA)
  gives nearest-neighbour upsampling to a size one short of the double, as diffusers' `upsample_size` does. The
  ControlNet condition embedding's stride-2 sizes are corrected the same way.
- The CPU `UpsampleNearest2D` kernel refuses non-F32 tensors instead of writing floats into a narrower buffer.

## alpha.196

- **Ideogram 4 no longer comes out hazy and washed out.** Its F16 blocks damp the attention-output and SwiGLU
  projections by 1/64 so they fit F16, relying on the RMSNorm that follows to cancel the factor. That cancellation
  needs the norm's eps scaled by the same factor squared. With the plain eps, the small sublayer outputs Ideogram
  produces were shrunk instead of normalized, and every F16 generation lost contrast, detail and composition
  (duplicate background figures, low-detail subjects) against ComfyUI's official template. `F16SandwichDamp` now owns
  the damp and the matching eps for both blocks that use it (Ideogram 4, Z-Image). Speed is unchanged.
- The `ideogram4` regression case (`tests/regression-cases.sh`) now uses the benchmark's structured caption instead of
  a plain-text prompt: Ideogram 4 is trained only on structured captions, and the plain prompt could not show this
  class of regression.

## alpha.195

- **Wan 2.2 TI2V-5B: warm 17.4 s → 6.6 s** through SwarmUI (512×320, 25 frames, 20 steps, 4090).
  - The Wan recipe pipeline caches the last prompt pair's umT5 embeddings, keyed on the raw text so weighted and
    plain prompts never alias; a repeat prompt skips the text encoder's upload and encode.
  - The Wan 2.2 VAE decode no longer round-trips activations through the host. The `DupUp3D` shortcut is a new
    backend op, `IBackend.DupUp3dVae` (CUDA kernel `wan_vae_dup_up3d`, F32 and bit-preserving BF16; the default is
    the host loop), the up-stages no longer clone their input on the host, and the upsample's time-conv interleave
    is a device `Permute0213`. Output frames are byte-identical.
  - `Wan22VaeDecoder` takes a compute dtype, and the Wan recipe decodes in `VaePrecisionHelper.PreferredVaeDtype`
    (BF16 on CUDA; `numerics.vaeF32` forces F32), which moves its convs onto cuDNN's tensor-core path. The attention
    block stays F32. Against the F32 decode: SSIM 0.985–0.990, PSNR 39–42 dB over five seeds, visually
    indistinguishable. Lance and Qwen-Image 2.1, which share the decoder, keep F32.

## alpha.194

- **Wan stays resident across warm generations.** The single-expert Wan DiT kept on the device (vram.keepModels)
  now tells the denoise planner it is resident, through the same `ResidentPrefixPin` LTX-2 uses. The planner's
  free-VRAM reading cannot see past the weights occupying it, so back-to-back Wan 2.2 TI2V-5B generations alternated
  between resident (≈19 s) and fully streamed (≈75 s at 512×320, 25 frames, 20 steps). A streamed denoise no longer
  reports the DiT as kept.
- A warm all-or-nothing pin is honoured only while this generation's activation reserve still fits beside it
  (`BlockStreamingScope`); a larger geometry releases the resident blocks and the planner decides afresh, instead
  of keeping them and running out of memory.
- `WanVideoPipeline.WanActivationReserveBytes` counts the patchified token grid, as its documentation said; it charged
  the latent grid, four times too many tokens at Wan's (1, 2, 2) patch. The recipe's memory estimate passes the
  checkpoint's own patch size.

## alpha.193

- **Engines sharing a GPU no longer break each other's captures.** `CudaMemory`'s synchronous copies and fills
  (`CopyHostToDevice`, `CopyDeviceToHost`, `CopyDeviceToDevice`, `Zero`, `Fill32`) run stream-ordered on the calling
  backend's compute stream and wait for it, instead of on the legacy stream. While any blocking stream in a context is
  capturing, a legacy-stream call fails (`CUDA_ERROR_STREAM_CAPTURE_IMPLICIT`) and invalidates that capture, so in
  SwarmUI an AudioLab or LLM request on the same GPU as an image generation failed and pushed the image model's
  step graph back to eager. The backend's remaining raw driver copies go through the same helpers. Code with no
  backend registered still uses the legacy stream.

## alpha.192

- **A step-graph capture that ends badly costs one eager step, not the backend.** A capture on the blocking compute
  stream is invalidated by any use of the legacy stream in the same context, including another engine's synchronous
  copy on the same GPU. Before, aborting that capture threw, left the capture flags set, and every later reset,
  generation and `FreeMemory` on that backend failed until the process restarted.
  - `CudaGraph.AbortCapture` ends an invalidated capture as expected and does nothing for a stream that has already
    left capture mode.
  - `StepGraphReset` clears the capture state before any native call and runs every cleanup step (abort, purge of
    graph-private allocations, graph reset, pool trim) even when one fails; the failures are rethrown together after.
  - `StepGraphBegin` starts tracking only once capture is open; `StepGraphEndAndLaunch` purges graph-private
    allocations when instantiation fails.
  - An out-of-memory inside a capture fails straight away instead of synchronizing (which invalidates the capture),
    so the owner falls back to an eager step where the usual recovery is legal.
- A failed capture no longer disables a model's step graph for the rest of the session: owners recapture, and only
  stop after `StepGraphFailureBudget.MaxFailures` (3) failures. A signature flip storm still disables it at once.

## alpha.191

- **A pipeline that fails to dispose no longer wedges the engine.** Model-switch eviction, `FreeMemory` and teardown
  release each cached pipeline and service independently: a failure is logged, the item is still dropped from the
  cache, and the device sweep behind it still runs. Teardown rethrows the collected failures as one
  `AggregateException`; a between-jobs `FreeMemory` only logs them. Eviction now runs under the device gate with
  construction, so it cannot sweep a sibling engine's in-flight generation on the same GPU.
- The Flux.2, Qwen-Image, Qwen-Image 2.1 and HunyuanImage recipe pipelines release every handle through one
  `CompositeDisposable` (a throw no longer strands the handles after it) and dispose once. Their loader lists take
  any `IDisposable`, so a `CheckpointSource` in them is no longer a cast away from failing.

## alpha.190

- **One resolver decides which variant a checkpoint is.** Some builds share an architecture but need different
  contracts: Qwen-Image base/Edit/Edit-Plus, Z-Image Base/Turbo, LTX-2.5 dev/distilled, Mage-Flow, Krea 2 and Lens
  Turbo, the Wan VACE/Animate/S2V/TI2V-5B task variants, and Wan-Animate-2's distillation build. Each family now
  declares these once as a `ModelVariantCatalog` instead of sniffing file names by hand. `ModelVariantResolver`
  checks, in order: the weights; the caller's hint; `modelspec.architecture` / `hartsy.model_id`; whole-token file
  names (logged as a guess); and finally the default.
  - The caller's hint is the new `ModelSpec.Variant` (SwarmUI's model class) or a `family:variant` selector such as
    `-m qwen-image:edit`.
  - The result feeds construction (`RecipeContext.Variant`), the pipeline cache key, `SupportsFor` / `DefaultsFor` /
    `InputLimitsFor`, and the new public `ModelCapabilities` queries hosts call.
  - `CheckpointProbe` replaces the video-only header peeks and adds tensor shapes.
- **Qwen-Image knows base from Edit.** The base no longer declares `RefEdit`, refuses reference images by name, and
  skips loading the vision tower. On an Edit build, Auto init-image mode edits rather than denoises. Edit v1 gets
  ComfyUI's `TextEncodeQwenImageEdit` template (one unlabelled reference, ~1 MP vision copy) instead of the Plus
  `Picture N:` form, and the input limits follow the variant (1 for v1, 3 for Plus). Repacks are stamped
  `qwen-image-edit` / `qwen-image-edit-plus`.
- File-name detection is whole-token now, so `credit` no longer reads as `edit`. Wan-Animate-2 distillation takes its
  10-step / CFG 1.0 defaults when a hint or metadata, not only a file name, identifies it.
- Removed: `LtxVideo2DistilledRouting`, `InferenceEngine.ResolveVideoFamilyId` / `VideoDefaultsFor` /
  `SupportedVideoFeatures` / `SamplingSupportForVideo` (use `ModelCapabilities`), `WanVideoRecipe.SupportsFor(path)`
  and its siblings (use the variant overloads), `WanAnimate2Transformer.ResolveLogScale`, and
  `ZImageCheckpointConverter.DetectVariantFromFileName` / `CheckpointVariant`.

## alpha.189

- **Image families declare how many input images they read, and the gate refuses past it.** A fourth reference on
  Qwen-Image-Edit used to be dropped with a log warning, and a reference image on Boogu, OmniGen2 or Mage-Flow was
  never read at all — with no init image the result was plain text-to-image. Each recipe now exposes
  `IArchitectureRecipe.InputLimits` (`ImageInputLimits { MaxImages, ReferencesRequireInitImage }`): Qwen-Image 3
  (`QwenImageEditConditioning.MaxReferences`, references usable without an init image), Boogu / OmniGen2 /
  Mage-Flow 1 with the init image required, every other init-image family 1 by default. `ImagesService` throws
  `NotSupportedException` after the feature check with `Model family '<id>' takes at most N input images; M were
  supplied.` or `Model family '<id>' needs an init image to edit; reference images alone are not used.`

## alpha.188

- **Wan claims an end frame only where one has been checked.** `WanVideoRecipe.Supports` declared
  `VideoFeatures.EndFrame` for the `wan-21-14b` compat class and the generic `wan` slug, though only the Wan2.2
  TI2V-5B has been run with one (`WanEndFrameRealWeightTests`). Both now declare the init image only. Because that
  real-weight run goes through the generic slug, `SupportsFor` adds the end frame back when the file's plain
  `patch_embedding.weight` takes 48 latent channels — the Wan2.2 VAE width only the 5B uses — read from the header
  with no weight I/O. A folder or an unrecognized layout stays init-image only.

## alpha.187

- **Vulkan's fp8 Linear is on wherever the device offers fp8 cooperative matrices**, validated on an RTX 4090 under
  driver 595 (E4M3 16x16x32). Unset `numerics.vkFp8` now follows the device, like `fp8Native`; the `reference` profile
  still pins it off. Cards without the extension (Ampere and older, or a pre-595 NVIDIA driver) keep the F16-cast path.
- **A cooperative-matrix-2 fp8 GEMM.** Where `VK_NV_cooperative_matrix2` lists an E4M3 configuration (RTX 4090 under
  595), the fp8 Linear runs `matmul_fp8_coopmat2`: `matmul_coopmat2`'s structure with E4M3 operands read from 8-bit
  storage and an aligned fast path. Including the activation quantization, it beats the F16 GEMM at every DiT shape
  measured on the 4090 (4096×3072×12288: 1.91 ms vs 2.69; 4096×12288×3072: 1.97 vs 2.46). The coopmat1 fp8 kernel,
  about 20 TFLOPS, remains the fallback for devices without it.
- **The fp8 GEMM now matches CUDA's accuracy.** Ada's fp8 tensor cores accumulate below F32 — 7.6e-4 of the output
  range at K=512 — so `matmul_fp8_coopmat` adds a partial into a true F32 accumulator every 128 of K. On the same
  operands that gives 1.731e-4, cuBLASLt's native fp8 error to the digit.
- A 595-or-newer NVIDIA driver without `VK_EXT_shader_float8` is now reported as the card lacking fp8 tensor cores,
  not as an old driver.

## alpha.186

- **Vulkan attention on cooperative-matrix-2.** `sdpa_flash_cm2` is a query-tiled flash attention on
  `VK_NV_cooperative_matrix2` (64 query rows × 64 keys, head dim 64 or 128, optional additive mask). It addresses
  Q/K/V by stride, so token-major layouts need no permute and grouped-query K/V need no head repeat. `IBackend`
  gains a grouped-query token-major overload, which Krea2 uses when the backend serves it. A shape the kernel does
  not serve falls back to the existing head-major path. On Krea2 at 1024² attention went from 167.6 s to 1.3 s of
  GPU time per generation.
- **The Vulkan memory pool reuses frees still pending on the GPU timeline.** A request that misses the pool first
  takes back what the GPU has finished with, then waits for the oldest pending free of the same memory type when
  those could cover it, and only then asks the driver. Before, per-Linear weight casts were never reclaimed between
  submits, so every cast became a new `vkAllocateMemory` until VRAM ran out. Each out-of-memory retry then drained
  the queue to idle and destroyed the pooled blocks. On Krea2 that was 2,423 driver allocations (17.7 s) and 28 full
  drains per generation; now it is 236, almost all of them weight loading. A free made while nothing is recording
  is released without waiting on a tick no submit will signal (that hung Boogu mid-denoise).
- **coopmat2 GEMM tiles.** Large products run on 128×256 tiles. Interior tiles with 8-element-aligned strides load
  unclamped, unrolled eight blocks deep, and workgroups walk eight tile rows per column band so B stays in L2. On
  Krea2 shapes that is 125–140 TFLOPS, against cuBLAS's 155–167 with F32 accumulation.
- **SDXL on Vulkan now matches CUDA** (SSIM 0.999 at 1024², was 0.57). Vulkan's `BroadcastAdd` read row 0 of a
  `[B, C]` bias for every batch item, so the conditional half of each CFG batch got the unconditional half's
  ADM-conditioned time embedding. It now reads each item's own row, and a `[C]` bias is still shared.
- **Flux.2 generations no longer fail after the image is made.** `Flux2RecipePipeline.Dispose` iterated its
  `IDisposable` loaders as `SafeTensorsLoader`, which threw `InvalidCastException` at teardown once the text encoder
  opened through `CheckpointSource`.
- **Vulkan caches a weight cast only while a fifth of the heap stays free**; past that the cast is made per call and
  freed. fp8, GGUF and bf16 checkpoints whose F16 casts did not fit beside their own weights ran out of memory
  mid-denoise on a 24 GB card. Flux.2 Q4_K_S (SSIM 0.998 against CUDA F16), ERNIE-Image Turbo (0.95), Boogu, Chroma
  and Lens now run; Chroma, Lens and Boogu produce wrong images on Vulkan, which the out-of-memory failure hid.
- `diagnostics.vkProfileGpu` times every dispatch with timestamp queries and prints GPU time per op and per kernel,
  plus blocking host waits by call chain. The host-wall profile charged a queue stall to whichever op was waiting.
- Krea2 Turbo at 1024², 8 steps, RTX 4090: 1.06 s/step on Vulkan, against 0.83 s on CUDA with
  `numerics.fp8Native=false` and 0.53 s on CUDA's fp8 tensor cores. The Vulkan image matches CUDA F16 at SSIM 0.998.

## alpha.185

- **Vulkan fp8 activation scale was an ulp off on NVIDIA.** `divRn` corrected the quotient with `fma()`, which
  Vulkan may run as a separate multiply and add, so the residual was not exact and the scale (and every E4M3 byte
  scaled by it) could differ from CUDA's `div.rn`. The residual is now Dekker's exact product from correctly rounded
  multiplies and adds, and the nearest of the quotient's neighbours is kept, rounded once even when the scale is subnormal.

## alpha.184

- **Native FP4 never ran on a real checkpoint, and nothing said so.** `numerics.fp4Native` on or off produced a
  byte-identical image from Z-Image Turbo's nvfp4 DiT on a Blackwell card, because `ZImageRecipe` opened its
  checkpoint with no options: all 180 nvfp4 groups unpacked to F16 at load, so the dispatch gate never saw a
  block scale whatever the knob said. `CheckpointOpenOptions.ForNativeNvfp4Gemm` asks the backend first and
  returns plain defaults when it cannot run the native GEMM, so nothing below Blackwell changes.
- `numerics.fp4Native` is now `Construction` scope. It was declared `Runtime`, but the capability it feeds decides
  **at checkpoint open** whether weights stay packed, so a per-request value could never reach the path. Nothing
  in this repo, SwarmUI or the API client sends it per request.
- **Three one-shot diagnostics**, because the question "did the native path run" had no answer in any log: the
  checkpoint open counts nvfp4 groups by outcome (resident / companion / fp8 / F16), the backend constructor
  lists every static condition refusing the native GEMM rather than the first, and the first block-scaled Linear
  latches whether it engaged or the specific gate condition that refused it. No per-layer logging.

## alpha.183

- **A converted audio checkpoint loads under its Hartsy name.** A converted file names, in `hartsy.stands_in_for`,
  the upstream paths it replaces. `AudioStandIns` hard-links it into each missing one on first use, so every loader
  finds it where it already looks: the HF cache, `ModelDownloader` assets and fixed-path loaders alike. An existing
  file is never replaced. Links are recorded in `.hartsy-standins.json`: one whose artifact is replaced is relinked,
  one whose artifact is removed is deleted, and a link replaced by a real file is left alone. Checked end to end in
  SwarmUI with all 253 upstream weight files deleted: 154 generations, 142 pass, 10 with quality notes.
- **Converting a checkpoint could silently ship a broken model.** `tools/CheckpointRepacker`:
  - It kept only the first state dict of a nested checkpoint (Kokoro converted as 25 of its 548 tensors, exit 0). It
    now refuses a conversion that would drop tensors, and lists what it would lose.
  - It ignored mistyped options; they are now rejected with a suggestion.
  - It wrote no identity unless it was hand-typed. Identity now comes from `ModelIdentityCatalog`
    (`--model/--variant/--component`).
  - Restamping a safetensors keeps its tensors byte for byte, recognizes the file by content (even under a `.pt`
    name) and drops identity keys the new stamp omits.
  - A folder output is named the way Hartsy stores files: `<model>[-<variant>][-<part>]_<precision>`.
- **The repacker reproduces third-party repacks from official releases:**
  - it merges shards and casts with PyTorch's round-to-nearest-even (`SafeTensorsMerger`);
  - recipes (`--recipe`) cover key maps, fusing, copies, drops, squeezes and embedded tokenizers
    (`TiktokenConverter`);
  - LoRA baking (`LoraBaker`, `--lora`, the recipe `lora` step) matches single-threaded torch bit for bit;
  - `--stands-in-for` stamps the paths a file replaces.

  ACE-Step XL, Orpheus, CSM, YuE2, the ACE-Step VAE, Stable Audio Open Small and SheetSage2 are all rebuilt from
  official releases, tensor-identical to the repacks they replace.
- **Every audio loader reads a converted checkpoint.** `AnyFormatCheckpointLoader` recognizes safetensors by content,
  replacing `PytorchPickleLoader` wherever an audio family opened a pickle directly.
- **AudioLab can admit a converted file.** Artifacts carry `hartsy.provider_id` and `hartsy.model_id`, and variants
  stamp their own class. Licenses were corrected from the model cards: ACE-Step `mit`, YuE2 `cc-by-nc-4.0`,
  Fish-Speech `cc-by-nc-sa-4.0`, NeuTTS `apache-2.0`.
- **Demucs htdemucs_6s never loaded.** It has no channel projection around its transformer (`bottom_channels=0`),
  and the engine required one.
- **ACE-Step selected by path used the wrong config.** This covers SwarmUI's core list and any renamed file: XL
  failed to load, and base and sft silently ran as turbo. It also missed its silence latent. The variant now comes
  from the checkpoint's `hartsy.model_id` when the given one is unknown, and the config and latent from the catalog.
- **SheetSage2 no longer discards a whole score over one chord shorter than a subbeat.**
- **Resemble-Enhance has a denoise-only mode** (`FxEnhanceRequest.DenoiseOnly`, upstream's `denoise()`). On a real
  recording it takes about 2 s, against 4.5 min for the enhancer.
- `SafeTensorsWriter.Save` writes a tensor over 2 GiB; the PocketTTS revision pin is applied (it was passed as the
  cache category).

## alpha.182

- **Vulkan can run an fp8 Linear on fp8 cooperative matrices, opt in.** Where the driver offers `VK_EXT_shader_float8`
  with an E4M3 × E4M3 → F32 cooperative-matrix shape, `numerics.vkFp8=true` runs CUDA's native fp8 scheme: the weight
  stays packed with its per-tensor scale in alpha, and the activation is quantized per tensor to E4M3 — the checkpoint's
  `.input_scale` under `numerics.fp8StaticInputScale`, else absmax/448 on the device. The quantizer writes the same
  bytes as CUDA's `fp8_quant`; GLSL's `/` is not correctly rounded, so the scale and its reciprocal take one FMA
  correction to match `div.rn`. **Off by default:** the GEMM has not yet run on a card. NVIDIA's 580 driver lacks the
  extension; 595 has it.
- A Vulkan backend logs once per device whether fp8 Linear runs and why not (the driver lacks the extension, the card
  has no fp8 tensor cores, or the knob), and warns when `numerics.vkFp8=true` cannot be honored.

## alpha.181

- **All 27 ComfyUI k-samplers.** The 18 that were listed as not implemented now run: `ipndm`, `ipndm_v`, `deis`,
  `res_multistep`, `gradient_estimation`, `ddpm`, `heunpp2`, `dpmpp_sde`, `dpmpp_3m_sde`, `er_sde`, `seeds_2`,
  `seeds_3`, `sa_solver`, `uni_pc`, `uni_pc_bh2`, `euler_cfg_pp`, `dpm_fast` and `dpm_adaptive`. Each matches ComfyUI
  0.37's `sample_*` per step within 5e-5 on eps, v-prediction and flow models, including img2img from a truncated
  schedule (`SamplerParityTests`, fixtures from `tests/python-reference/dump_k_samplers.py`). The `_gpu` names resolve
  as aliases. `euler_cfg_pp` refuses a model whose pipeline does not produce a separate unconditional prediction.
- **Flow models get ComfyUI's flow math.** `euler_ancestral`, `dpm_2_ancestral` and `dpmpp_2s_ancestral` use the
  rectified-flow ancestral step, `dpmpp_2m_sde` is CONST-aware, and the SDE samplers draw Brownian-bridge noise so
  their overlapping draws have the right statistics. `lms` no longer indexes past its history in img2img.
- **SDXL and SD1.5 fed timestep 0 to any sampler that evaluates between schedule points.** `SigmaToTimestep` searched
  the ascending training sigmas as if they descended, so every off-schedule sigma mapped to timestep 0: `heun`,
  `dpm_2`, `dpmpp_2s_ancestral` and the new multi-stage samplers produced noise. It now interpolates in log sigma.
  Karras schedules also took their endpoints from the wrong ends of the table.
- The SDXL CFG-parallel loop used to replace any non-default sampler or schedule with Euler without saying so. It now
  uses the sequential loop for them.

## alpha.180

- **The Vulkan suite runs against a non-NVIDIA driver.** Mesa's software ICD reports subgroup size 8 and no
  cooperative matrix — the two regimes NVIDIA hardware never reaches — and 220 of 221 GPU-integration tests pass
  there. `VulkanPipelineCache` now reports `InitialDataBytes`, the on-disk bytes it was handed at construction,
  because the pipeline-cache test asserted the reload from the file the *second* backend wrote: a driver that
  persists no pipelines writes a fresh 32-byte header either way, so that assertion was green without any reload.
- The rental script gains an opt-in `swarm` stage: SwarmUI loads the extension in its own load context against
  the pinned NuGet engine, so the engine generating from the command line never proved the extension does.

## alpha.178

- **A missing side model now downloads instead of failing the generation.** Every recipe resolves its text
  encoder, VAE and CLIP through `ModelDownloader.EnsureSideModelAsync`, whose three-argument overload was strict:
  an absent file threw before any network call and told the operator to fetch it by hand. Found on a rented
  Blackwell card, where Z-Image Turbo refused to generate through SwarmUI because `VAE/Flux/ae.safetensors` was
  not on disk — with the repo, the path and the hash all sitting in the catalog entry. The overload now follows
  `paths.sideModelAutofetch`, new and **on by default**, which is what SwarmUI's own ComfyUI backend does; LTX-2.5
  already opted in per-call and can stop special-casing it. Turning the setting off restores the old behavior for
  an air-gapped install, and the error then names the setting and the repo rather than a bare path. A caller that
  must never reach the network still passes `downloadIfMissing: false` explicitly.
- **Z-Image Turbo LoRAs load.** `hartsy image -m zimage --lora …` failed with "Could not detect LoRA format":
  the format detector has arms for Flux, Wan, SDXL and SD1.5 prefixes and none for Z-Image's
  `diffusion_model.{layers,context_refiner,noise_refiner}.`, so every Z-Image adapter was rejected outright.
  Comfy-Org's own `z_image_turbo_distill_patch_lora_bf16` is one. A new `ZImageLoraMapper` plus a detector arm
  covers them; the `.lora_A.default.weight` suffix vocabulary already worked, so nothing there changed. Q/K/V
  arrive split and the checkpoint stores them fused, which the existing `FusedProjectionLayouts` row resolves at
  merge time — all 238 of that file's mapped targets hit the real checkpoint, 136 directly and 102 as fused
  slices, none missing.

## alpha.177

- **A missing side model now downloads instead of failing the generation.** Every recipe resolves its text
  encoder, VAE and CLIP through `ModelDownloader.EnsureSideModelAsync`, whose three-argument overload was strict:
  an absent file threw before any network call and told the operator to fetch it by hand. Found on a rented
  Blackwell card, where Z-Image Turbo refused to generate through SwarmUI because `VAE/Flux/ae.safetensors` was
  not on disk — with the repo, the path and the hash all sitting in the catalog entry. The overload now follows
  `paths.sideModelAutofetch`, new and **on by default**, which is what SwarmUI's own ComfyUI backend does; LTX-2.5
  already opted in per-call and can stop special-casing it. Turning the setting off restores the old behavior for
  an air-gapped install, and the error then names the setting and the repo rather than a bare path. A caller that
  must never reach the network still passes `downloadIfMissing: false` explicitly.

## alpha.175

- **The Blackwell kernel never compiled, and it took the whole CUDA backend down with it.** `block_quant.sm120.ptx`,
  shipped since alpha.166, contained `cvt.rn.satfinite.e2m1x2.f32` with a 16-bit destination; that instruction packs
  two nibbles into one byte and takes a `.b8`, so ptxas refused the module. `CudaKernels` loads it on any
  compute-capability 12.0 card, so construction threw and **every** CUDA operation failed on an RTX 5090 or RTX PRO
  6000 — not just the FP4 path. Measured on a PRO 6000: 165 of 178 GPU tests failed as shipped, 178 of 178 pass with
  the corrected kernel. The inline asm now routes through an explicit `.b8` temporary, the shape NVIDIA's own
  `cuda_fp4` header uses.
- **The build now assembles what it emits.** nvrtc and `nvcc -ptx` both stop at PTX, so an instruction whose operands
  are wrong for the target survives to the card and fails at module load. `build_common.sh` runs `ptxas` against each
  emitted file wherever a toolkit is present. That is the check that would have caught this at the commit that
  introduced it.
- **The native block-scaled GEMM is correct, and the bring-up gate was measuring the wrong thing.** Its NVFP4 case
  compared a 4-bit-activation product against a 16-bit-activation one, so its error was dominated by quantization
  loss — 10.2% on hardware, against a budget of 8% written without a card. `Nvfp4GemmReferenceTests` feeds the
  reference the same activation the native path quantized and bounds the kernel itself at 0.36%; the older test keeps
  its end-to-end comparison with a budget set from that measurement. First hardware numbers, RTX PRO 6000, DiT shape
  4096×3072×3072: native 0.184 ms/Linear against 0.404 for the unpack path, 2.19× faster.

## alpha.174

- **Vulkan's INT8 Linear runs on the device.** With `numerics.vkInt8=true` the weight is quantized per row by
  `quant_int8_rowwise` once and cached under `I8` beside its other casts — the scales packed at the buffer's tail
  behind push-constant offsets, so it is one buffer per weight and dtype and is freed with the weight — the activation
  is quantized by the same shader per call, and the bias goes through `broadcast_add`. Before, both quantizations and
  the bias add were host loops and the weight was re-quantized on every call. Still off by default.
- The Vulkan tests take `HARTSY_TEST_VULKAN_DEVICE` (a test-harness switch, not an engine knob) to run on a second card.

## alpha.173

- **Vulkan computes a 16-bit-weight GEMM in F16.** `ResolveGemmDtype` takes both operands and the output, the CUDA
  backend's rule: an fp8 or GGUF operand computes in F16 (what it unpacks to), a BF16/F16 operand makes the product
  F16 with the F32 side cast to it (the weight's cast cached, the activation's transient), F32 × F32 stays F32. Before,
  the output's dtype decided alone, so an F32-output Linear over F16 weights ran the scalar F32 kernel with the weights
  widened. An F16 product written to an F32 output goes through the cooperative-matrix kernels' F32 store, or the tiled
  kernel's transient plus one cast where the shape admits no cooperative-matrix kernel; `Conv2D` computes and adds its
  bias in the product's dtype and casts once. `numerics.vkF16Gemm=false` restores the output-dtype rule, and the
  reference profile pins it off.

## alpha.172

- **Every Vulkan GEMM goes through one dispatcher.** `DispatchGemm` takes a `GemmOperands` record (buffer handles,
  element offsets, transposes, leading dimensions, alpha/beta, a bias in either form) and picks coopmat2, then
  coopmat, then the tiled kernel by what the operands admit — the three shaders already shared one push-constant
  layout. `Linear`, `MatMul`, `BatchedMatMul`'s per-slice loop, the naive attention's score and value products and
  `Conv2D`'s im2col GEMM all call it, so a batched or convolution product on F16 reaches the cooperative-matrix
  kernels it used to bypass (the column tile of a convolution is now 16-aligned where it can be). GEMMs whose dtype
  resolves to F32 run the same tiled kernel as before; the compute-dtype policy is a separate change.
- `matmul_coopmat_blocked` — a diagnostic shader on no production path — is deleted with its benchmark.

## alpha.171

- The IQ1_S and IQ1_M dequant kernels drop the word and sign helpers they never called (regenerated PTX, same
  code path), and the GGUF format table gives IQ1_M its 56-byte block and IQ1_S / IQ2_S their exact bits per
  weight — the two follow-ups from the alpha.170 review that a mid-rebase push left behind.

## alpha.170

- **The rest of llama.cpp's i-quant family loads and stays packed on CUDA.** IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
  IQ3_S, IQ1_S and IQ1_M each gain a host codec and a CUDA dequant kernel; `IqTables` / `iq_tables.cuh` carry ggml's
  codebook grids, sign patterns and mask verbatim (a kernel includes only the tables it indexes). The three formats
  with a small published file — IQ3_S (Llama-3.2-1B IQ3_M, Qwen2.5-1.5B IQ3_XS), IQ3_XXS (Qwen2.5-1.5B IQ3_XS) and
  IQ2_S (Qwen2.5-1.5B IQ2_M) — are settled against their Q8_0 copies in `GgufRealFileCorrelationTests`; IQ2_XS,
  IQ2_XXS, IQ1_S and IQ1_M rest on hand-built known-block tests, one field flipped at a time, until a small file
  exists. Below decode they take the same dequantize-then-GEMM route as IQ4_XS; none has a fused GEMV yet.

## alpha.169

- **Q3_K weights were decoded wrong everywhere.** The host codec unpacked the sixteen 6-bit scales in Q4_K's
  interleaved order, and the CUDA dequant kernel had been written to match it; against the same tensors in a Q8_0
  file the decoded Q3_K weights correlated at 0.21. All four decoders (host, CUDA dequant, the new CUDA GEMV, the
  new Vulkan shader) now read ggml's `dequantize_row_q3_K` layout — low nibbles of bytes 0..7 are entries 0..7, high
  nibbles entries 8..15, byte 8 + s % 4 carries entry s's high bits — and correlate at 0.99. Any Q3_K_M GGUF, and
  the Q3_K tensors inside every Q2_K file, generated garbage before this.
- **Real weights settle a codec, not synthetic blocks.** `GgufRealFileCorrelationTests` dequantizes every tensor a
  lower-bit Llama-3.2-1B file stores in the format under test and correlates it with the Q8_0 copy; that is what
  caught the Q3_K bug that the block-vs-block GPU tests could not (both sides shared the mistake). It runs when the
  files are staged.
- **IQ4_XS loads and stays packed on both GPU backends, IQ4_NL on CUDA.** `Codec_IQ4_XS` (host), `dequant_iq4_xs_to_f16`
  and `dequant_iq4_nl_to_f16` (CUDA) and `dequant_iq4_xs` (Vulkan). The CUDA backend's GGUF dequant is one table —
  a dtype maps to its kernel and launch width, `SupportsResidentQuant` reads the keys — so a new format is a kernel
  and one line, not an if-chain entry in three places.
- **Q2_K and Q3_K decode with fused GEMVs on both tiers.** `mul_mat_vec_q2k_q8_1` / `q3k_q8_1` are the int8
  dp4a tier (Q2_K's per-run min rides on `dp4a(0x01010101, xq)`, Q3_K's signed 3-bit value on `__vsubss4`, each
  with the k-split entry the other K-quants have) and `mul_mat_vec_q2k_f32` / `q3k_f32` the float tier; both
  dispatch at M ≤ 8 like Q4_K, where these formats used to fall to the dequantize-then-cuBLAS route. Vulkan
  gained `dequant_q2_k` and `dequant_q3_k`, so a Q2_K or Q3_K weight is resident there too.
- `tests/regression-cases.sh` carries the three new-format text cases under the `quant` tag; they run head-only
  (`--no-base`) because no prior build could load them.

## alpha.168

- **`tests/blackwell-run.sh` is the rented-GPU session, scripted.** Preflight (driver ≥ 580 — every nvcc-built PTX
  here is ISA 9.0 — the right card, cuBLAS resolvable), bootstrap (SDK, build, settings), probe (the shipped PTX
  JITs or the run stops there), GPU tests with a skipped test counted as a failure, the regression gate head-only
  and native-block-scaled off vs on, within-session determinism, an optional Vulkan pass capped at five minutes,
  and a bundle plus summary; stage markers make a rerun repeat only what did not finish, `--budget-minutes` stops
  the clock, `--auto-stop` stops a RunPod pod. `--rehearsal` runs the same script on the local card with the
  Blackwell-only rows allowed to skip, so the pod run differs only in hardware. `benchmarks/CLOUD_GPU_RUNBOOK.md`
  points at it. The GPU test stages run pinned to the session card by UUID and with the models root exported, so
  the real-weight tests find their assets instead of failing under the repo.
- `tests/regression-ab.sh` labels a knob arm by a hash of its knobs, so off-vs-on arms of one commit never share a
  run directory, and `--gpu-name ""` with `--gpu N` names the card directly.

## alpha.167

- **MXFP8 weights stay packed the way nvfp4 ones do.** `Mxfp8Codec.TryAttachResident` hangs a weight's UE8M0 block
  scales on `QuantInfo` (`Format = "mxfp8"`) instead of unpacking it at load, and the CUDA backend's resident path
  is now written for any block-scaled weight: one scale upload, one eligibility check, one unpack substitution
  (`dequant_mxfp8_to_f16`, the mxfp8 twin of the nvfp4 kernel) and the same native branch — the
  `BlockScaledGemmExecutor` multiplies MXFP8 operands with `VEC32_UE8M0` scales, and `block_quant` gained the
  MXFP8 activation quantizer (an OCP MX shared exponent per 32 elements). The weight stays packed only where that
  native GEMM will consume it — Blackwell with `numerics.fp4Native` on; anywhere else it widens on the host to the
  same BF16 as before, so a card below Blackwell generates the same bytes at the same speed. `Mxfp8ResidentCodec`
  in Core is that host decode and what the tests measure against.
- **Residency is answered per weight, not per dtype.** `IBackend.SupportsResidentQuant(Tensor)` defaults to the
  dtype answer; CUDA says yes to an mxfp8 weight only where the native GEMM runs. `QuantizedWeightPolicy`'s
  predicate takes the weight, and `Widen` gained the two arms it lacked — nvfp4 and mxfp8 — so a CPU or Vulkan
  shard receiving either widens on the host instead of throwing. The Lens pipeline runs the policy like every other
  recipe, before its LoRA hook: a merge onto a packed weight rides as a runtime adjunct on that tensor, which a
  later widening would have left behind, and `LoraStack` now treats any block-scaled weight that way regardless of
  its dtype (an mxfp8 target used to be requantized as per-tensor fp8, dropping its block scales).
- The native fp8 dispatch gate refuses a weight carrying block scales, so an mxfp8 weight can never be multiplied
  as per-tensor fp8. A block-scaled weight now splits by whole 128-row tiles of its swizzled scales — the fused QKV
  weights of the Lens DiT split into Q, K and V with their own scales instead of refusing. MXFP4 (GPT-OSS's ggml blocks) is not repacked in this release: its experts go through the MoE
  slice path, not `Linear`, and gain nothing here until that path is resident.
- An `int8_tensorwise` weight without a ConvRot rotation is eligible for the resident int8 path again: the
  shared-memory ceiling introduced in alpha.163 divided by the rotation group, and a group of zero threw before the
  check could say no.

## alpha.166

- **A kernel can ship a per-architecture PTX beside its baseline.** `CudaKernels.PtxPath` loads
  `<kernel>.sm<CC>.ptx` for the device's exact compute capability when one exists and `<kernel>.ptx` otherwise;
  every module path in the kernel set, the VSA probe and the tensor-core GEMM go through it, and the backend logs
  which variants it picked. Exact match only — PTX built for a family-specific arch (`sm_120a`) does not JIT
  anywhere else, so the baseline serves every other card unchanged. The first and only variant is
  `block_quant.sm120.ptx`, whose e2m1 packing uses the hardware `cvt` on consumer Blackwell; the `Ptx\*.ptx`
  packaging glob already carries it.
- **Nine `build.sh` scripts are now one body and nine kernel lists.** `Kernels/build_common.sh` holds the
  toolchain resolution, compilation, the PTX ISA 9.0 check and the install step; each domain script declares its
  lists and calls `build_all`. `--arch sm_120a` builds a domain's `ARCH_VARIANTS` as suffixed variants;
  `--install-tuned` keeps its meaning for `lm`. Rebuilding every domain through the shared body reproduces 52 of
  55 shipped artifacts byte for byte; the three that differ — `lm_f32`, `mul_mat_vec_q6k_q8_1`, `h3_vsa` — are the
  known nvcc-built kernels whose sources match their PTX's commit, and they are left as shipped.
- `CudaKernels` takes the `CudaContext` it will run under instead of two shared-memory numbers; `TensorCoreGemm`
  takes the compute capability rather than its major digit.

## alpha.165

- **Native block-scaled GEMM is wired, behind `numerics.fp4Native`.** `Fp4GemmExecutor` is now
  `BlockScaledGemmExecutor`, one executor for NVFP4, MXFP4 and MXFP8 driven by a `BlockScaleFormat` descriptor
  (group size, operand type, cuBLASLt scale mode, checkpoint format string) that the quantizer, the dispatch gate
  and the codecs all read. On Blackwell with the knob on, a resident nvfp4 weight and its checkpoint scale tensor
  are the GEMM operands as stored; the activation is block-quantized on the stream by the new `block_quant`
  kernel — e2m1 nibbles, E4M3 scales written straight into cuBLASLt's blocked layout, and the per-tensor scalars
  left in device memory where the GEMM reads alpha (pointer mode DEVICE), so dynamic quantization costs no host
  sync. Below Blackwell nothing changes: the knob is off by default, and stays off until a card has run it.
- **The nvfp4 fold now follows the backend.** `Flux2Recipe` opened its encoder with `Nvfp4ToFp8` unconditionally,
  which would have made the native path unreachable on exactly the hardware it targets.
  `CheckpointOpenOptions.ForNvfp4Consumer` keeps the groups packed where `BackendCapabilities.NativeBlockScaledGemm`
  says they multiply natively and folds to fp8 everywhere else, which is today's behaviour on every card here.
- **The device-side block-scale codecs live in one header.** `swizzled_scale_index`, the e4m3/e2m1 decoders, the
  e4m3 encoder and the F16 bit reader moved from `dequant_nvfp4_to_f16.cu` and `fp8_quant.cu` into
  `block_scale.cuh`, joined by the e2m1 encoder; both kernels' PTX reproduces byte for byte. The e2m1 packer uses
  the hardware `cvt.rn.satfinite.e2m1x2.f32` under the sm_100a/sm_120a family-feature macros and bit math
  elsewhere, with the same rounding.
- The bias epilogue the native fp8 branch applied by hand is one helper both native branches call; it now
  applies a row range after the dtype cast rather than before, which the cast used to discard.

## alpha.164

- **Vulkan now enables the shader features its own shaders declare.** `im2col` requires 64-bit integer arithmetic
  and the two BF16 casts require 16-bit, but the 1.0 feature block inside `VkPhysicalDeviceFeatures2` was never
  filled at device creation, so `shaderInt64` and `shaderInt16` shipped disabled and the pipelines worked only
  because the NVIDIA driver does not check; a conformant driver may reject them. Both are queried, reported on
  `VulkanCapabilities` and enabled, and the kernel registry refuses a shader whose feature the device lacks by
  naming the feature, before pipeline creation. A lint test reads every shader's `#extension … : require` lines
  and holds the registry's map to them.
- **A descriptor pool is no longer reset while the GPU may still be reading it.** The pool ring flipped on
  exhaustion with a bare `vkResetDescriptorPool`, from inside a dispatch, with no check that the submissions
  binding that pool's sets had completed. A pool now retires at the tick the next submit signals; the flip back
  to it submits that recording if it is still open and waits for the tick before the reset. The set is also
  allocated before the command buffer is touched, so that submit cannot split a dispatch across two buffers.
- `DtypeSuffix` throws for any dtype other than F16/F32 instead of silently choosing the F32 shader.
- `tests/regression-ab.sh` runs a backend other than CUDA only over the cases tagged with its name (Vulkan takes
  `sd15` and `krea2`), and a case that crashes on both arms every seed is reported as pre-existing rather than
  failed; a head crash with a running base still fails.

## alpha.163

- **One home each for the CUDA arch gate, the cuBLASLt executor boilerplate and the dtype map.** `CudaArch` turns
  a compute capability into one comparable number with the tiers the engine gates on; the FP8 executor, the FP4
  executor and the cuBLAS-version warning had each spelled their own threshold, and the warning's (`major >= 12`)
  skipped SM 10.x datacenter Blackwell entirely. `CublasLtExecutorBase` owns the handle, workspace, TN
  descriptor/layout creation and teardown that `Fp8GemmExecutor`, `Fp4GemmExecutor` and `Int8GemmExecutor` had
  each carried a copy of. `CublasApi.DataTypeOf` is the single `DType → cudaDataType` map every layout is created
  through — which is the F8E5M2 fix: the fp8 executor hard-coded E4M3 for both operands and would have multiplied an
  E5M2 weight as E4M3. The one pairing cuBLASLt cannot run, E5M2 × E5M2, now takes the cast path instead.
- **Shared-memory limits come from the device.** `CudaContext` reports the per-block default and the opt-in
  ceiling; the int8 mma GEMM is left unbound, with a warning, on a device whose ceiling is below its tiles rather
  than failing at launch, and the ConvRot group-size refusal is the rotate kernel's own shared footprint against
  the default instead of a literal.
- **`tests/regression-ab.sh` is the regression gate every PR runs.** Both arms are built fresh, PTX and SPIR-V
  included; generations alternate seed by seed; per-step ms, wall and peak VRAM are medians over the warm seeds;
  quality is SSIM plus a raw-pixel digest per seed pair (the token stream for text), with a `before | after | diff`
  montage of each; the verdict is against a declared expectation, `identical` or `bounded:<ssim>`, and a step time
  past the speed tolerance fails the case either way. The case list lives in `tests/regression-cases.sh`, shared
  with `migration-baseline.sh`.

## alpha.162

- **Hosts can now ask whether a generation fits a GPU before sending it there.** New
  `IInferenceEngine.MemoryEstimation` (`IMemoryEstimationService`): `EstimateAsync` returns the per-phase VRAM a
  model needs at a geometry (text encoder, denoiser, VAE, each with its weights and working memory), and
  `AssessAsync` judges that against the engine's own device as `Resident`, `Streamed`, `Infeasible` or `Unknown`.
  Nothing is loaded: weights are sized from checkpoint headers (quantized weights the backend cannot hold packed
  are widened, as `QuantizedWeightPolicy` does at load), read once per checkpoint per process and cached, so every
  later answer is arithmetic. This is what lets the SwarmUI extension route a large video model to the large card
  instead of the one that was idle longest.
- **The verdict uses the VRAM policy the generation will actually run with.** The backend's policy with the
  request's `VramOverrides` applied (`VramPolicyRegistry.Resolve`), so Performance never streams and keeps every
  phase resident, and a per-request tier override changes the answer exactly as it changes the generation.
  Only memory levers the engine acts on count: block streaming when the recipe wires it and the backend has a
  streaming cache, phase unload unless pinned off, DiT sharding and component placement when the recipe wires them.
  Capacity is total VRAM less the placement planner's per-device reserve, never free VRAM, so identical requests
  get identical answers.
- **Recipes can describe their own activations.** New default-null `DescribeMemory(CheckpointHeader)` on
  `IArchitectureRecipe` and `IVideoRecipe`. Wan implements it with the pipeline's own
  `WanActivationReserveBytes` and the decode estimate now shared as `WanDecodeReserveBytes`, so the estimate and the
  planner cannot drift. Families without it get header weights plus a pixel-scaled allowance, reported as
  `MemoryEstimateAccuracy.HeaderOnly`.
- `ByteFormat.GbF1` and `BlockStreamingOptions.DefaultPrefetchAhead` join the shared primitives.

## alpha.161

- **A bare `vulkan` selector pinned the loader's device 0, so alpha.158's fallthrough could land on a software
  rasterizer with a real GPU sitting next to it.** `VulkanDevice.Create` has always taken a nullable ordinal whose
  null path ranks devices (discrete GPU first, rejecting anything that fails the kernels' capability requirements),
  and `VulkanBackend`'s own summary claims it creates "a Vulkan backend on the best discrete GPU". Its parameter
  defaulted to `0`, and `BackendFactory.CreateVulkan` took a plain `int`, so nothing in production ever reached the
  ranking: `PickBest` was live only in tests. That was survivable while Vulkan had to be asked for by name. It stopped
  being survivable in alpha.158, when `auto` started choosing Vulkan on its own, because the machines that gain are
  exactly the ones that enumerate Mesa's lavapipe alongside the real card. `VulkanContext` would correctly count one
  GPU, `auto` would correctly answer `vulkan`, and `Create` would then bind raw index 0 and run the model on a CPU
  implementation of Vulkan. The test harness already knew: `BackendGate` hand-rolls a scan for a non-software device,
  its comment noting that probing only ordinal 0 "would report a machine with a 4090 in it as having no usable
  Vulkan". Selection now reaches production, and an explicit `vulkan:N` still means raw index N.
- **New `BackendFactory.HasExplicitOrdinal`, because `ParseOrdinal` answers 0 for a selector that named nothing.**
  `vulkan` and `vulkan:0` are different requests (rank the devices versus pin index 0) and no existing API could tell
  them apart, so `Resolve`, `Create` and the probes all treated an absent ordinal as an explicit zero.
- **The Vulkan probe now builds the device it is vouching for.** `ResolveProbed` probed ordinal 0 and then let `Create`
  choose, so on a box whose index 0 is a rasterizer the probe tested the one device guaranteed to pass and reported
  the real GPU as proven. Both take the same nullable ordinal now and agree by construction.
- **Probe results are cached per device rather than once per API.** A single `bool?` handed the first caller's verdict
  to every later one, so `ProbeCuda(1)` returned device 0's answer. Latent while one ordinal was ever probed; live as
  soon as "ranked best" and "index 0" became distinct requests.
- **The LLM path built its backend from the slot key, which pinned index 0 on exactly the boxes ranking exists for.**
  `TextService` canonicalizes a request device into a slot-and-gate key, and `CanonicalDeviceKey` MANUFACTURES an
  ordinal — a bare `vulkan` comes back as `vulkan:0`. That key was then handed to `CreateBackendFor`, so the one
  spelling that should rank was the one guaranteed not to, and an explicit `vulkan` request reached the rasterizer
  even after the fix above. Blank requests were unaffected and so disagreed with named ones, because `PrimaryDeviceKey`
  goes through `WithOrdinal`, which keeps ordinal 0 bare. The key still identifies the slot and the gate; the backend
  is now built from the selector as written.
- **New `IBackend.DeviceKey`: the identity of the device a backend actually bound to.** Hosts that track which engines
  share a GPU were composing a key from the selector they requested, which stops being the device in use the moment
  selection is left to the engine. Vulkan reports its device UUID, so two identical cards stay distinguishable and one
  shared card cannot read as two; CUDA reports its ordinal, which it honours as given. `VulkanDevice` also exposes the
  index it chose, and `VulkanBackend`'s `DeviceKind` now carries that instead of the one it was handed.

## alpha.160

- **HeartMuLa and MiniMax Music 3 quant caches were read back transposed.** Since alpha.130 (#41) `GgufWriter`
  emits ggml `ne` order, the reverse of the engine's, and the checkpoint loader relabels on the way back in. The
  two disk-cached quantizations never did: `CsmWeightCache.LoadQuantized` and
  `MiniMaxMusic3WeightPolicy.QuantizeToCache` read their own cache through the raw `GgufLoader`, so every
  projection came back `[in, out]`, the backend derived `M = 0` from it and the first dp4a launch failed with
  `CUDA_ERROR_INVALID_VALUE` after the model had spent its minutes loading. Every `:q8`/`:q4` HeartLib and
  MiniMax variant on the CUDA path was affected; bf16 never touched the cache and kept working, which is what made
  it look like a card problem. `CsmWeightCacheTests` asserted the source shapes all along and has been failing
  since the writer changed.
- **The shape now comes from the source dictionary, not from a guess about the writer.** New
  `GgufQuantizer.ReadBack(loader, source)` hands each cached tensor back under its source tensor's shape, which is
  a no-op for a cache written before the writer changed and a swap for one written after, so no cache on any host
  has to be deleted and re-converted. A tensor with no source falls back to reversing the file's axes, which is
  right for anything the current writer produced. Both readers use it; a MiniMax depth-decoder test and a
  quantizer round trip over both file orders join the CSM one.

## alpha.159

- **NVFP4 weights were being dequantized with the wrong block scales.** ComfyUI stores them in NVIDIA's blocked
  layout — `BlockScaleSwizzle` says so and says it was verified byte-exact against `comfy.float.to_blocked`, and
  `Nvfp4ResidentCodec` and `Nvfp4Linear` both honour it — but `DequantNvfp4ToF16`/`ToFp8` indexed the same bytes
  row-major. That is the path every `LlamaStyleEncoder` text encoder takes. Nothing caught it: when rows are a
  multiple of 128 the stored and padded shapes match, so the shape guard passes and the output is merely
  degraded. Measured against a BF16 copy of the same real tensor (Qwen3-8B `layers.0.self_attn.k_proj`, 20k
  sampled elements): row-major correlates **0.9186**, swizzled **0.9954**, and 0.9954 is nvfp4's own
  quantization error. Five staged encoders are affected, Gemma-3-12B worst at 302 nvfp4 groups.
- **The test that should have caught it asserted the bug.** Its reference was generated with
  `repeat_interleave(16, dim=1)` — the row-major assumption — and it skipped unless two local files existed.
  Replaced with a swizzle round-trip over distinct per-block scales, confirmed to fail against the old indexing.
- **FP4 is finished, and still unexecuted.** `Fp4GemmExecutor` existed but was never wired, its docs listing two
  things to confirm on hardware. The installed cuBLAS 13.6 headers answer both without a card: the scale-mode
  attributes are `A/B_SCALE_MODE` 31/32 with `VEC16_UE4M3` for NVFP4 and `VEC32_UE8M0` for MXFP4, and the layout
  question dissolves — cuBLASLt wants its own blocked layout, which is what checkpoints already store, so their
  scale tensors pass through untouched. The mode must be set or cuBLASLt reads each pointer as one per-tensor
  F32. `CUDA_R_8F_UE8M0` was bound to 34, past the end of `cudaDataType`; it is 30. Native FP4 GEMM has **never
  run** — no Blackwell hardware here — so it stays SM-gated with its refusal tested.
- **Flux.2 Klein 9B is no longer refused.** The check scanned `DType.Name` for `F4` and the encoder holds only
  U8/F32/F8_E4M3/BF16, so it never fired; a file that did declare FP4 would have died earlier in `ParseDType`,
  which now maps it. The encoder opens through `CheckpointSource`, which also frees what it allocates —
  `LlamaStyleEncoder.Dispose` never released projection tensors, so the previous route leaked them.

## alpha.158

- **`auto` considers Vulkan, so a non-NVIDIA GPU stops being treated as no GPU.** `BackendFactory.Resolve` chose
  between CUDA and CPU and nothing else, so every machine whose GPU is served by Vulkan rather than CUDA (AMD,
  Intel, and NVIDIA cards with no CUDA toolkit installed) resolved `auto` to the CPU backend while a working GPU
  sat idle. Vulkan was selectable, but only by naming it explicitly, which means it was reachable only by someone
  who already knew the default had failed them. The order is now CUDA, then Vulkan, then CPU, in both
  `Resolve` and `ResolveProbed`.
- **A software rasterizer does not count as a GPU.** New `VulkanContext.IsAvailable`/`GetDeviceCount` back the
  decision above, and they exclude `VK_PHYSICAL_DEVICE_TYPE_CPU` devices: Mesa's lavapipe enumerates as a Vulkan
  device on a machine with no graphics hardware, and it is a CPU implementation of Vulkan, so counting it would
  make `auto` pick something slower than the CPU backend it was chosen over. Linux CI images ship lavapipe, so
  without the exclusion this would have moved every containerized `Create("auto")` onto a software rasterizer.
  The count is cached, because unlike CUDA's device query it has to start the loader, and `Resolve` is called from
  banner and cache-key paths that assume it is cheap.
- **`ProbeVulkan` is the Vulkan twin of `ProbeCuda`**, sharing its matmul-against-the-CPU body. It earns its keep
  for a reason CUDA's does not have: a Vulkan device can enumerate and still fail the engine's own requirements
  (FP16, the subgroup ops the kernels are written against, a compute queue), which surfaces as a throw from device
  creation rather than as a missing device.
- **`Validate("vulkan")` checks for a device instead of only checking spelling.** Its remark that "Vulkan has no
  cheap availability probe" stopped being true with `VulkanContext`. An explicit `vulkan` selector on a machine
  with no Vulkan GPU now fails at startup with the loader's reason, matching what an explicit `cuda` already did,
  rather than deferring to a driver error mid-generation. Only presence is checked, not the ordinal: a
  `vulkan:{n}` ordinal indexes the loader's raw device list, software rasterizers included, so bounding it by the
  GPU count would reject valid ordinals.

Behavior change worth calling out: a machine with a Vulkan GPU and no usable CUDA now runs on the GPU where it
previously ran on the CPU. That is the point, but it is a change of device for anyone who was relying on `auto`
meaning "CUDA or CPU"; naming `cpu` explicitly still pins it.

## alpha.157

- **`--set` works.** It has never worked: it shipped in alpha.39 on 2026-08-26 parsing the flag out of `args`
  but never removing it, and `StrictParsing` then refused the run with `Unexpected option 'set'`. So the only
  way to change a setting for one run, without writing to the settings file, failed on every invocation.
  `--profile` was broken the same way and is fixed with it. A trailing `--set` with no value is deliberately
  still rejected rather than silently dropped.
- **The benchmark scripts set knobs instead of environment variables.** They exported `HARTSY_*`, which the
  engine stopped reading at the settings rebuild, so seven scripts had been configuring nothing: `h3_gold.sh`'s
  deterministic reference ran with neither its precision settings nor its probe, `ltx25_distilled_bench.sh`'s
  single-pass arm ran the two-stage path it meant to disable, and `run_benchmarks.sh` wrote a "flags this run
  executed under" section listing flags that were never applied — that section now asks the engine what is in
  force rather than asserting it.
- **`h3_gold.sh` resolves its checkpoint through the configured models root**, as `h3_bench.sh` does. The
  hardcoded repo path stopped existing when the checkpoints moved to the array, so the script could not run.
- **Flux.2 Klein is not unsupported.** `Flux2Recipe` already detects it from the transformer's hidden size
  (3072/4096/6144 → Klein 4B / Klein 9B / Dev) and carries its 10-step distilled defaults. Klein 9B is blocked
  on its FP4 text encoder, which the recipe refuses by name; the weights are gated on HuggingFace, not open.

## alpha.156

- **One README serves GitHub and nuget.org.** `README.nuget.md` existed because nuget.org renders a package
  readme with no repository context — but what that needs is absolute links, not a second file, since
  nuget.org does not resolve repository-relative paths. Every link and the benchmark badge are now full URLs,
  `PackageReadmeFile` and the pack include both point at `README.md`, and the packed nupkg carries it without
  NU5039. The merged file also gains the Configuration section the package-facing copy never had, because a
  consumer otherwise has no route to "settings live in one file and the engine reads no environment variables".
- **The docs name knobs again instead of environment variables that stopped working.** The settings rebuild
  moved every engine switch onto a knob and left the engine reading no environment variables, but 28 mentions
  across 9 files still told a reader to export something inert — the same failure that had every MiniMax-H3
  benchmark running at Warning level while its harness claimed Info. Twenty renames (`HARTSY_STEP_CACHE` →
  `vram.stepCache`, `HARTSY_FP8_NATIVE` → `numerics.fp8Native`, and so on); three switches have no
  replacement and now say so. `HARTSY_REQUIRE_REAL_WEIGHTS` and `HARTSY_RUN_H3_GUIDE_MASK_REAL` are untouched
  — those are real variables the test harness reads.
- **The model status docs say what is missing, from the upstream diff rather than memory.** Both "not yet
  built" lists are now `RecipeRegistry`/`VideoRecipeRegistry` against SwarmUI's `T2IModelClassSorter.cs` and
  ComfyUI's `supported_models.py`, dated and attributed; video had no such section at all. "Flux.2 Klein 9B
  (no public weights)" was false, and the extension already routes both Klein classes into the 32B `flux2`
  recipe. Wan 2.5/2.6/2.7 video are not open weights and are now a do-not-chase note.
- **MiniMax-H3's status entry records the alpha.97-to-154 regression**, which it previously read straight past.

## alpha.155

- **MiniMax-H3 generates again.** It has produced nothing since alpha.97: the DiT's weight preload runs out of
  VRAM 310 weights in (`requested 250 MB but only 310 MB available`) on a 4090 with the card otherwise empty, at
  the same 141f 512x288 geometry that completed in 215 s on alpha.75. Both consumer paths fail identically —
  the CLI errors out, and `/v1/native/video/stream` answers 200 and then emits one `event: error` carrying that
  message and zero frames — because the engine has a single video path and both consume it.
- **The cause was the text encoder materializing in full, not the DiT.** alpha.97 moved H3's components onto
  `CheckpointSource`, and its default folds quantization companions onto the weights they describe. H3's
  conditioning tower is the one consumer in the tree that does not want that: it binds every projection as
  `Nvfp4Linear`, which keeps the U8 bank packed and dequantizes one BF16 slice per forward out of a shared
  scratch, and it finds the block scales by KEY (`.weight_scale`, `.weight_scale_2`, `.pre_quant_scale`). Folding
  removes those keys, so the pass widened all 350 banks to F16 instead: `qwen3vl_32b_minimax_h3_nvfp4_awq`'s 2054
  tensors arrived as 1002, the load went from a memory-mapped open to 50 GB of host RSS, and what it left on the
  card was 59 MB short of the DiT.
- **`CheckpointOpenOptions.KeepNvfp4Companions` leaves a complete nvfp4 group exactly as the file wrote it** —
  the packed U8 weight plus its two scale companions — while fp8, int8 and NF4 fold as before. A group is
  identified structurally (U8 `.weight` + rank-2 F8E4M3 `.weight_scale` + F32 scalar `.weight_scale_2`), by the
  same three conditions the eager branch tests, so a weight this does not cover keeps its existing handling
  rather than being stranded with companions no loader expects.
- **The two narrower options do not reach this case.** `ResidentNvfp4` moves the scales onto `Tensor.QuantInfo`,
  which has no `pre_quant_scale` field, so `Nvfp4Codec.TryAttachResident` refuses AWQ layers — and this encoder
  ships 100 of them. Turning `FoldQuantCompanions` off wholesale would stop H3's `model.embed_tokens.weight`,
  which is int8 with an F32 `[151936, 1]` row scale read through `QuantInfo.RowScale`, from getting a scale at
  all, taking the published `int8_convrot` build down with it.

## alpha.154

- **Every checkpoint the engine converts now stamps itself.** The five conversion sites that ran on a user's own
  machine — Kokoro's `.pth` fallback, RVC's ContentVec encoder and RMVPE pitch estimator, YuE's x-codec and its
  Vocos vocoders — wrote anonymous files. Nothing ever stamped them afterwards either: the install-time sidecar
  only runs for models fetched whole from a repo, so a locally converted file had no identity by any route.
- **Kokoro's repack is the one primary artifact and carries a full ModelSpec block**; the other four are
  components and carry provenance without an architecture, so converting them cannot add four unusable entries
  to the model list.
- **Provenance records the source, not just the family.** `ArtifactProvenance.FromSourceFile` hashes the input, so
  a converted file names the exact bytes it came from rather than a file name that may since have been replaced.
  Hashing failures are swallowed — provenance is worth recording and never worth failing a conversion over.
- `YuePipeline`'s `yue_dump.safetensors` is deliberately left unstamped: it is a parity diagnostic, not a model.

## alpha.153

- **Converted checkpoints can now say what they are.** Every safetensors the engine wrote was anonymous: of the
  48 artifacts staged for publishing, none carried a `modelspec.*` key and 26 had no `__metadata__` block at all.
  `SafeTensorsWriter.Save` and `PickleCheckpointRepacker.Repack` both took an optional metadata map that defaulted
  to null, and six of the eight places that write a checkpoint passed nothing. SwarmUI classifies a scanned model
  from `modelspec.architecture` before it reads a single tensor, so a repack we produced landed with a null class —
  which for an audio model means its parameters silently disappear from the UI.
- **`ModelIdentityCatalog` is the one place a family's publishing identity lives.** Class id, title, author,
  license, upstream repo and standard resolution, keyed by engine model id, for 74 families. That information was
  split across three tables that no conversion site could reach: the backend extension's `ModelSupport`, the
  classes that extension registers itself, and a JSON file beside a Python tool. It sits in `ModelAssets` rather
  than `Engine` so the conversion sites in `Audio` can read it.
- **`ArtifactMetadata` builds the header, and emits a resolution only when the class declares one.**
  `IdentifyClassFor` accepts a model whose resolution matches the class standard or is absent; for anything else
  it substitutes a clone carrying the stamped size with its heuristic matcher disabled, so the class ends up
  reporting a standard size nobody declared. Audio classes are registered 0x0, so they must carry none, and
  Qwen-Image 2.1 must carry its own 1024 rather than Qwen-Image v1's 1328.
- **Only the primary weights get an architecture.** A codec, vocoder or pitch estimator that lives in its own
  file is part of a model, not a model, and the index admits only `hartsy.component=main`. Naming the component
  is required rather than defaulted, because the default is the dangerous one: four of the five conversion sites
  in the engine write components, and stamping them as primary would put each one in the model list as something
  a user can select and generate nothing with. A component also claims no author or license — ContentVec and
  RMVPE ship inside RVC but are other people's work under other terms.
- **`ArtifactNaming` writes down the file-name convention that was never written down.**
  `<engine-id>[-<variant>]_<precision>.<ext>`, so `fp8_scaled` and `fp8 scaled` cannot produce two names for one
  build, while GGUF presets keep their upstream `Q4_K_M` casing.

No behaviour changes yet: this release adds the catalog and the builder. Threading them through the conversion
sites, and the `hartsy pack` command that produces a whole upload-ready bundle, follow.

## alpha.152

- **Community benchmark runs now attest the GPU and sample it while they run.** `EnvironmentRecord.PowerProfile`
  was a cohort-key component that nothing ever assigned — it was the literal `"unreported"` on every campaign —
  so a card power-limited to 300 W pooled with a stock 450 W one and their medians were averaged together. The
  controller now reads the device's power limit, clock caps, persistence and ECC mode from `nvidia-smi` before
  the budget starts and makes that the cohort profile.
- **A campaign refuses to start on a GPU another process is already using.** Nothing checked before, and the
  protocol notes conceded as much ("background utilization is currently operator-controlled and unverified").
  `run` names the offending PIDs and their VRAM and stops before producing anything; `--allow-shared-device`
  records the sharing and proceeds. `doctor` reports the same two things without running a workload.
- **Per-request GPU telemetry replaces the two dead memory fields.** Each worker session runs one long-lived
  `nvidia-smi -lms` child and stores aggregates per measured request: peak device VRAM, utilization, power,
  temperature, clocks, sample coverage, and throttle-reason counts. `Measurement.SampledUsedDeviceBytes` and
  `MemorySource`, which the validator previously *required* to be absent, are gone in favour of a
  `DeviceTelemetry` record. Only aggregates ship, because `Bundle.Export` whitelists what leaves the machine.
- **A thermally or hardware-throttled session is retained but never published.** The limit is frozen in the
  suite manifest (`maxThrottledSampleFraction`), the worker marks the session `throttled`, and the validator
  re-derives the same fraction so a session cannot publish by claiming otherwise. `gpu_idle` is not counted as
  a throttle — an idle card reports it continuously — and neither is a software power cap, which is what a
  stock card under sustained load looks like.
- **A device `nvidia-smi` cannot describe stays publishable, in its own cohort.** Its profile records as
  `unattested`, so it never pools with attested runs and the explorer marks it. Vulkan and CPU campaigns record
  no telemetry and are not disqualified for its absence.
- **Binding is by GPU UUID, never by ordinal.** CUDA enumerates fastest-first, so on a two-card host
  `cuda:0` is nvidia-smi's index 1. The controller matches a device by hashing each `nvidia-smi` UUID with the
  recipe that produced `DeviceRecord.Identity`, which keeps the raw UUID out of the exported record, and
  refuses rather than guessing when nothing matches. NVML is deliberately not used: its process-list entry
  point is struct-size versioned and its throttle reasons are header constants, while `nvidia-smi` names both
  as CSV columns readable with no toolkit installed.
- **The explorer shows peak VRAM, peak power, and ms/step for image cases**, and marks unattested rows. The
  image scoreboards already quote ms/step, so community data is now in the same unit as our own tables.
- **The CLI step counter prints each step's own duration** (`denoise [7/20] 691 ms`). It printed only the
  counter, which is why `benchmarks/minimax_h3/h3_bench.sh` could never report a per-step mean; that harness now
  parses the timing and resolves its checkpoint through the engine's configured models root instead of a repo
  path that does not exist.
- **The README benchmark badge points at the checked-in snapshot.** The Pages URL it used returns 404 until
  Pages is activated for the repository, so the badge was a broken image.

## alpha.151

- **Qwen-Image 2.1 now matches SwarmUI's own native support.** SwarmUI core gained a `qwen-image-2.1` model
  class, compat class and VAE family of its own (`2de300f6`, "Adds Qwen2.1 support"), so this release lines the
  engine up with what its ComfyUI backend does rather than running a parallel set of choices.
- **The text encoder is `qwen3vl_8b_int8_convrot.safetensors`, the same file SwarmUI downloads for ComfyUI**
  (`GetQwenImage21TextEncoder`), instead of `qwen3vl_8b_bf16`. One 8.6 GB copy serves both backends where two
  backends previously wanted 8.6 + 16.4 GB of the same encoder. Per-row int8 with a Hadamard rotation is also a
  tighter fit than fp8 at this model's no-final-norm tap.
- **Fixed: an int8-quantized token embedding loaded at ~100× its true magnitude.** `DType.I8` reports
  `IsQuantized == false`, so `LlamaStyleEncoder` widened `embed_tokens` through `Tensor.CastTo` and dropped both
  the per-row scale and the ConvRot rotation. Every int8 encoder shipped so far (LTX-2.5's Gemma-4) keeps its
  embedding BF16, which is why nothing had hit it; Comfy-Org's Qwen-Image 2.1 encoder quantizes it.
- **Fixed: the Qwen-Image 2.1 VAE downloaded to a second path.** `SideModels.QwenImage21Vae` wrote
  `VAE/qwen_image_2.1_vae_bf16.safetensors` while SwarmUI core registers
  `VAE/QwenImage/qwen_image_2.1_vae_bf16.safetensors`, so the two backends each fetched their own copy. Same sha,
  same file, now the same path — matching what every other VAE in `SideModels` already did. An install that
  already has it under the alpha.149 name keeps using it: the asset carries `LegacyTargetNames`, so
  `ModelDownloader.TargetPath` resolves to the existing file rather than re-fetching 675 MB. (Deliberately *not*
  done for the text encoder — bf16→int8 is a content swap, not a rename, and falling back would silently keep
  serving the wrong file.)
- **Fixed: Qwen-Image 2.1 reported that it takes no sampler or scheduler.** It had no row in
  `SamplingCapabilities`, and a miss there is indistinguishable from a family that owns its own solver — so
  SwarmUI hid the Sampler and Scheduler controls and refused any explicit pick, while the pipeline was calling
  `FlowMatchSampling.Resolve` all along. It now declares the full seam, as Qwen-Image v1 does.
- **Fixed: the test that was supposed to catch that could not fail.** `CapabilityTable_NamesOnlyRealFamilies`
  asserted `Count > 0 || == Unknown || Count == 0`, which is a tautology. Replaced with
  `CapabilityTable_CoversEveryImageRecipe` over the new `SamplingCapabilities.HasImageEntry`, plus a negative
  control so the coverage check cannot silently become vacuous again.

## alpha.150

- **Settings have one home and can be written.** The file is now exactly `~/.config/hartsyinference/settings.json`
  (`hartsy settings path` prints it), instead of being searched for in the working directory, beside the entry
  assembly and then the home directory — which meant the settings that applied depended on where a process was
  started from. A host that keeps its settings elsewhere still sets `KnobFile.ExplicitPath`.
- **`hartsy settings list | get <id> | set <id> <value> | path`.** `set` persists, so changing the models folder
  survives a restart, which is the thing users were expected to change and could not. `get` reports the effective
  value **and which layer supplied it**. `--set` and `--profile` are unchanged and still affect one run only.
  `--list-settings` is replaced by `settings list`, which hides the diagnostics domain unless `--all` is passed.
- **`GET /settings` now describes the engine, not just the host.** It gained an `engine` section listing every
  setting with its value, source, type, default and when it applies; `GET`/`PUT /settings/engine/{id}` read and
  persist one setting. Server options (ports, backend, API key) stay ASP.NET-owned and read-only here.
- **A written value is validated and coerced when it is written.** An unknown id, a wrong type or an
  out-of-range value fails at `set` rather than at the next startup, through the same parse the file load uses,
  and the stored value is the one the engine will actually honour — `numerics.gemvWpb=999` is written as `16`
  because that knob clamps rather than rejects.
- **`KnobStore` records which layer set each value** instead of inferring it. A host override and a file value
  share one dictionary, so once the file had supplied a value a later `KnobStore.Set` was indistinguishable from
  it. This is load-bearing for the SwarmUI extension, which drives `paths.modelsRoot` from SwarmUI's own
  `ModelRoot`; `settings get` now reports that as `host` rather than claiming the file set it.
- **The environment names nothing reads are gone.** The environment layer was removed in alpha.40, but every
  knob still recorded the variable it used to be read from — 219 in the registry, plus 293 comments and `--help`
  strings telling an operator to export something inert. Those are rewritten to the setting id that replaced
  each (`HARTSY_KEEP_MODELS=0` → `vram.keepModels=false`), and `LowVramPolicy.EnvironmentVariable` became
  `SettingId` — it only ever built log lines, so the VRAM logs had been announcing a variable the engine had
  not read in months. **A stale export is now silently ignored**; the reporter that named it is deleted.
- Guard tests replaced rather than dropped: the source scan now asserts engine code reads **no** environment
  variable outside the third-party set we do not own, and the two deliberate knob pairs (graph capture,
  SageAttention) are pinned by id and default instead of by a shared variable name. `docs/SETTINGS.md` replaces
  `ENV_VARS.md`.

## alpha.149

- **Qwen-Image 2.1 (`Comfy-Org/Qwen-Image-2.1`) generates end to end.** Despite the version number it shares no
  block structure with Qwen-Image v1: a **single-stream** DiT over the concatenated `[text, image]` sequence (32
  blocks, hidden 4096), **one modulation shared by every block**, scale-only adaLN with no shift, a fused-`gate_up`
  SwiGLU, no biases anywhere, a 64-channel patch-1 latent, **Qwen3-VL-8B** tapped at the last decoder layer with
  **no final norm**, and the **Wan 2.2 VAE** at temporal kernel 1 emitting **four** channels — alpha is a model
  output here, not a matte, so `ImageResult.Alpha` carries it when the image is actually transparent. New family
  id `qwen-image-2.1`, its own detector rule, and both Qwen families now refuse the other's checkpoint by name.
  Verified at 1024²/25 steps on a 4090, plus `--cfg 2.5`, the `int8_convrot` build, and bare-path detection.
- **The text prefix is evaluated once per prompt instead of once per step.** Text rows modulate from `t = 0` and
  attend only to earlier text rows, so their per-block K/V are constant across the denoise loop; they run once
  into a `QwenImage21PrefixCache` and the image rows run alone against it. Every block call then sees a uniform
  modulation and needs none of the per-row-range scale/gate splitting the reference performs.
- **The Wan 2.2 VAE's `temporal_kernel` now threads through every conv, not just the resample's `time_conv`.**
  The reference applies it in `conv3x3` too, so `decoder.conv1`, both convs of every residual block and the head
  are all `(k,3,3)` with padding `(k//2,1,1)`. Keeping Wan's `padT=1` against a depth-1 kernel grew `T` by 2 per
  conv and the residual add read past its operand — an access violation on CPU, and on CUDA an async illegal
  address that surfaced at an unrelated `Free` during teardown. **Wan 2.2 itself is unchanged**: the default
  kernel is still 3. `Wan22VaeLatentNorm` also gained span overloads, because a consumer's latent statistics are
  its own and Wan's embedded 48-channel table is both wrong-shaped and wrong-valued for a 64-channel model.
- **Prompt weighting on Qwen-Image 2.1 is declared, wired, and provably inert** — the Kandinsky5 situation by a
  different route. The DiT's first act on the conditioning is `txt_in.text_norm`, a per-row RMSNorm, which is
  scale-invariant per row, so a per-row multiply cancels exactly. SwarmUI scales the same tensor, so its
  CondScale does nothing here either and reproducing the no-op is the parity behaviour. Recorded with the
  inverted gate and the control that makes it meaningful (a different prompt moves the image by 34/255).
- **Performance vs ComfyUI at master, same 4090, same checkpoint, 1024²:** **691 ms/step against their 493 ms**,
  per-step taken as the slope between 25 and 50 steps so load, encode and decode cancel on both sides. Output
  quality is equivalent. The gap is the activation dtype and its consequences, not the GEMM: F32 activations
  against BF16 weights already resolve to a BF16 matmul, but pay a per-call activation cast and double the
  elementwise bandwidth. F16 is the named lever and the modulation chain is already F32-internal so the dtype is
  free to change. F16 itself was tried and is blocked on weight residency rather than precision: it renders
  correctly at 512² (0.59/255 from the F32 image) but OOMs at 1024², because F16 activations against a BF16
  weight force a BF16→F16 materialization of the whole DiT, where F32 activations pick a BF16 GEMM and cast no
  weight at all. The lever is converting the checkpoint to F16 at load, not flipping the activation dtype.

## alpha.148

- **Eight `VkStructureType` values were wrong, and the consequence was that Vulkan enabled no optional feature at
  all.** A wrong sType does not fail loudly — a driver that meets an unrecognized struct in a `pNext` chain skips
  it — so the `VkPhysicalDeviceVulkan12Features`/`Vulkan13Features` structs were invisible both when querying and
  when creating the device. The query came back all zeros, which this code has carried vendor-ID fallbacks for
  since bring-up under the belief that the NVIDIA driver misreports promoted features; it does not, and the
  fallbacks were papering over this. More seriously, `vkCreateDevice` enabled **nothing**: `synchronization2`
  (every barrier and submit here is the 2 form), `timelineSemaphore` (the whole stream is built on one),
  `subgroupSizeControl` and `computeFullSubgroups` (the reduction kernels ask for full subgroups),
  `storageBuffer16BitAccess` (every F16 kernel), and `maintenance4` — which is what permits `LocalSizeId`, the
  spec-constant workgroup size every kernel in this backend declares. NVIDIA permits all of it unrequested. A
  driver is not required to, which is the likeliest reason cross-vendor was expected to be painful.
- **The vendor-ID allowlist that compensated for it is gone.** With the right sTypes this device answers 1 for
  every feature the query asks about, so the apiVersion-plus-vendor fallback only ever claimed features a device
  might genuinely lack — the direction that breaks rather than the direction that is slow. The query is the answer
  now.
- `MemoryBarrier2` and `BufferMemoryBarrier2` were swapped, so every barrier this backend recorded was tagged as
  the other kind; the KHR cooperative-matrix features/properties pair was swapped the same way; and
  `ShaderModuleCreateInfo` was 15 (image view) instead of 16.
- **Found by running a real generation under `VK_LAYER_KHRONOS_validation`**, which names each one by VUID. Worth
  keeping as a habit: the backend had been developed for months against a driver that tolerates all of it.
## alpha.147

- **Buffer copies on Vulkan are synchronized against the dispatches around them.** Every compute dispatch ends with
  a barrier whose destination scope is `ComputeShader`/`ShaderStorageRead`. A `vkCmdCopyBuffer` reading the same
  memory is a `Copy`/`TransferRead` access and sits outside that scope, so nothing ordered a dispatch's writes
  before the copy that reads them — and on the other side, a copy's `TransferWrite` sits outside the source scope
  of the next dispatch's barrier. `Concat` (every DiT forward joins the text and image sequences through it),
  `CopyInto` and `CopyTo` all recorded copies into that gap. Both directions are closed now.
- **The staging copies had the same gap.** `VulkanGpuTransferHelper`'s upload and download each record a copy with
  a post-barrier only, so a destination a dispatch just wrote — or one an earlier staging copy wrote — was not
  ordered against it. Consecutive uploads are the bulk of what synchronization validation reports on a real
  generation.
- **So did the barrier every dispatch records.** Its destination scope was `ShaderStorageRead`, so two dispatches
  writing the same buffer — an in-place op following the op that produced its input — were a write-after-write
  nothing ordered. With every copy ordered, that pair is what synchronization validation reports, and it is the
  single hottest barrier in the backend.
- **A copy's destination scope named only reads.** Every post-copy barrier made the copy visible to
  `ShaderStorageRead`, so a dispatch that *writes* the buffer it just received — which is what an in-place op does
  straight after an upload — was a write-after-write nothing ordered. That pair is what synchronization validation
  still reported once the other gaps were closed.
- **`Concat`'s trailing barrier was the wrong one.** It recorded the compute→compute barrier after a transfer, so
  its source scope named `ShaderStorageWrite` for writes that were `TransferWrite` — it ordered nothing. It is now
  a real transfer→compute barrier. `CopyInto` and `CopyTo` already had a correct post-copy barrier through
  `RecordCopyAndBarrier` and needed only the pre-copy half; the remark shipped in alpha.146 said otherwise and is
  corrected.

## alpha.146

- **The head-major chunked-attention trio runs on the GPU on Vulkan.** `QkvSplitNormHeadMajor`,
  `ApplyRopeSingleHeadMajor` and `ScatterSeqHeadMajor` are the three ops MiniMaxH3's DiT calls back to back, and
  all three were interface defaults on Vulkan — host loops over `DataPointer`. Every attention block synced the
  packed projection down, normalized it on the CPU, uploaded three tensors, synced two of them back to rope them,
  uploaded them again, and did it twice per block on the chunked path, which projects k+v in one pass and q in the
  next precisely to keep a full-sequence q from staying resident. Wan-Animate-2's per-frame attention reaches the
  scatter on the same terms and Gemma-4's text encoder reaches the rope.
- **The head-major rope shares the token-major kernel.** The two layouts hold the same elements with heads and seq
  swapped, and `headDim` is innermost either way, so a vector's own offset is unchanged and only the cos/sin row
  has to be recovered differently — one spec constant (`HEAD_MAJOR`), not a second binary. The parity rows use
  more than one batch AND more than one head, because at either equal to one the two layouts coincide element for
  element and a kernel reading the wrong one passes.
- **The head-major QKV split is its own kernel, and serves a subset.** A caller can ask for any of q/k/v, from a
  source that may itself be narrower than `[q|k|v]`: a 3-wide source keeps the canonical q=0, k=1, v=2 segments
  even when only some outputs are wanted, while a narrowed `[k|v]` or `[q]` carries only what it names. Reading k
  from segment 0 and from segment 1 are both correct, for different sources, and picking the wrong rule returns a
  well-formed tensor of wrong values — so there is a parity row per source width. Deliberately NOT spec constants
  on `qkv_split_norm`: that one is on a shipped generation path, and CUDA split its own kernel for the same reason.
- **`ScatterSeqHeadMajor` is one multi-region `vkCmdCopyBuffer`, not a shader.** A head's chunk rows are contiguous
  and heads are not, which is the per-slice shape `Concat` already issues; CUDA spends one device-to-device copy
  per head and this spends one command for all of them. It also does not upload the destination's host contents
  when allocating it — matching CUDA, because the destination is the whole attention key/value buffer
  (Wan-Animate-2 builds a `[1, heads, s + hw, headDim]` one per forward) and uploading it to write one chunk would
  move hundreds of megabytes. **This is a real divergence from the interface reference**, which writes only the
  chunk rows and so leaves everything outside them intact; on both GPUs that region holds whatever the allocation
  came with. Every shipped caller fills the whole buffer across its chunks, so nothing reaches it today.
- **Buffer copies recorded between dispatches now carry their own barriers.** The compute→compute barrier every
  dispatch ends with has `ShaderStorageRead` as its destination scope, so a transfer reading the same memory sits
  outside it — and a transfer that writes sits outside the source scope of the next dispatch's barrier in the same
  way. `RecordComputeToCopyBarrierOn`/`RecordCopyToComputeBarrierOn` close both directions; the new scatter uses
  them. `Concat` and `CopyInto` still record only the compute→compute barrier around their copies and have the
  same gap — noted here rather than changed, since both are on a shipped generation path and that is its own
  change with its own gate.

## alpha.145

- **`build.sh` stopped hiding what it did not build.** `set -e` aborted the whole run on the first kernel a given
  glslang could not compile, and every kernel listed after it was silently left stale — no error naming them, and
  an exit that read as success. Ubuntu's packaged glslang cannot build `matmul_int8` (no
  `GL_EXT_integer_dot_product`) and that entry sits partway down the list, so on an ordinary dev box the last five
  kernels had not been rebuilt by this script in a long time. **Found by editing one of them and watching the
  committed binary not change.** Failures are collected and reported by name at the end now, and the run still
  exits non-zero.
- **`ArgMaxLastDim` runs on the GPU on Vulkan.** The per-row form of what `ArgMaxInto` already did for a single
  decode step, off the same kernel — one workgroup per row instead of one in total. The interface default reads
  `input.DataPointer`, so it synced a whole vocabulary-wide row set to host to pick one index per row.
- **The reduction breaks ties explicitly now, to the lower index.** Which thread holds which candidate in a tree
  reduction is an artifact of the stride order, so an exact tie resolved differently depending on the workgroup
  size; the shader's own comment called that "measure-zero for real logit distributions", which is true right up
  until a head saturates. It matches the reference for ties as well as for maxima, and the parity row plants two.

## alpha.144

- **`CastToBf16` runs on the GPU on Vulkan.** The interface default builds a whole host-side cast tensor and
  memcpys it — on a resident input that is a device sync, a host conversion of every element, and a re-upload. The
  shader it needed already existed, reachable only as an internal dtype conversion and never as the op itself.
- **A pre-existing divergence this surfaced, deliberately not "fixed".** `Tensor.CastTo(BF16)` **truncates** — its
  own doc says so — while both GPU kernels round to nearest, ties to even, which is what hardware and every other
  framework do. So the same op yields different bytes on CPU and GPU, off by one unit in the last place on roughly
  half of all inputs, and has since CUDA first overrode it. The new parity row therefore pins the rule the GPUs
  implement rather than comparing against the host cast: changing either side would alter shipped numerics to make
  a test pass. Which rounding should be canonical is a real question and belongs in its own change.
- **F16 input chains through F32**, as CUDA's override already did — without it an F16 caller would have fallen to
  the host round-trip this override exists to remove. Both real callers pass F32 today.
- **A second divergence, in the NaN the two backends produce.** Vulkan emits the canonical positive quiet NaN and
  CUDA keeps the input's sign. IEEE 754 fixes neither the sign nor the payload of a produced NaN, so neither is
  wrong, and the parity row checks that element for NaN-ness rather than for bytes — every other element,
  infinities and both zeros included, is compared exactly.

## alpha.143

- **`ChwF32ToHwcU8` runs on the GPU on Vulkan.** The last step of every image generation — the VAE's `[B,3,H,W]`
  F32 output in `[-1,1]` to the `[H,W,3]` u8 an encoder wants — was an interface default here: a device sync of the
  whole F32 image, then a host loop over every pixel. The pixels have to reach the host either way to be encoded,
  but as bytes that is a quarter of the transfer, and the loop goes away.
- **One invocation per output WORD, not per pixel.** A pixel is three bytes, so pixels straddle word boundaries,
  and GLSL cannot address bytes without an extension. Composing whole words needs neither atomics nor a
  pre-zeroed buffer; the allocation rounds up to a whole word so the tail lanes write zero rather than reading
  past the image.
- The parity row uses 5×7 on purpose: only every fourth pixel starts on a word boundary, so the final word holds
  one real byte and three past the image. Its values are offset so a good share land outside `[-1,1]`, because the
  clamp is part of the contract and an unclamped kernel wraps rather than saturating. The comparison is exact —
  the rounding is round-half-up, and a pixel off by one is a byte off in the PNG.

## alpha.142

- **`ApplyRopeInterleaved` runs on the GPU on Vulkan.** Twelve call sites across the audio and LLM stacks reached
  an interface default that reads `x.DataPointer` — a device sync, a host loop over every head and position, and a
  re-upload for the next op, on a tensor the caller had just produced on the device.
- **One kernel now serves both rotary conventions.** GPT-NeoX split-half pairs `(i, i+half)` with frequencies at
  `i` and `i+half`; GPT-J interleaved pairs `(2i, 2i+1)` with one frequency serving both. Same dispatch shape, one
  invocation per pair, so a spec constant picks the offsets rather than a second committed binary.
- **The partial-rotary boundary follows CUDA, not the tidier rule.** Interleaved dispatches every pair in the head
  and drops the ones past the rotary window, rather than dispatching `rotaryDim/2` pairs — which is what the CPU
  reference and the CUDA kernel have always done. For an ODD `rotaryDim` the two rules genuinely differ: at 5 the
  shipped rule rotates the pair `(4, 5)` and the tidier one would not. Aligning the reference instead would have
  changed a shipped backend's numerics to make a new one look neater.
- **`IBackend.ApplyRopeInterleavedReference`** joins the other reference statics. An override cannot reach its own
  interface default — `((IBackend)this).X(...)` binds back to the class and recurses until the stack ends, which a
  lint already fails the build on — so a backend bailing on a dtype or rank needs somewhere to bail TO.
- Parity rows for both conventions, including a partial rotary. They differ only in which elements pair and where
  the frequencies live, so a kernel that confuses them still writes plausible numbers of the right magnitude in the
  right places; only a reference comparison separates them. Flipping the spec constant fails both interleaved rows
  and leaves the split-half ones green — checked by doing it. An odd rotary dim is among the rows, because that is
  the one input where the two boundary rules disagree.

## alpha.141

- **`WanRopeInterleavedPerHead` runs on the GPU on Vulkan.** It was an interface default that reads
  `x.DataPointer`, so MG3's sigma_theta rotation cost a device sync, a host loop over every head and position, and
  a re-upload on the next op — on a tensor that was already resident. The rotation is identical to the shared-table
  form already implemented here; only the cos/sin offset differs, so a spec constant selects it and both come off
  one binary rather than a second committed one.
- Parity rows for **both** forms, on both backends. Per-head tables need more than one head and tables that differ
  between heads to test anything: with one head, or with equal tables, the two layouts address the same bytes and a
  wrong index passes. Flipping the spec constant fails the per-head row and nothing else — checked by doing it.

## alpha.140

- **The safetensors quantizer streams, so a large source no longer needs the whole checkpoint as F32.** The GGUF
  writer has always interleaved — widen one tensor, quantize it, free the wide copy — but the fp8/int8 path built
  a complete F32 dictionary before the writer saw anything. That is what made a 13 GB Q4_K source ask for ~50 GB
  and get the process OOM-killed. `WriteSafetensors` now takes the `CheckpointSource` and does the same
  one-at-a-time loop the GGUF path does.
- **Two ownership traps this exposed, both of which would have quietly undone the change.** An ineligible weight
  is stored AS its wide copy, so that copy has to outlive the loop while every other one is freed — handled by
  reference identity against what the iteration added to the output, not by key. And the fp8 branch was parking
  its narrowed BF16 copy of every quantized weight in `owned` until the end; on a streaming loop that is half the
  checkpoint held for no reason, so it is disposed the moment the fp8 weight and its scale exist.
- **The refusal is narrowed to what is actually held.** It claimed the whole checkpoint as F32 plus the output;
  the real peak is the largest single tensor as F32 plus the finished file, because the output still has to be
  complete before a safetensors header can be written.
- **Verified as a pure refactor, byte for byte.** The pre-change code from a clean `origin/main` worktree and
  this one were run over the same SDXL source to the same target: **identical sha256**
  (`0f6d12f2517a9e60…`), identical size (4111075535). So the interleave changes when memory is held, not what
  gets written.
- **And verified against what it replaces.** Old path on this box: refused, `about 0 GiB is free`. New path, same
  job: `exit=0`, **10.7 GB peak RSS**, 4.1 GB output, and the result loads and renders. Widening SDXL whole is
  ~26 GB before counting the output, which is why the old one only runs on an idle machine. 80 quantizer tests
  pass unchanged.
- One thing the gate does NOT show, stated so it is not misread: the fp8 output renders a visibly different image
  from the dense original at the same seed (SSIM 0.379). That is fp8 being lossy on a chaotic 8-step trajectory,
  not a defect — the byte-identical comparison above is what rules out a regression, and SSIM cannot tell the two
  apart.

## alpha.139

- **CUDA is on the shared op scope.** `CudaBackend` was still a standalone `IBackend` carrying its own copy of
  everything `GpuBackendBase` exists to hold, so a layer written to be inherited had one inheritor. Two hundred op
  entries, the weight preload and free, activation release, the pool trim and the full device sweep now go through
  the base; the D2H counters and pin/unpin are gone entirely.
- **Why the first attempt at this failed, and what fixes it.** It broke three `CudaGraphTests` with
  `CUDA_ERROR_INVALID_VALUE`. The base's op scope calls the residency cache's own `SweepOrphans`, while CUDA's ops
  called a static wrapper — and the guards were on the wrapper, chief among them a `cuStreamIsCapturing` check,
  because freeing a buffer allocated before a capture began is rejected outright on a capturing stream. The
  conversion therefore dropped that guard at every op entry, and the first capture to park an orphan died. The
  guards now live on the cache, where the scope reaches them.
- **Op entry is two jobs, and only one of them is per-op.** CUDA resolves its transfer state through a
  thread-static ambient, so entry must bind this backend AND reclaim what the last op left. The reclaim stays at
  depth 0 — an op built of other ops must not free what its own later dispatches read — but binding runs at every
  depth through a new `OnOpEnter` hook, because `CopyFromPeer` enters another backend mid-op and returns to find
  that one bound. Its four calls were never scopes; they are rebinds and now say so.
- **Disposing twice still waits.** The base owns the once-only flag, so a second caller reaches `OnDisposeRepeated`
  and waits for the teardown it would otherwise return in the middle of, rethrowing what the first caller hit. The
  base suppresses finalization in a `finally`, since CUDA's teardown reports failure by throwing.
- A lint fails the build on a discarded op scope. `EnterOp();` as a bare statement still compiles — the result is a
  struct and C# will throw it away — and it raises the op depth permanently, so every later op is treated as
  nested: no finalizer drain, no orphan sweep, no flush.


## alpha.138

- **The conv half of the LoRA merge is verified against a real adapter.** Rank-4 convolution support has existed
  since the conv-LoRA work landed, but every test until now built its deltas by hand — so what was pinned was the
  arithmetic, not that a shipped adapter's conv modules resolve to weights this engine holds. Those are different
  claims, and only the second fails when a key rule is wrong. `LoraLoconConvRealTests` loads a LoCon
  (`conv_dim`/`conv_alpha` set, 98 rank-4 tensors = 49 modules) and asserts every one names a key the converted
  SDXL checkpoint has, with the delta's folded `[out, in·kh·kw]` matching the base conv's own shape:
  **49 conv modules checked, 0 unmatched, 0 mis-shaped.**
- **`hartsy image` gains `--lora` / `--lora-weight`, which it never had.** The video and inspect commands both
  carried them; `ImageCommand` had 32 options and no LoRA among them, and `ImageRequest.Loras` was never populated
  from the CLI — the stack builder, the request field and the merge all existed with no way to reach them.
  `git log -S"Loras"` on that file is empty, so it was an omission rather than a removal. This is also why the
  conv merge had never been exercised end to end: there was no way to ask for it.
- **Gate (real weights).** SDXL base at 512², seed 1. Two same-code runs hash identically (`e90f80c4`), so the
  family is deterministic here; adding the LoCon moves the image to `8582c5b4`. That alone would not prove the
  CONV layers merged — ~1000 linear modules would move it regardless — so the merge log is part of the gate:
  `Merged 788 of 788` into the UNet, `72 of 72` into CLIP-L, `192 of 192` into CLIP-G. 1052 of 1052 modules, none
  skipped, and the 49 convs are inside that UNet count.
- The adapter is Pony-trained, which is irrelevant to what is being shown: it shares SDXL's UNet and kohya's key
  grammar, and the claim is that the conv path resolves and fits, not that the output looks like anything.


## alpha.137

- **The free-VRAM report means one thing now.** The driver path described a single heap while the total described
  all of them — not theoretical on an RTX 4090, which is 24564 MiB while the summed total reported 24810 MB,
  because NVIDIA exposes a small second device-local heap for ReBAR. Callers comparing free against total were
  comparing two bases, and the recipes that refuse to construct below a VRAM floor compare exactly those. Both
  halves now come from the same heap, the device-local one with the most left.
- **"Zero free is an answer" is now true downstream as well.** `AudioRuntime` gated eviction on `free > 0`, so a
  driver honestly reporting a full card disabled the eviction written for that case; the LTX-2.5 decoder returned
  its optimistic default instead of its floor. Both gate on the TOTAL, which is what separates "no report" from
  "nothing left".
- **Tests that could not fail what they described.** The release test compared two probes of what used to be
  allocator arithmetic with no slack, and alpha.135 made it a live figure that moves with every process on the
  card. The test that claimed to assert which path answered never checked that `GetVramInfo` returned the driver's
  number — delete the branch and its upper bound holds at equality. Its sibling bounded only the difference between
  the two spellings, so a 2 GB window swallowed a return to the old constant zero. All three are fixed, and zero is
  admitted as the legal answer the production code says it is.
- **CUDA's two spellings are covered for the first time.** On Vulkan `FreeMemoryBytes` IS `GetVramInfo().FreeBytes`
  through the interface default and cannot disagree; `CudaBackend` keeps genuinely independent implementations
  reading different routes to the driver, which is the shape that drifts. A cross-backend contract test now asks
  both.
- **Every stream drain tells the backend it happened.** A drain always submits first, and the dispatch count that
  drives submit batching did not learn about it, so the next op crossed the flush threshold early against a number
  that was simply wrong. Harmless in effect — a submit with nothing recorded is a no-op — which is exactly why it
  survived at seven separate call sites, including the scalar read-back that runs on every decode step and the
  device-to-host sync behind every lazy activation read. Each class now has one drain helper and every site uses it.
- **The fallback describes the same heap the driver path does.** Fixing the free/total basis in the driver query
  left `Vk.TotalVramBytes` — a sum over every device-local heap — under the fallback's free figure for one heap, so
  the defect was relocated rather than removed, reachable whenever the budget extension is absent. And a disposed
  backend, which now reaches that path, would have computed "entirely free" from an empty allocator; it answers
  zero, as it did before.
- The fallback walks the device-local heaps itself rather than destructuring `MemoryStats`, which also computed a
  per-block free-list scan and a full weight-cache sum for a number nobody read — several times per denoise step on
  any device without the budget extension.

## alpha.136

- **LTX-2 honours `(word:N)`, and the deliberately-unwired ledger is now EMPTY** — every registered image and
  video family consumes its prompt weights. One recipe class serves `ltx-video-2` and `ltx-2.5-distilled`, so
  both come off together.
- **The scale goes inside `LtxVideo2TextConnectors`, after the per-modality projection and before the learnable
  registers**, and both neighbouring placements are wrong. Scaling the connector's OUTPUT hits register rows:
  our pipeline mirrors ComfyUI's `compat_mode`, where the real tokens sit at the FRONT and the tail is padding
  replaced by learnable registers. Scaling its INPUT is worse than it looks — the reference normalizes by a
  global min/max over the whole sequence before projecting (`lt.py:174-176`), so one token's emphasis would move
  the divisor every other token shares.
- **One deliberate divergence from SwarmUI, stated rather than buried.** SwarmUI scales whatever the CLIP node
  returns, which under `compat_mode` is post-connector — register rows. Reproducing that literally would be
  scaling learnable padding, so this implements ComfyUI's DEFAULT path semantics instead
  (`LTXAVTEModel.encode_token_weights` returns token-length embeddings there and lets the DiT connect them),
  which is what the weighting is for.
- **`ILtx2PromptTokenizer` gains `EncodeSpan` and `ConditioningStartId`**, because its two implementations
  disagree about specials: the Gemma-3 SentencePiece is constructed with `addBeginningOfSentence: true`, so
  every call prepends a BOS that has to come off a span, while `Gemma4Tokenizer.Encode` adds none. The start id
  is read from the tokenizer rather than assumed, even though both are 2.
- **LTX-2's conditioning cache is now weight-aware.** Its key was the token ids alone, and the emphasis is
  stripped before tokenization, so `(fox:1.5)` and `fox` produce identical ids — the second generation of a
  weighted prompt would have been served the previous weighting's conditioning. Same defect class already fixed
  for Chroma and HiDream.
- **The gate found a real bug, and it was in this change.** `ApplyTokenWeights` first scaled the rows HOST-side
  through `AsSpan<float>()`, on a tensor `backend.Linear` had just produced on the device — the
  discarded-device-write pattern. The symptom was not wrong pixels but `OutOfVramException`: a weighted prompt
  exhausting the 4090 at a geometry the same prompt completed at unweighted, twice, including once on an idle
  card. `(fox:1.0)` passed throughout because the scale returns early when every weight is 1, so only a prompt
  with a weight that actually differs took the bad path. It now goes through `backend.MaskRows` and stays on
  device.
- **Gate (real weights), 320x192 / 25 frames / seed 1, comparing FRAME PNGs** rather than the mp4, because this
  family's audio decode is nondeterministic run to run while its video is not:

  | run | frame hash |
  |---|---|
  | plain | `78ead7cd` |
  | `(fox:1.0)` | `78ead7cd` — byte-identical |
  | `(fox:0.5)` | `0802180b` — differs |
  | `(fox:1.5)` | `64847778` — differs from both |

  Two same-code plain runs hashed identically before the fix, establishing that this family's video really is
  deterministic run to run, so the differences above are signal. `plain` still hashes `78ead7cd` after the fix,
  the same value the pre-fix runs produced, so the unweighted path did not move.
- **Loading LTX-2 peaks at ~42 GB of host RSS**, measured across three runs (42.1 / 42.3 / 42.5 GB) against a
  21 GB on-disk int8-convrot checkpoint — roughly 2x the file, and worth knowing before scheduling a run.
- **A correction to alpha.134's MiniMax-H3 note.** It said H3's OOM was "the family's own load footprint, not
  contention", citing 42 GB free at the time. That is withdrawn: this box runs concurrent agents, one later
  measured holding 19 GB, and a `free` reading taken between their jobs looks like headroom that is not there.
  H3 "did not load here" — the number should not be read as a property of the family.
- `PromptWeightingModeLedgerTests.NotYetWired` is empty and stays in place: a NEW recipe that cannot weight yet
  needs somewhere honest to declare that rather than silently claiming a mode. The per-family notes are kept as
  the record of what each turned out to need — several contradict what the entry predicted before the work.

## alpha.135

- **Vulkan asks the driver how much VRAM is left.** `VK_EXT_memory_budget` reports, per heap, how much this process
  may still allocate and how much it already holds, both of which move as OTHER processes take and release memory.
  The extension was already detected and already enabled on the device; nothing queried it. Measured on this box:
  the driver reports **16711 MB free of 24810 MB**, where the old total-minus-our-own-blocks arithmetic would have
  claimed all 24810 MB — roughly eight gigabytes belong to other processes. Over-reporting free VRAM is how a
  planner OOMs a decode it was told would fit. The arithmetic stays as the fallback where the extension is absent.
- **`FreeMemoryBytes` is the free half of `GetVramInfo`, on the interface itself.** It was an independent `long
  FreeMemoryBytes() => 0`, and 0 is not "unknown" to its callers — it is "nothing fits", which kept every backend
  but CUDA on its smallest path forever. Fixing it on `IBackend` rather than on the shared GPU base means a backend
  that answers one answers both, including implementors and test doubles that never inherit that base. `CudaBackend`
  keeps its own override for now, reading the same driver call by a different route; it collapses onto the shared
  base with the rest of CUDA.
- **Zero free is an answer, not a missing one.** The first cut of the driver query returned "no answer" when every
  heap reported nothing left, which fell through to arithmetic that cannot see the process filling the card — an
  optimistic number at exactly the moment the honest one matters. It now reports the **largest single device-local
  heap's** remainder rather than the sum: no allocation spans two heaps, and some drivers expose a second
  device-local heap carved from the same physical memory, where summing reports twice what exists.
- **A live figure has consequences the constant did not.** A recipe that refuses to construct below a VRAM floor
  (`BooguImageRecipe`, `HunyuanVideoPipeline`, `DitShardPlanner`) now sees a number that moves with every other
  process on the card, so a co-tenant's spike can fail a request that would have succeeded a second later. That is
  the honest reading rather than a regression, but it is a behaviour change and single-sample preflights are now
  worth revisiting.
- **What that did NOT change, measured rather than assumed.** The engine's own text path preloads unconditionally
  (`TextService`), so Llama-3.2-1B on Vulkan was already resident: peak device memory 8491 MB before and 8483 MB
  after, 64 tokens in 8.84 s before and 8.90 s after. What the zero did suppress is the budget-aware preload inside
  `TextGenerationPipeline`, which only callers driving that pipeline directly reach, and the VAE decoder's full-res
  attempt — which stayed tiled here, so all three Vulkan byte-identity digests are unchanged.

## alpha.134

- **HunyuanImage 2.1 and MiniMax-H3 honour `(word:N)`**, taking the deliberately-unwired ledger from 4
  registered families to 2. Both are `CondScale`; what each needed was different, and in both cases the ledger's
  own prediction turned out to be wrong in an instructive direction.
- **HunyuanImage returns the weights at the sequence's REAL length, not padded to 1034.** Its encoder trims to
  the attention mask's real length, encodes that, then slices `[34, realLen)`. Right-aligning `realLen` weights
  against `realLen − 34` conditioning rows gives offset −34, so the template weights fall off the front exactly
  as SwarmUI's `pos = condLen − len(batch) + i` intends. Handing the padded array through would give offset
  `keep − 1034` and push every prompt weight off the front — a silent no-op that would have looked like coverage.
- **MiniMax-H3 needed less than the ledger predicted.** That entry said the weights would have to be a
  full-length array with every non-text `TagRun` forced to 1. Reading `MiniMaxH3TextEncoding.Build` showed the
  user prompt is appended LAST, after every condition label and vision block, so it is contiguous at the tail and
  a prompt-length array right-aligns onto exactly those rows. Its cond is already rank-2 F32.
- **A PRE-EXISTING tokenizer defect, found because the CPU test was written in the same commit rather than
  after it.** The first HunyuanImage prefix check asserted the encoder's hard-coded 34 template tokens and every
  test failed at 33. `Qwen2Tokenizer.EncodeRaw("\n")` returns ZERO ids where HF emits 198, so every newline in a
  chat template vanishes — `"user\n"` gives 1 id, not 2. HunyuanImage's BASE conditioning is therefore missing
  several ids ComfyUI feeds in, and the 34-token slice drops the prompt's first token's hidden state on top of
  that. Recorded as TODOs on the recipe and the tokenizer; unfixed, because the fix moves every existing
  HunyuanImage generation. The check now compares the split against what `EncodeChat` actually emits, which is
  the invariant that matters, instead of against a constant describing a different tokenizer.
- **The LTX-2 seam is located rather than guessed at, and the family stays unwired.**
  `LTXAVTEModel.encode_token_weights` (`lt.py:163-189`) runs the embeddings connectors ONLY under `compat_mode`;
  its default path returns token-length embeddings, which is why SwarmUI's unconditional right-alignment lands
  correctly there. Our pipeline mirrors `compat_mode`, so right-aligning would put every weight on REGISTER
  rows. Scaling `feats` before the connector is also wrong: `lt.py:174-176` normalizes by a global min/max over
  the whole sequence, so scaling one token moves the divisor every other token shares. The scale belongs inside
  `LtxVideo2TextConnectors`, after the projection and before the register concat.
- **Two pre-existing HunyuanImage checkpoint problems, recorded with evidence.** Its catalog asset
  `QuantStack/HunyuanImage-2.1-GGUF` no longer resolves on HuggingFace, and the safetensors build cannot be
  loaded at all: Comfy-Org's repack uses a `model.model.` prefix with DOTTED sub-modules
  (`img_attn.norm.query_norm.scale`) while the converter detects the fused underscore form and strips only
  `model.diffusion_model.`. The family loads from GGUF alone today.
- **Gate.** HunyuanImage is real-weight gated at 512², seed 1, on `svjack/HunyuanImage_gguf` Q4_0:
  `(fox:1.0)` byte-identical to plain, `(fox:0.5)` and `(fox:1.5)` both differing and differing from each other.
  **MiniMax-H3 is NOT gated** — its load was OOM-killed three times (exit 137), the last with 42 GB of host RAM
  free and nothing else large running, so the footprint is the family's own rather than contention. It carries a
  TODO naming the runs a capable box should do, including the same-code determinism control, which is not
  optional for this model because its output is only reproducible with the GPU otherwise idle.

## alpha.133

- **Every GPU backend gives device memory back when the engine asks.** `FreeActivations`, `TrimMemoryPool` and
  `FreeAllDeviceMemory` are empty-bodied defaults on `IBackend` that only CUDA implemented, so the ~25 engine call
  sites at generation, phase and model-swap boundaries did nothing on Vulkan and VRAM came back only when the GC
  reached each tensor's binding. They are implemented once on `GpuBackendBase` rather than a second time per
  backend, so ROCm and Metal inherit them.
- **The shared residency cache gains the one primitive it lacked**: release every unpinned activation without
  reading any back. That is deliberately neither `FreeAllCached` (drops resident weights too) nor
  `OffloadActivations` (pays a device-to-host transfer per buffer to preserve values) — an activation nobody has
  read is scratch, and a phase boundary reclaims it at the cost of recomputing it, exactly as
  `docs/Research/MEMORY_SCHEDULING_SERVING.md` describes.
- **A Vulkan bulk release now actually lands.** A deferred free is tagged with the tick the NEXT submit will take,
  and a bulk release is followed by no submit — so the timeline never reached that tick and the buffers stayed
  allocated until whatever work happened to come next. Releasing weights had the same shape. All three paths now
  drain the stream and take their frees immediately, which teardown already did for a different reason.
- **Two hooks carry what differs between backends.** `OnActivationsFreeing` runs before the release, where a
  backend invalidates anything that baked an activation's device address — a captured step graph holds the
  addresses of the buffers about to go, so replaying it afterwards reads memory the allocator has taken back.
  `OnAllDeviceMemoryFreed` is where a backend drops device memory its residency cache never owned.
- `TrimMemoryPoolCore` is abstract rather than a virtual no-op: a backend with nothing to trim should say so with
  an empty body, because inheriting silence here is how this whole set came to do nothing.
- **Vulkan's `ReleaseAttentionExecutionCache` is the drain half of its contract.** There is no plan cache to
  discard — attention is this backend's own shaders — but a caller reaching a phase boundary does need the
  previous phase's attention to have stopped reading the memory it is about to reuse.
- **The model-swap soak is no longer CUDA-only.** Its new cross-backend theory runs the shape a server actually
  runs — one backend, many models through it — and asserts the device memory comes back each time. Measured on both:
  cuda 61 MB and vulkan 8 MB unreturned across four swaps. What it does NOT do is hold this change: neutering the
  release calls and re-running leaves it green, because disposing a model disposes its tensors and each tensor's
  binding releases its own buffer. The regression is held by the Vulkan device tests, where 3 of 4 fail the moment
  the release path goes inert. That is the honest shape of the defect — bounded by tensor lifetime rather than an
  unbounded leak, with the loss being release at the named boundary the engine actually asks for it at.

## alpha.132

- **Seven more families honour `(word:N)`** — HiDream, OmniGen2, Lumina-2, Kandinsky5, Kandinsky5-Video,
  HunyuanVideo and the SDXL refiner — taking the deliberately-unwired ledger from 11 registered families to 4.
- **HiDream blends its T5 and Llama arms.** The Llama side is blended ONCE, before `SliceLastDimIntoChunks`:
  ComfyBlend broadcasts across the last dimension, so a single call covers all 48 layer slices. Its conditioning
  cache key carries the weights through `ConditioningCacheKey`, because the emphasis is stripped before
  tokenization — `(cat:1.5)` and `cat` produce identical ids, so an id-only key serves the wrong tensor on the
  second generation of a weighted prompt.
- **OmniGen2, Lumina-2 and HunyuanVideo are the first variable-length baselines.** Every ComfyBlend family wired
  before them padded to a fixed window, which makes one empty encode valid for any prompt. None of these three
  pads, so the baseline is rebuilt at each prompt's own length — one extra encoder forward per uncached prompt,
  not avoidable by caching a longer baseline and slicing, because the encoders are causal.
- **Two ComfyBlend baselines were wrong and are corrected.** They were written from the plausible assumption that
  an empty baseline means "the family's own template with an empty prompt". ComfyUI's `gen_empty_tokens`
  (`comfy/sd1_clip.py:15-25`) emits `start + end + padding` from each model's declared `special_tokens`, and both
  of these declare only a pad: OmniGen2's `Qwen25_3BModel` is `{pad: 151643}` and Anima's `Qwen3_06B` the same
  shape, so each baseline is padding alone. Anima's `Encode("", appendEos: true)` was additionally putting an EOS
  at row 0 — and row 0 is a prompt position, so that row IS read whenever the first word is weighted.
  **Anima shipped in alpha.131, so this changes its weighted output; `w=1.0` is unaffected**, because a baseline
  is never read when nothing is weighted.
- The rule that makes these tractable, now stated where the code can be checked against it: ComfyBlend only
  rewrites rows whose weight is not 1, and pad rows always weigh 1, so a baseline's padding region is never read.
  Pad-id conventions between us and ComfyUI are therefore immaterial. Row 0 is not.
- **HunyuanVideo blends the full sequence and crops afterwards**, which is the order `encode_token_weights` uses
  (`hunyuan_video.py:104-110`) — the crop count is computed from the already-encoded output there. Only the Llama
  arm blends; CLIP-L contributes a pooled vector and just needs the grammar taken off. Its hard-coded 95-token
  `CropStart` is now checked against the tokenized template instead of trusted: a tokenizer revision that moved
  the template's length would crop into the prompt and still render something plausible.
- **HunyuanVideo's text encoder was configured 64 vocabulary rows short of its checkpoint, and the gate is what
  found it.** It reused `LlamaStyleEncoderConfig.Llama31_8B` (vocab 128256), but
  `Comfy-Org/HunyuanVideo_repackaged`'s `llava_llama3_*.safetensors` ships `model.embed_tokens.weight` with
  **128320** rows — the LLaVA fine-tune adds its own specials. Harmless for prompts, since every id a caption
  produces is below 128256, but it made ComfyUI's declared pad token 128258 unreachable and the weighted run
  died with "Token id 128258 is outside vocabulary size 128256". A new `LlavaLlama3_8B` config carries the real
  size; `Llama31_8B` is left alone because HiDream's `llama_3.1_8b_instruct_fp8_scaled` really is 128256.
- **A correction to how the ComfyBlend baseline rule was stated in alpha.130-131.** "Pad rows always weigh 1, so
  the baseline's padding is never read" holds only for a FIXED-WINDOW family, where the conditioning's pad rows
  line up with the baseline's. On a variable-length family the baseline is start-plus-pad at the prompt's own
  length, so its pad rows sit directly under WEIGHTED prompt rows and are read every time. The pad id is
  load-bearing there, which is why the short vocabulary above was a hard failure rather than a nuance.
- **Kandinsky5 and Kandinsky5-Video declare the mode in order to apply NOTHING, and that is parity rather than a
  gap.** SwarmUI's probe puts them on ComfyBlend because CLIP-L keeps weights, but
  `Kandinsky5TEModel.encode_token_weights` (`kandinsky5.py:39-43`) returns the Qwen cond plus CLIP-L's POOLED
  vector and discards the blended hidden states. What they owe is the STRIP — declaring the mode is what stops
  the service collapsing the tag, so without it the parens would reach Qwen as prose. Blending the Qwen arm to
  "fix" the no-op would BREAK parity, not achieve it.
- **The SDXL refiner needed a batched weighted CLIP encode.** `EncodeWeightedPenultimate` concatenates its inputs
  along the SEQUENCE axis as 77-token chunks of ONE prompt, so handing it a (negative, positive) batch would
  splice the negative onto the end of the positive. `EncodeBatchWeightedPenultimate` keeps each prompt in its own
  batch row, skips the baseline encode entirely when nothing is weighted, and leaves the pooled vector unweighted
  the way the reference does. Chunk 0 only; a multi-chunk weighted prompt (`<break>`) is not wired.
- **Lumina-2's per-span ids drop the SentencePiece BOS.** ComfyUI tokenizes each word alone from `tokens_start=1`
  and prepends the start token once per batch; keeping it per span would put a stray sentence start in the middle
  of the caption.
- An `AuraFlow` `TODO` records a suspected PRE-EXISTING defect found while reading `gen_empty_tokens`, and
  deliberately not fixed here: ComfyUI's `aura_t5.py` declares `special_tokens={"end": 2, "pad": 1}` and
  `pad_token=1`, while `T5Tokenizer` fixes EOS=1/PAD=0 for every T5 family it serves. Unverified against the
  Pile-T5 vocab; it would fail as plausible output rather than an error, and a fix moves every existing AuraFlow
  generation.

## alpha.131

- **Flux.1, Anima and SD3 honour `(word:N)`**, taking the deliberately-unwired ledger from 14 registered
  families to 11. Both are dual-encoder families where only ONE arm can be blended, and wiring the other would have been
  a no-op that looked like coverage.
- **Flux.1: the T5 arm only.** CLIP-L contributes its POOLED vector and its hidden states are disposed
  immediately after the EOS extraction — ComfyUI's blend rewrites hidden states, so there is nothing on that arm
  for it to act on. Flux applies no trim to its T5 output, so it takes AuraFlow's shape rather than Chroma's: the
  cache keeps the PLAIN conditioning, which still shares the baseline's shape, and the blend lands on a
  per-request copy after the fetch.
- **Anima: the Qwen-3 arm only.** Its T5 side is an id lookup inside the adapter (`embed[t5_ids]`), not an
  encoder output, so there is nothing to interpolate — that text only needs the grammar taken off, or the parens
  reach the lookup as prose. ComfyUI's own Anima text encoder drops that arm's weights the same way. The blend
  lands on the full padded window before the real-length slice, and the Qwen-3 padding is mirrored rather than
  assumed: EOS then BOS-as-pad, since Qwen3 has no dedicated pad token.
- **SD3 blends all three arms, and it is the only wired family that has to.** Everywhere else CLIP contributes
  a pooled vector and its hidden states are discarded, so only the T5/LLM arm is blendable; SD3's CLIP hidden
  states reach the DiT. The CLIP halves reuse `ClipTextEncoder.EncodeWeightedPenultimate`, which builds its own
  empty chunk and returns an UNWEIGHTED pooled from chunk 0 — the reference's own behaviour, not a convenience:
  the blend rewrites hidden states only and `first_pooled` is read before its loop, so the pooled vector must not
  carry the emphasis. Every arm is a fixed window (77/77/256) and SD3 caches no conditioning, so there is neither
  a per-prompt baseline nor a cache key to make weight-aware. CLIP chunk 0 only, which is what the single-array
  signature carries and what the plain encode already produced; a multi-chunk weighted prompt (`<break>`) would
  need the chunked signature and is not wired.
- Flux.1's regions weight per leaf and a base prompt weighted alongside one is refused — the fourth family to
  need that split, reusing `RegionalPromptWeightSplit` rather than reimplementing it.

## alpha.130

- **Nine more families honour `(word:N)`**, taking the deliberately-unwired ledger from 22 registered families
  to 14.
- **The four Wan variant recipes: Wan-Animate, Wan-Animate-2, Wan S2V and Wan VACE.** Each is a separate recipe
  class from `WanVideoRecipe` with its own prompt path, which is why declaring the mode on the parent never
  reached them. That is the whole Wan family wired.
- **The blend is hoisted into `VideoRecipeUtils` rather than copied four times**, and `WanVideoRecipePipeline`
  delegates to it, so there is one definition of how this family weights a prompt instead of five that can drift.
- Wan-Animate-2 carries a third conditioning stream — the driving clip's own prompt — and it is weighted as its
  own leaf, which is what SwarmUI's `encode_leaves` does per leaf. Its empty baseline rides the same four-row
  batch, for the reason the two-stream form already gave: the baseline has to share the padding and the layer
  selection exactly, and a second pass cannot guarantee that.
- umT5 pads to a fixed window, so ONE empty encode matches any prompt's shape. That is what makes ComfyBlend
  cheap for this family, and it is not true of every ComfyBlend family — a family whose conditioning length
  tracks its prompt needs a per-prompt baseline instead.
- **Four ComfyBlend families follow: AuraFlow, LTX-Video, Chroma and Chroma-Radiance.** `T5WeightedConditioning`
  is to the T5-style encoders what `WeightedConditioning` already is to CLIP, and it owns the two things a
  ComfyBlend caller gets wrong. First, the empty baseline is subtracted row by row, so it must match the
  conditioning's shape exactly — a tokenizer that pads to a fixed window makes that free, and every family here
  has one. Second, a conditioning cache keyed on token ids will serve the wrong tensor: once the recipe has taken
  the emphasis off the text, `(cat:1.5)` and `cat` tokenize identically, so the second generation hits the cache.
- **The four split into two opposite caching strategies, and the trim is what decides it.** AuraFlow has none, so
  its cached conditioning still shares the baseline's shape: it caches the PLAIN encode and blends a per-request
  copy after the fetch. Chroma and Chroma-Radiance trim to the prompt's kept tokens before caching, so the
  full-window baseline is no longer subtractable from what they stored — they blend before the trim and carry the
  weights IN the cache key instead.
- LTX-Video's baseline rides the same three-row encode batch as its prompt and negative, and the blend lands at
  the full 128-token window before the pad drop. Both are the same rule: the weights describe the padded rows,
  and the baseline is only subtractable while both still have them.
- **A disposal bug the unit tests could not have found.** AuraFlow's first weighted generation died with "Cannot
  access a disposed object": the baseline was being released inside the cache-eviction block, immediately after
  being encoded, because that is where the sibling `_cachedUncond?.Dispose()` lives. `w = 1.0` passed throughout —
  only a weighted run reaches it. It belongs in `DisposeCore`, which `AuraFlowPipeline` did not override at all,
  so its two existing cached tensors were already leaking; the override now covers all three.
- The token/weight pairing is now unit-tested once for all five recipes. The invariant worth naming: EOS and pad
  rows weigh exactly 1. They are not part of the prompt, and blending them would pull the padding toward the
  empty encode along with the words — a whole-sequence drift that reads as the weighting being far too strong.

## alpha.129

- **Six more families honour `(word:N)` prompt weighting: Z-Image, Boogu, Zeta-Chroma, Lens, ERNIE-Image and
  Ideogram 4.** All six disable weights in their ComfyUI tokenizer, so the mechanism is `CondScale` — the prompt
  is encoded at weight 1 and each token's conditioning row is scaled afterwards, right-aligned so a trimmed
  template prefix takes its weights with it. That leaves 22 registered families deliberately unwired, each with
  its reason recorded in `PromptWeightingModeLedgerTests`.
- **`TemplatedPromptTokens` is the decision these recipes would otherwise each have to remember.** A chat-template
  renderer usually BPEs the prompt together with the text immediately before it — `Qwen3Tokenizer` concatenates
  `"user\n"` with the prompt in ONE call, deliberately, so that a prompt beginning with whitespace merges its
  newline with the template's. Tokenizing the prompt on its own puts a pre-tokenization boundary there and can
  produce different ids for the same text. SwarmUI has the same property and accepts it, because splicing
  template ids around a separately tokenized leaf is what `calc_leaf` does — but only where a weight actually
  exists. An unweighted prompt keeps the family's own encode, so wiring weighting moves no existing generation.
- **A weight of exactly 1 is stripped, not encoded.** Caught by the Z-Image real-weight gate rather than by any
  test: the unweighted path was byte-identical before and after the change, yet `(fox:1.0)` was not byte-identical
  to plain `fox` — it was reaching the encoder as literal parens. Flattening does not remove them; it rewrites
  SwarmUI's `<weight[N]:>` tag, while a literal `(word:N)` typed at a CLI arrives untouched. It is the first thing
  a weighting gate checks and the first thing a user tries, so it is now a unit test as well.
- **Every family scales a per-request COPY.** Four of the six cache conditioning across generations keyed on token
  ids, and a weighted prompt tokenizes to the same ids once the grammar is off — scaling in place would hand the
  next plain request the previous one's emphasis. Boogu's cache is keyed on the prompt string instead, which
  separates the two, but a repeat of the SAME weighted prompt would still compound.
- `WeightedTokenSequence.Wrap` composes with `Truncate` for a tokenizer that caps the TEXT and keeps its specials:
  `ErnieTokenizer.Encode` reserves room for BOS/EOS, so building the whole sequence and cutting the tail would
  drop the terminator the encoder expects.
- `GptOssTokenizer.ChatTemplateIds` splits the Harmony wrapper where the prompt sits. Splitting is exact there,
  unlike the Qwen3 template, because `EncodeWithSpecials` already breaks its plain-text runs at every marker. The
  prefix is checked against the fixed 97-token offset the encoder strips — the first real run reported `97+22`
  against an assertion that expected the two halves to sum to 97, which is what established that the constant
  counts the prefix alone and the suffix after the prompt is retained.
- Lens passes the FULL weight array rather than a sliced one: `ScaleRightAligned` right-aligns, so the 97 entries
  covering the stripped prefix land at negative positions and drop out by themselves.
- Z-Image and Ideogram 4 weight their regions per leaf and refuse a base prompt weighted alongside one, because
  the base encode covers the region tags and the two sets of weights would land on the same conditioning rows.

## alpha.128

- **Krea 2 honours `(word:N)` prompt weighting, both halves of it.** Krea 2 is the one family whose SwarmUI
  workflow inserts `SwarmAttnTokenWeights` on top of the ordinary cond scaling (`WorkflowGenerator.cs:965-972`),
  so the two mechanisms are not alternatives here — declaring only the first would look like parity while
  under-emphasizing every weighted word. The mode ledger refused the partial declaration, which is what it is for.
- The cond-scale half encodes the prompt at weight 1 and then scales each token's conditioning row, right-aligned
  so a dropped template prefix takes its weights with it. It runs on a per-request COPY: the prompt-embedding
  cache is keyed on token ids alone, and a weighted prompt tokenizes to the same ids, so scaling the cached tensor
  would hand the next plain request the previous one's emphasis.
- The attention half splits by direction the way `SwarmText.py:281-312` does, because the two are not the same
  operation scaled differently. Below 1 multiplies that token's attention VALUE rows post-projection — it removes
  what the token contributes. Above 1 adds `(w-1)*2` to every query's logit for that KEY position — it makes the
  other tokens look at it harder. Neither is expressible as the other.
- Cond slots only, mirroring the patch's own `cond_or_uncond` filter: Krea 2's negative pass is a separate forward
  that already runs unbiased, so passing the scale on the cond call alone IS that filter rather than an
  approximation of it. Text leads Krea 2's joint concat, so a conditioning row index is already a joint-sequence
  index — the same layout SwarmUI's `seq == img_slice[1]` guard establishes before it applies anything.
- The value scale is an elementwise multiply against a buffer expanded once per forward and shared by all 28
  blocks, not a per-row op: `MaskRows` is F32-only and the DiT activation is F16 on the fast path. The key bias is
  a `[1,1,1,Skv]` additive mask, which SDPA broadcasts over every query without materializing the Sq x Skv
  duplicate. Like the regional bias, a live weight excludes the step cache, the captured graph and DiT sharding.
- **A weighted region works; a weighted base prompt alongside one does not, and says so.** A region is its own
  leaf — which is what SwarmUI's `encode_leaves` does per region — so it gets the cond-scale half; the region
  already owns the attention bias, leaving no slot for the patch. But the base encode covers the region tags too,
  so weighting BOTH would land two sets of weights on the same conditioning rows; that is refused by name. With
  regions present the base drops the weight grammar in both its spellings (SwarmUI's `<weight[N]:>` tag and the
  literal `(word:N)` a CLI user types), so its token ids are what they were before this change.
- **img2img and masked inpaint with attention weights are refused too**, because those run the pixel-space route,
  which reaches the transformer with no bias surface at all. Both of these read as "emphasis is weak" when they
  fail quietly, which is why neither is a warning.
- `ModelSpecificEnhancements` is now a request field, defaulting on to match
  `UserInput.Get(T2IParamTypes.ModelSpecificEnhancements, true)`, with `hartsy image --no-model-enhancements` to
  turn it off.

## alpha.127

- **`AffineMix`, `FillBias` and `PixelShuffle2d` run on Vulkan.** All three were host fallbacks costing a device
  round-trip per call.
- `AffineMix` is one kernel rather than scale-scale-add because the intermediates are activation-sized: composing
  it costs three full reads and three writes where the fused form costs two reads and one write.
- `FillBias` writes zero when there is no bias rather than treating that as a caller's problem — an unbiased
  convolution still needs its output cleared before the taps accumulate into it. The absent case binds the output
  buffer and turns the read off with a spec constant, because a null binding is not legal and an unbound slot
  reads whatever bound it last.
- `PixelShuffle2d` is driven from the output, one invocation per output element, so writes are contiguous. Its
  channel packing is `(c·r + p1)·r + p2`; read the other way round it produces a plausible image with the
  sub-pixel grid scrambled, so the tests include a ratio above 2 — at r=2 with few channels the two orders can
  coincide.
- `PixelShuffle2d`'s reference no longer drains the whole device before reading. Reading `DataPointer` forces that
  tensor's own lazy sync, which is the dependency that matters; the full drain was wider than the op needs.

## alpha.126

- **`GeluErf`, `GegluErf`, `Mish` and `Prelu` run on Vulkan.** All four were host fallbacks, so each cost a device
  round-trip per call. Three needed no new kernel: the elementwise shader already computed exact-erf GELU for an
  op code nothing dispatched, and Mish is one more code beside it.
- `GegluErf` is the existing GEGLU kernel with a spec constant selecting the exact GELU over the tanh
  approximation. The two are not interchangeable — they agree near zero and diverge by about 1e-3 at the tails,
  which is visible in a DiT's output — so which one a family wants is a real choice rather than a detail.
- `Prelu` gets its own kernel because its slope is indexed by channel. The elementwise path carries one scalar in
  its push block and has no notion of the `[B, C, T]` layout; a single-element alpha is the shared-slope case and
  falls out of the same indexing.
- Mish's softplus is written `log1p(exp(-|x|)) + max(x, 0)`, the form that does not overflow `exp` before the log
  for a large positive x. The tests feed +/-12 for that reason, and because the middle of the range is exactly
  where a kernel wired to the wrong GELU still looks correct.
- **Removed an op that does not exist.** An earlier pass added a Vulkan `Erf(output, input)`; `Erf` on `IBackend`
  is a private scalar helper, not a tensor op, so that method implemented nothing. Its elementwise op code went
  with it.

## alpha.125

- **Quantizing a tensor that does not fill whole blocks corrupted the heap.** A block quant stores a fixed
  element count per block and nothing enforced it: H3's `adaln_t_table` is `[8, 1025]`, eight elements short of
  the 33rd Q4_K block, so the buffer was sized by integer division at 32 blocks while the codec wrote 33. The
  only guard was a `Debug.Assert`, which Release compiles out, and the overrun surfaced as an allocator abort in
  unrelated code. Such a tensor is kept at F16 now, and `ComputeByteCount` throws rather than under-allocating.
- **The GGUF write streams.** Each tensor is widened, quantized and its wide copy freed before the next, instead
  of holding the whole checkpoint as F32 — which wanted 74 GiB for a 13 GB source and was OOM-killed. The
  working-set refusal no longer applies to GGUF targets because they no longer need it.
- **A profile sidecar for an artifact declaring no step count was written with `Steps = 0`, which the resolver
  refuses outright** — the file cost a full hash of the output and was then thrown away. It carries the base
  count explicitly, because the format has no way to say "use the recipe's".

## alpha.124

- **`UnpatchifyTokens` runs on Vulkan**, the last of the six true host fallbacks the Flux path reaches. Every one
  of them is now a kernel: `QkvSplitNorm`, `LayerNormModulate`, `ApplyRopeSingle`, `ModulationSplit4`,
  `AffineBroadcastRowIndexed`, `GatedResidualRowIndexed` and this. What is left on that path is `LinearMulti`,
  whose default composes GPU ops and is correct, just extra dispatches.
- Driven from the OUTPUT: one invocation per output element, each computing the single source it reads. Driving it
  from the input would have each invocation write patch·patch·C scattered locations — the same work with none of
  the coalescing.
- It calls the same `PatchTokenContract.ValidateUnpatchify` that CUDA and the host default use, rather than
  re-deriving the geometry. Getting that wrong produces a plausible image with the patch interior transposed,
  which is the kind of wrong that survives a smoke test.
- Both packings are covered by tests because both are in use and neither is a default, along with a non-square
  packed grid (an h/w swap looks fine on square) and patch=1. A shuffle moves values without arithmetic, so the
  assertion is exact equality rather than a tolerance.

## alpha.123

- **`ModulationSplit4`, `AffineBroadcastRowIndexed` and `GatedResidualRowIndexed` run on Vulkan.** All three were
  host fallbacks reading `DataPointer`, so each cost a device round-trip per call on the DiT path. With these,
  five of the six true fallbacks the Flux path reaches are implemented; `UnpatchifyTokens` is the remaining one.
- The `1+x` on scales and `tanh(x)` on gates are stated in the shader rather than assumed, because they are not
  interchangeable: a scale is a residual around identity so zero must mean "leave it alone", while a gate is
  bounded so a block can be turned off smoothly. Swapping them yields plausible output that is wrong everywhere,
  which is why the test checks all four outputs rather than the first.
- An absent shift table binds the scale table and turns the read off with a spec constant. A null binding is not
  legal in Vulkan, and leaving the slot unbound would read whatever bound it last — the failure the dispatch
  guards now catch, avoided here by construction.
- The row-indexed tests use a deliberately non-identity, repeating index. An implementation that ignored the
  gather and read row *r* of the table passes an identity index and is wrong for every real call, since the whole
  point is that many rows share few modulation vectors.

## alpha.122

- **`LayerNormModulate` and `ApplyRopeSingle` run on Vulkan.** Both were true host fallbacks on the DiT path, not
  compositions: each reads `DataPointer`, so every call meant a device round-trip. Every DiT block modulates right
  after normalizing, and applies rotary to q and k, so both ran once per block per step.
- **Correcting the previous entry's count.** It said seven of the eight Phase-3 ops the Flux path calls were GPU
  composition and only `QkvSplitNorm` left the device. Reading the defaults properly rather than their opening
  lines, **six of the eight are host fallbacks** — `LayerNormModulate`, `ModulationSplit4`,
  `AffineBroadcastRowIndexed`, `GatedResidualRowIndexed`, `UnpatchifyTokens` and `ApplyRopeSingle`. Only
  `LinearMulti` composes. The remaining work on that path is larger than the earlier note implied.
- `LayerNormModulate` keeps the `(1 + scale)` convention explicit in the shader, because it is load-bearing: scale
  is a residual around identity, so a zero modulation must leave the normalized value alone rather than zero it.
- `ApplyRopeSingle` handles partial rotary in the same kernel as full — cos/sin keep the full headDim stride either
  way, and only the first `rotaryDim` dims rotate.
- **An in-place op still has to re-cache its buffer**, and the parity test is what said so. The first version
  dispatched, wrote the device buffer and returned without rebinding, so a later host read got the untouched host
  copy and the op looked like it did nothing — a 3.3 absolute error against the reference, versus 2e-7 after.


## alpha.121

- **The fp8 quantize path leaked every weight it wrote.** It worked out what it owned by rescanning the whole
  output after each tensor, which re-registered earlier companions on every later pass and missed the fp8 weight
  itself — that one reuses the source's key, so a "not already in the source" test excluded it and nothing freed
  it. It now diffs against the keys present before the call.
- **The quantizer's memory refusal measured the wrong number.** `TotalAvailableMemoryBytes` is total physical RAM
  where no cgroup limit applies, so it reported 62 GiB "available" on a machine with 54 free — a job could pass
  the check and still be OOM-killed, which is the failure the check exists to replace. It reads `MemAvailable` on
  Linux, and passing it is documented as necessary rather than sufficient.
- **Prompt weighting reached Wan's streaming path.** That entry point encoded its own prompt and never got the
  blend, so `(word:1.5)` meant one thing batched and another streamed. Both share one encode now.
- An int8 convolution weight reaching the LoRA merge hit ConvRot's internal rank-2 assertion — reachable only
  because convolution targets were allowed through at all. Refused by name instead.

## alpha.120

- **`QkvSplitNorm` runs on Vulkan**, the one op on the Flux/DiT path whose host default was a TRUE fallback rather
  than composition. It reads `DataPointer` on six tensors, so every call meant a device-to-host sync, a scalar loop
  over every token and head, and an upload of the results. Flux runs 19 double plus 38 single blocks per forward,
  each calling it once per step, so the round-trip was paid 57 times a step.
- Chosen by measurement rather than by working down the list: of the eight Phase-3 ops the Flux path calls, seven
  have defaults that compose other GPU ops — correct, just extra dispatches — and this was the only one that left
  the device.
- **The cross-subgroup fold is not optional here even though the reduction is only headDim wide.** `subgroupAdd`
  reduces within a subgroup, and subgroup width is hardware: 32 on NVIDIA, 64 on AMD, as low as 8 on Intel. A
  workgroup spanning more than one would have normalized by the wrong denominator and still produced plausible
  output — the failure worth spending shared memory to avoid, and the one that would have shown up first on the
  AMD card this work is for.
- Verified against the CPU reference at four head geometries chosen to straddle every plausible subgroup width,
  including a 256-wide head that spans several. Max absolute error 4.8e-7 on q and k; `v` is a straight copy and is
  exact. And on a real generation — Flux.1-dev fp8, 512x512, 4 steps, seed 42 — the decoded image is byte-identical
  to the same run through the host fallback.

## alpha.119

- **`LayerNormNoAffine` runs on Vulkan.** It had no override there, so every call fell through to `IBackend`'s host
  default — which refuses anything but F32 outright. That is what a Flux generation on an AMD card hit: the DiT
  runs its block activations in F16 by default (`numerics.ditF16`), so the pre-modulation norm handed the default an
  F16 tensor and the generation failed with "LayerNormNoAffine default fallback only supports F32". Every DiT
  normalizes before modulating, so the op is on the hot path of the whole family, not a corner of it.
- The shader is deliberately its own kernel rather than `layernorm` with an identity weight. The identity would
  cost a per-element multiply-add and, more to the point, two device buffers that do not exist at the call site —
  every caller would have to allocate and fill them per call.
- `IBackend.LayerNormNoAffineReference` joins the other reference statics, so the host default and any backend
  falling back share one implementation. An override cannot reach the default through `((IBackend)this)`: that
  re-enters the override and recurses until the stack ends.
- Verified against the CPU reference at four shapes, including a row count that is not a multiple of the workgroup
  and a dim that is not a multiple of the subgroup, since the cross-subgroup fold is where a norm like this goes
  wrong on small-subgroup hardware. Max absolute error 1.4e-6 in F32, 4.9e-4 in F16 — the latter being F16's own
  precision rather than a disagreement.


## alpha.118

- **A quantized video build keeps the semantics of the build it came from.** Video planning resolves by exact
  file hash, so a checkpoint quantized here is a stranger to it — an H3-class output planned as an unknown base
  and the task, acceleration and step count its source declared were simply gone. `QuantizationService` writes
  `<output>.hartsy-video-profile.json` when the SOURCE's hash resolves to a known artifact, and writes nothing
  when it does not: inventing provenance for a file nobody has verified is worse than leaving it unknown.
- **Quantizing a large block-quantized source now refuses instead of being OOM-killed.** Every tensor is widened
  to F32 before the writer sees it, so requantizing a 13 GB Q4_K checkpoint wants about 74 GiB — and the process
  died with no message, no partial file and nothing to read. It measures that from the real element counts and
  says so up front, with the numbers.

## alpha.117

- **Twenty-seven Vulkan ops were dispatching outside an op scope.** The scope is what suppresses the batched
  auto-flush, so without one a flush could land between two dispatches of the same op and free the transients the
  later dispatches still read — the per-slice loops are exactly that shape. `Silu`, `Add`, `RmsNorm`, `LayerNorm`,
  `Concat`, `Split`, `BatchedMatMul` and twenty more were missing it, including the internal entry points the
  coopmat benchmarks call directly. They were not failing yet because the flush threshold is eight dispatches.
- **The dispatch path checks three correspondences that were maintained by hand and verified by nothing.** The
  storage-buffer count against the count the kernel was built for, the push-constant size against the range every
  layout reserves, and that an op scope is open. Each failed silently when wrong: a short buffer list leaves the
  remaining bindings pointing at whatever bound them last, an over-long push block is truncated at the layout, and
  the flush hazard above. Turning the check on is what found the twenty-seven.
- `PushConstants` builds a push block by appending, so byte offsets are not written by hand. An op used to
  `stackalloc byte[9 * 4]` and write nine offsets that had to agree with a GLSL struct in another file and with
  each other; inserting a field meant renumbering every line below it, and an error does not fail — the shader
  reads a plausible number from the wrong place. It is a MUTATING ref struct deliberately: chaining by value would
  copy it, and every field would land at offset zero.
- `VulkanDescriptorManager.PushConstantRangeBytes` names the 128-byte floor that was a bare literal in the layout
  and an implicit assumption in every op's `stackalloc`, with nothing connecting the two.
- A source lint enforces both rules without a GPU: no override reaching its own interface default (which recurses
  to a stack overflow), and every dispatching op opening a scope. The runtime guard only fires on a path some test
  runs, and Vulkan's least-covered ops are the ones most likely to be written next.

## alpha.116

- **`(word:1.5)` works on Wan.** umT5 keeps token weights, so Wan is a ComfyBlend family: the prompt is encoded at
  face value and the output blended toward the empty-prompt encode, `z = (z − z_empty)·w + z_empty`. The baseline
  is encoded in the same batch as the prompt and the negative, because it has to share their padding and layer
  selection exactly. The negative is weighted too, which is what ComfyUI does.
- The emphasis grammar comes off the text whether or not it does anything. Declaring a weighting mode is what
  stops the service stripping `(word:N)` upstream, so a weight of exactly 1.0 would otherwise have reached umT5
  as literal parens and digits — and differed from the same prompt written plainly.

## alpha.115

- **The engine asks a backend what it can do instead of what class it is.** Every `is CudaBackend` outside the CUDA
  package is gone. They were not stylistic: each one silently denied a capability to any backend that was not CUDA,
  so a second backend inherited the restriction whether or not it applied.
- `dequantizeToF32` is `!Capabilities.SupportsQuantized` in both `TextService` and `ModelManager`. Equivalent today
  — CUDA publishes true, Vulkan and CPU false — but it means Vulkan stops paying an F32 expansion the moment it
  publishes the capability, rather than forever because it is not CUDA.
- `PreloadWeights` is called unconditionally. It is a no-op on a backend with no device memory, and a backend that
  HAS device memory wants its weights resident; gating on the class meant Vulkan re-uploaded every weight over PCIe
  on every op. Same for `FreeAllDeviceMemory` on unload, which simply never ran on Vulkan and left VRAM held.
- The Ideogram 4 and Boogu VRAM preflights use `GetVramInfo()` and `StreamingCache`, so they now apply on any
  backend that reports memory rather than being skipped entirely off CUDA — which is what made them silent there.
- **Two of these change behaviour rather than just spelling, and deliberately.** SeedVR2's BF16 VAE activations
  were keyed on "is CUDA" and are now keyed on `SupportsBF16`, which on CUDA is compute capability 8.0 or newer —
  so a pre-Ampere card gets F32 activations where it previously got BF16 it has no hardware for. MiniMax-Music3's
  half-precision KV moves from "is CUDA" to `SupportsF16` on the same reasoning.
- Left alone on purpose: `ValidateShardDevices` still requires a CUDA device for LLM layer-split. The real
  requirement is a backend that computes on quantized tensors, and Vulkan does not yet — relaxing the check to any
  GPU kind would admit it to a path that would fail further in. It changes when Vulkan publishes the capability.

## alpha.114

- **`POST /admin/models/quantize`** exposes offline quantization over the API, with the same formats and presets
  the CLI takes so a caller does not have to learn two vocabularies for one operation.
- It is synchronous on purpose. A caller that gets a 200 has a finished file on disk; a multi-GB write that
  reported success before it was durable is the sort of thing nobody notices until the file is loaded.
- Validation happens before the filesystem is touched, so a bad format or preset is a 400 rather than a
  partially-written file.

## alpha.113

- **The op scope every GPU backend needs is written once.** `GpuBackendBase` owns it: the point at which a backend
  runs the device cleanup a finalizer could not, and reclaims what the previous op displaced. Both backends had
  written it separately, and it is easy to get subtly wrong because it is defined by what has ALREADY happened
  rather than by what the op is about to do. Nesting is handled there too — an op built out of other ops must not
  sweep in the middle of itself, since the buffers it would reclaim are the ones its own later dispatches read.
- The `IBackend` members that are simply the residency cache live there too: the D2H counters, preload, free,
  pin/unpin and bulk offload. Expanding low-rank adjuncts before a preload or free is part of that — both backends
  did it, because a weight carrying an adjunct is read as its factors, so preloading the weight alone leaves the
  factors to miss one at a time on the hot path.
- **Vulkan reports VRAM.** Every memory decision on that backend logged "no VRAM report" and fell back to fixed
  budgets, because `GetVramInfo` did not exist there. Total now comes from the device's DEVICE_LOCAL heaps, which
  is exact; free is total minus what this backend's allocator holds, which underestimates what the card has left
  because another process is invisible to it. Reported anyway: the alternative was no number at all. A live figure
  from `VK_EXT_memory_budget` is the replacement, and the capability is already detected with nothing querying it.
- Profiling stays a hook rather than a shared object. CUDA's is NVTX ranges and Vulkan's is `VulkanProfiler`;
  unifying those is its own change, and all the scope needs is whether to time an op and somewhere to hand the
  answer.
- **CUDA does not derive from the base yet, and that is deliberate rather than unfinished-by-accident.** Converting
  its 181 op entries to the shared scope makes three `CudaGraphTests` fail at teardown with
  `CUDA_ERROR_INVALID_VALUE`, freeing an activation whose address the driver already reclaimed with a directly
  constructed `CudaGraph`. Bisected: the base plus the Vulkan adoption is clean, disposal routing is not the cause,
  and forcing the old always-enter behaviour on nested scopes does not fix it. Landing the half that is proven
  beats landing a regression next to it.

- **A LoRA can be merged into a convolution weight.** SD1.5 and SDXL UNets are mostly convolution, so a LoCon or
  LyCORIS adapter targeting conv layers had its deltas silently skipped — counted as unmatched and stepped over.
- The delta arrives flattened to `[out, in*kh*kw]` while the weight is `[out, in, kh, kw]`. Row-major those are
  the same bytes in the same order, so the previous rank-2-only gate was conservative rather than necessary; the
  add only needed the two shapes to agree.
- **DoRA on a convolution is refused by name.** Its magnitude vector normalizes by a row norm of the weight as a
  matrix, and a convolution has no such matrix until it is flattened — which axis the vector then describes is the
  file's choice, not ours.

## alpha.112

- **`hartsy quantize --format` writes ComfyUI's two safetensors quant shapes as well as GGUF.** `fp8-scaled`
  stores each eligible weight as F8E4M3 beside the scalar it was divided by; `int8-convrot` stores it as I8 beside
  a `[rows,1]` row scale and a `.comfy_quant` descriptor.
- Two details that exist only because interoperating with published files is the point. The row scale is written
  `[rows,1]` rather than the codec's flat `[rows]`, which is the shape real repacks carry — our own reader goes by
  element count either way. And the descriptor uses ComfyUI's key names, where `convrot` gates `convrot_groupsize`:
  a group size written without that flag reads back as zero.
- Verified by loading what we wrote back through the container and checking the `QuantInfo` it attaches, rather
  than only that the bytes parse.

- **CUDA's residency cache is the shared one now.** `GpuTransferHelper.State` derives from
  `GpuResidencyCache<ulong>`, so the weight/activation/cast collections, the four-step rebind, weight demotion,
  the keyed bindings, orphan parking, bulk offload and teardown exist once instead of twice. Every static signature
  is unchanged — the 1399 call sites in `CudaBackend` did not move — and what is genuinely CUDA's rides the
  overrides: the graph-capture arena in `AllocateDevice`, the persistent weight allocator in `AllocateWeight`, the
  stream-ordered copy and H2D profiling in `Upload`, Q8_1 sidecars in `OnActivationEvicted`, auto-promotion in
  `TryMakeResidentOnMiss`, and re-promotion blocking in `OnWeightDemoted`/`OnActivationOffloaded`.
- **Five behaviours the shared algorithm did not have, each found by a failing test rather than by reading.** The
  CUDA suite went 71 failures to 0 as they were fixed, and they are what "one implementation" actually costs:
  - *Arena-backed buffers must be recorded at allocation time.* Asking "is this address in a live arena?" at free
    time is a different question: a captured graph's arena leaves the live list when the graph is disposed, and
    after that every buffer it handed out looks ordinary, so teardown frees memory the driver already reclaimed.
  - *The orphan sweep may not run during a stream capture.* `cuMemFreeAsync` on a buffer allocated before the
    capture began is rejected outright — the guard existed, naming the three tests it fixed, and the port lost it.
  - *Teardown frees synchronously on a drained stream.* Mid-op a transient goes back to the pool so the free is
    ordered after the work that used it; at teardown that ordering is meaningless and the stream is about to go.
  - *A demoted weight is parked as an orphan OR queued for the persistent free, never both.* Both freed one
    pointer twice.
  - *`CachedBytes` is computed, not accumulated.* It was maintained by hand at half a dozen sites, and delegating
    those silently stopped updating it. A counter that drifts to zero while the memory is still resident is worse
    than no counter.
- **Bulk offload's pinned policy is stated rather than assumed, and the two backends assumed opposite things.** The
  shared cache skipped pinned activations; CUDA paged them out first. CUDA is right and its reason is in the type
  now: a pin means "survive `FreeActivations`", which DESTROYS the device copy, whereas offloading is
  non-destructive — contents go to host and come back on the next read. The low-VRAM lever's whole target is that
  cross-step state, so `MayOffload` is a hook with the conservative default and CUDA overrides it.
- `TryGetWeightCast`/`CacheWeightCast` take the target dtype. CUDA kept one cast per weight while the shared cache
  keys per (weight, dtype) — a superset — and both call sites already knew the GEMM dtype.

## alpha.111

- **CUDA's residency cache is the shared one now.** `GpuTransferHelper.State` derives from
  `GpuResidencyCache<ulong>`, so the weight/activation/cast collections, the four-step rebind, weight demotion,
  the keyed bindings, orphan parking, bulk offload and teardown exist once instead of twice. Every static signature
  is unchanged — the 1399 call sites in `CudaBackend` did not move — and what is genuinely CUDA's rides the
  overrides: the graph-capture arena in `AllocateDevice`, the persistent weight allocator in `AllocateWeight`, the
  stream-ordered copy and H2D profiling in `Upload`, Q8_1 sidecars in `OnActivationEvicted`, auto-promotion in
  `TryMakeResidentOnMiss`, and re-promotion blocking in `OnWeightDemoted`/`OnActivationOffloaded`.
- **Five behaviours the shared algorithm did not have, each found by a failing test rather than by reading.** The
  CUDA suite went 71 failures to 0 as they were fixed, and they are what "one implementation" actually costs:
  - *Arena-backed buffers must be recorded at allocation time.* Asking "is this address in a live arena?" at free
    time is a different question: a captured graph's arena leaves the live list when the graph is disposed, and
    after that every buffer it handed out looks ordinary, so teardown frees memory the driver already reclaimed.
  - *The orphan sweep may not run during a stream capture.* `cuMemFreeAsync` on a buffer allocated before the
    capture began is rejected outright — the guard existed, naming the three tests it fixed, and the port lost it.
  - *Teardown frees synchronously on a drained stream.* Mid-op a transient goes back to the pool so the free is
    ordered after the work that used it; at teardown that ordering is meaningless and the stream is about to go.
  - *A demoted weight is parked as an orphan OR queued for the persistent free, never both.* Both freed one
    pointer twice.
  - *`CachedBytes` is computed, not accumulated.* It was maintained by hand at half a dozen sites, and delegating
    those silently stopped updating it. A counter that drifts to zero while the memory is still resident is worse
    than no counter.
- **Bulk offload's pinned policy is stated rather than assumed, and the two backends assumed opposite things.** The
  shared cache skipped pinned activations; CUDA paged them out first. CUDA is right and its reason is in the type
  now: a pin means "survive `FreeActivations`", which DESTROYS the device copy, whereas offloading is
  non-destructive — contents go to host and come back on the next read. The low-VRAM lever's whole target is that
  cross-step state, so `MayOffload` is a hook with the conservative default and CUDA overrides it.
- `TryGetWeightCast`/`CacheWeightCast` take the target dtype. CUDA kept one cast per weight while the shared cache
  keys per (weight, dtype) — a superset — and both call sites already knew the GEMM dtype.

## alpha.110

- **SD1.5 and SDXL open through the container**, the last two recipes still calling `SafeTensorsLoader` directly.
  A `fp8_scaled` or GGUF UNet of either was invisible to them, and companion scales went unfolded.
- SDXL's checkpoint mapping is now `using`-scoped. It was disposed only on the failure branch, so a successful
  construction left it open for the life of the pipeline.

## alpha.109

- **LTX (0.9.x and 2.x) and HunyuanVideo open through the container.** Wan was flipped in Phase A; these three
  still opened their checkpoints with `SafeTensorsLoader` directly, so a GGUF was invisible to them and a
  `fp8_scaled` or `int8` build had its companion scales left unfolded unless a converter happened to fold them
  itself. Their loader lists hold `IDisposable` now, because what owns the mapping depends on the format.
- HunyuanVideo's side-component loader drops its explicit `ApplyFp8ScaledDequant`. The container folds companions
  on open, and leaving the call in would have implied it does not.

## alpha.108

- **`hartsy quantize` writes a quantized copy of a checkpoint offline.** The engine still never quantizes at load,
  so what a generation runs is the file on disk rather than a runtime decision about it; this is how the smaller
  file comes to exist.
- **It reads through the container, which is the part that matters.** `GgufQuantizer` could already write a GGUF,
  but only from a raw `SafeTensorsLoader` dictionary — and a `fp8_scaled` or `int8` checkpoint keeps its scales in
  companion tensors, so quantizing those bytes without folding them first writes a file wrong by exactly those
  scales. A plausible file, not an error. Reading through `CheckpointSource` folds them, and also means a GGUF can
  be re-quantized into a smaller one.
- Each tensor is materialized to F32 by the route its own form needs, because picking the wrong one is silent:
  `CastTo` folds an fp8 scale into the values but refuses a block-quantized source by design, and an int8 weight
  carrying no descriptor has an unknowable scale and is refused by name rather than guessed at.

## alpha.107

- **Qwen-Image and Mage-Flow schedule their conditioning per step**, finishing C.2 for every family that applies
  prompt weighting. Each distinct step-text is tokenized through the recipe's own template, so a branch carrying
  its own `(word:1.5)` emphasis is weighted per branch, encoded once inside the single text-encoder-resident
  window, and selected by step.
- Both recipes flatten for their base ids. Declaring `PromptScheduling` is what stops the tag being collapsed
  upstream, so the raw tag text reaches the recipe and would otherwise be tokenized as prose — the bug the Flux.2
  gate caught, avoided here by construction.
- Qwen-Image narrows two things that assume fixed conditioning: its cross-generation prompt cache is keyed on a
  single token array and is bypassed, and its first-block step cache is calibrated on drift between consecutive
  steps under fixed conditioning, so a reused feature would carry one branch's text into another's step. The
  activation reserve is sized from the longest variant rather than whichever is current.
- Which families declare the scheduling bit is now ledgered. Whether a pipeline really consumes a schedule cannot
  be checked by reflection, and declaring it without one is silently wrong rather than an error.

## alpha.106

- **The shared residency cache learns that weights and transients can need different allocators.** A new
  `AllocateWeight` seam, defaulting to the transient allocator so a backend with one allocator ignores it. CUDA is
  the case that forces it: a preloaded weight comes from `cuMemAlloc` and is released with `cuMemFree`,
  deliberately outside the stream-ordered pool every transient uses, and routing weights through the transient
  allocator would hand a pool block to a synchronous free — an error on that API, not a style question.
- **And that a cache miss can be satisfied by the backend itself.** `TryMakeResidentOnMiss` fires after the miss is
  counted and BEFORE a transient is allocated. The original design mapped CUDA's auto-promotion onto the
  post-upload `OnTransientUploaded` hook; measured, those are different moments and different buffers — promotion
  happens before any transient exists and allocates a fresh persistent buffer — so a post-upload hook could only
  have promoted the pool block it was handed, or uploaded the same bytes twice. Both hooks now exist, each with one
  caller and one clear moment.
- `PromoteToWeight` drops any activation for the same tensor rather than assuming there is none. A lookup checks
  weights first, so an activation left behind is shadowed by the weight on every later read — the device write
  silently discarded. Today's only caller fires on a miss, where the tensor is in neither tier, but it is a general
  seam and the invariant is cheaper to keep than to assume.
- The weight-cast cache keys conversions by `DType` instead of by `DType.Name`. `DType` is a `readonly record
  struct` with structural equality; projecting it to a string to use as a key was stringly-typed for no gain.

## alpha.105

- **Swapping one model for another in a single process is now tested, and the test can fail.** Load, generate,
  tear down, load something else — the ordinary shape of a long-running server, and the scenario the
  `State.Unregistered` guard was written for after a GGUF model switch crashed. Nothing exercised it: the
  byte-identity harness runs one model per CLI invocation, and the CUDA suite's backends come and go without a
  second model taking their place. It alternates an orderly dispose with abandoning both model and backend, since
  only the latter leaves a queued cleanup pointing at a state the GC may already have collected.
- **It asserts device memory comes back, which is the assertion with teeth.** All three bugs the Vulkan half of
  this work produced were teardown leaks — invisible to a single generation, and to any test that builds one
  backend and stops. Disabling `cuMemFree` makes this test fail with 16 GB unreturned across four swaps; that is
  what it is for.
- **The measurement needs a pool trim, and finding that out was the interesting part.** Before the trim it
  reported 6.9 GB "unreturned" on a perfectly healthy run: every activation free goes through `cuMemFreeAsync`,
  which hands the block back to the stream-ordered mempool, and a pooled block still counts as USED in
  `cuMemGetInfo` until trimmed. Measuring without the trim would have reported a leak every time.
- Recorded honestly: the soak does **not** reproduce the original crash. That needs a `ConditionalWeakTable`
  finalized and then resurrected, a GC-timing window no test can force — verified by removing the retirement guard
  entirely and watching the soak still pass. What it does cover is the swap working end to end, every state
  retiring, and the memory returning.
- Missing models make it skip, as the suite's convention is — except under `HARTSY_REQUIRE_MODEL_SWAP_SOAK=1`,
  which turns absence into a failure. These tests print `SKIPPED` and return, which xunit records as a **pass**,
  so a gate that silently skips is a gate that silently passes.
- `TestPaths.Llm` gains Qwen3-4B Q4_K_M: a different family and a different quantization from the Llama beside it,
  so a swap between them crosses the dequantize path as well as the residency cache.

## alpha.104

- **Flux.2 schedules its conditioning per step.** `<alternate:a, b>` and `<fromto[N]:a, b>` reached it as prose or
  not at all; now each distinct step-text is tokenized through the recipe's own chat template — so a branch
  carrying its own `(word:1.5)` emphasis is weighted per branch — encoded once, and selected by step. `<alternate:>`
  over 30 steps costs two encodes, not thirty, because variants are deduped globally rather than per contiguous run.
- Two things a schedule cannot coexist with, both narrowed rather than ignored. **Step-graph capture** replays one
  op sequence with the conditioning pinned as step-invariant, so it is skipped for a scheduled prompt and logs why
  — the output is unaffected, the step is slower. **A regional prompt** builds its text stream and attention bias
  once from the base conditioning, and switching underneath it would leave both stale while the shapes still line
  up, so that pairing is refused by name.
- **`<fromto[0]:a, b>` was a silent no-op for every family that cannot schedule.** A fromto switches at
  `step < when`, so at `when = 0` it has already switched before step 0 runs and `b` is its step-0 text — but the
  flattener returned the first branch unconditionally, resolving the tag to exactly the phrase it was written to
  replace. Pre-existing; it took wiring a family far enough to compare against a plain prompt to see it.

## alpha.103

- **The residency cache answers questions about a tensor now, instead of handing out its dictionaries.**
  `TierOf(tensor)`, `IsPinnedActivation`, `OwnsBuffer` and the four counts land on `GpuTransferHelper.State`, and the
  ~35 test assertions that reached into `WeightCache`/`ActivationCache`/`WeightCastCache`/`PinnedActivations`/
  `CachedPointers` now ask those instead. The collections are an implementation of residency, not the definition of
  it, and every assertion written against them was coupled to that choice — which matters immediately, because the
  shared `GpuResidencyCache<TBuffer>` keeps them `protected` and those are the teardown and isolation tests that have
  to keep working across the migration.
- **`GpuResidencyTier` makes "resident as both" unrepresentable, and `TierOf` throws when it happens anyway.** A
  lookup checks weights first, so a tensor in both tiers has its activation — the bytes an op just wrote — shadowed
  by a stale weight on every later read. That is the auto-promote-discards-device-writes bug, and until now nothing
  asserted it could not occur: `PreloadWeight` tests only the weight cache before inserting, so a tensor computed as
  an activation and then preloaded was the plausible route in. The whole CUDA suite passes with the check live,
  including the RoPE-table lifecycle tests that take exactly that route, so the invariant is now a tested fact
  rather than an assumption the shared cache was about to be built on. The check stays on the query surface and off
  the production lookups deliberately: a guard at the weight-cast call site would fire after `CopyToDevice` had
  already served the stale bytes, crashing the generation somewhere unrelated to the cause. Guarding the read
  itself is a design call for the migration, not for this PR.
- **The write that could create that state is guarded, at the write.** `RegisterCachedWeight` refuses to make a
  tensor a weight while it is still a live activation. Nothing could reach that state today, but only by accident:
  all three callers read `DataPointer` to find the host bytes to upload, which fires the activation's sync callback
  and evicts the entry. An invariant held by a side effect of an unrelated read is one line from being lost — a
  caller uploading from a pinned or mapped buffer would never touch `DataPointer`, and `CudaStreamingWeightCache`
  is already most of the way there. The check sits where the state would be established rather than where it would
  be noticed, and weight registration is a load-time path, so it costs a dictionary probe per weight.
- `HartsyInference.Cuda` references `HartsyInference.Gpu` for the first time. Nothing depends on it yet beyond the
  enum — it is landed here, on its own, so the package-graph change is proven separately from the cache migration
  that needs it. Verified by packing rather than by building: `HartsyInference.Cuda.nupkg` declares
  `HartsyInference.Gpu 2.0.0-alpha.103`, which is the claim that matters at publish time.
- **The publish workflow's partial-release guard did not know `HartsyInference.Gpu` exists.** That list is there to
  refuse a release where one project failed to pack, precisely so consumers pinning exact versions do not find a
  dependency missing from the feed — and `Gpu` has been shipping unguarded since it was created, with `Vulkan`
  already depending on it. This PR adds a second dependent, so it adds the package to the list.

## alpha.102

- **Six more Runtime knobs were frozen, in mutable statics the previous check did not look at.** The scope lint
  required `readonly`, so `internal static bool FusedMmaGemm = EngineKnobs.Int8FusedMma.Value;` and five like it
  slipped through — bound at type-initialization exactly as a readonly field is, and worse rather than better, since
  a mutable process-wide static has no per-request isolation at all. The lint no longer asks for `readonly`: a
  static field initialized from a knob is frozen, full stop.

  The tell was who wrote to them. Every one of the writable ones was assigned only by tests, reaching past a knob
  that could not reach the code — `CudaBackend.FusedMmaGemm`, `FuseHeadGateIntoQuant`, `GroupedLinear` and
  `LtxVideo2Attention.TokenMajorAttention`. Those are live reads now and the tests set the knob instead, so what
  they exercise is the path a real request would take.

  `WanVideoDebugDump` kept its runtime setter, which is legitimate, but seeded it from the knob at
  type-initialization — so the first generation in a process named every later one's dumps. An explicit tag now wins
  and the setting decides when nobody set one.

- **The dump directory next to that tag was frozen too, one hop further out.** `DebugDumpSink` resolved its knob in
  its constructor, and all nineteen dump sinks are held in `static readonly` fields, so eighteen of them bound the
  directory at type-initialization. The lint cannot see this shape — the read is inside an instance constructor —
  but the tell was there again: the one sink that opted out of caching was the one whose parity tests needed the
  knob to reach the code. The opt-out is gone and every sink resolves per access, which also removes a
  created-once flag that would have sent later dumps to a directory it never made.

- **And a third shape, in a lazily-initialized static.** `Hunyuan3DDebugDump` resolved its dump directory behind a
  double-checked flag, so the first `Enabled` read in the process decided it for every later one. The lint could not
  see that either — the read sits in a property body, which is normally the safe place for one — so it now also
  flags a knob read whose result is assigned to a static field, wherever that assignment lives. It gained a second
  detection at the same time: a namespace-qualified `Configuration.EngineKnobs.X.Value` was invisible to it, and
  though no read in `src/` is written that way today, a check with a way around it is worth less than the diff that
  closes it. Both rules were mutation-tested — a planted violation of each fails the build, and a live read in a
  property still passes.
- `WanVideoDebugDump.SetTag(null)` pinned an empty prefix instead of handing the choice back to the setting, so the
  CFG pipeline's first clear masked the configured tag for the rest of the process. Three places said it should do
  the opposite, including the doc comment directly above it.

- **A fourth shape, and the one that reached real numerics: an instance field on a cached object.**
  `MiniMaxMusic3ArPipeline` bound `numerics.mm3CfgBatch` into a `readonly` field in its constructor, and that
  pipeline is built during model load and then cached and reused by `MusicService`, so every request after the
  first inherited whatever the loading request happened to see — its own `KnobProfileScope` override could not
  reach it. The value is now read once per generation rather than per use, deliberately: the two call sites must
  agree, since the feedback is built with two rows exactly when the batched step consumes two, so reading live at
  each would let them tear. The lint flags an instance field initializer now as well; a syntax tree cannot know
  which objects outlive a request, and presuming the freeze costs nothing when no legitimate instance exists.
- `WanVideoDebugDump`'s tag override is `AsyncLocal`, for the reason `KnobProfileScope` is. The CFG pipeline sets it
  around each branch forward, so on a plain static two generations on two devices would relabel each other's dumps.
  Only filenames are at stake, but a dump whose name lies about which branch produced it is worth nothing.

## alpha.101

- **MiniMax-H3 runs from a GGUF, verified by generation.** The `unsloth/MiniMax-H3-GGUF` Q4_K build renders the
  same scene as the fp8 checkpoint it is a requant of, at 141 frames 512x288 seed 1 and 30 steps, and its hash is
  now bound in `VideoProfileManifest` so it plans with H3's real task semantics instead of `UnknownBaseProfile`.
- **The GGUF is about twice as fast per step here** — 5.20 s/step against fp8's 10.39, measured as a two-point
  difference so load and decode cancel. That is the opposite of the expectation that a transiently-dequantized
  build must be slower. The likely reason is residency rather than arithmetic: 19.4 GB of fp8 weights in a 24 GB
  card leaves little room for anything else, where Q4_K needs roughly 3.7 GB. Inferred, not measured.
- **Plain Q2_K from that repack is published but unusable**, and the loader is not at fault. It renders a
  repeating lattice where Q4_K renders the scene at identical settings, though the two files differ only in the
  format of the same 208 weights; our Q2_K dequant agrees with our CPU codec and matches ggml's reference walk.
  Whether 2.6 bits is simply too coarse for an already-pruned DiT, or that repack's quantizer is at fault, is not
  established — so the hash is deliberately left out of the manifest rather than recorded as broken.
- `h3_bench.sh` stops claiming you need more steps when the truth is that it found no timings at all. The CLI
  prints `denoise [n/m]` with no per-step figure, so the harness cannot report s/step; it now says so and names
  the two-run method instead.

## alpha.100

- **A GGUF MiniMax-H3 text encoder loads.** Preflight refused the published repack for a shape error —
  `visual.patch_embed.proj.weight` "must be [1152,3,2,16,16], got [3456,2,16,16]" — which reads like a corrupt
  download but is a container limit: ggml caps a tensor at `GGML_MAX_DIMS = 4`, so no GGUF can hold a rank-5 weight
  and every repack folds the leading pair (1152 x 3 = 3456). Row-major the two layouts are the same bytes in the
  same order, so the fold is a relabeling and nothing needs converting.
- Behind that refusal was a silent one. The input channel count was read as `Shape[1]`, which is 3 on the rank-5
  weight and the temporal patch size — 2 — on the folded one, so the tower would have built a `[1152, 1024]`
  projection out of a `[1152, 1536]` weight and copied the front of it. The count is now divided out of the element
  count, which gets the same answer from either layout, and the copy checks its source length instead of trusting
  the caller's arithmetic.
- The fold is accepted as that exact shape, not as "rank 4 that multiplies out": `[1152,6,16,16]` is still refused,
  and the refusal names both forms it would have taken.
- A patch embedding that is itself block-quantized is decoded rather than cast. Our own GGUF policies quantize
  rank>1 non-norm weights, and this is one; `CastTo` refuses a quantized source by design, so letting the shape
  through at preflight without decoding here would only have moved the same failure into construction.

## alpha.99

- **`(word:1.5)` reaches five more families the way SwarmUI means it.** Prompt weighting worked only on SD1.5 and
  SDXL, because the code that applied it was written against `ClipTextEncoder` and nothing else could reach it.
  Qwen-Image, Flux.2 and Mage-Flow now apply the mechanism ComfyUI would apply to them, and SD1.5/SDXL keep theirs
  unchanged.
- The mechanism is not a per-family opinion. SwarmUI runs one probe —
  `use_attn_token_weights = not token_batches_have_weights(clip.tokenize("(x:2)"))` (`SwarmText.py:553`) — so a
  family blends on the encoder output only when EVERY tokenizer arm keeps weights, and one weight-keeping arm puts
  the whole family back on the blend. Reading that rule rather than grepping for `disable_weights` moved four
  families off the mode the plan had assumed for them, Lumina2 among them.
- **A recipe may not declare a mode its pipeline cannot act on**, and a test now enforces it. The declaration is
  what keeps the `(text:N)` parens in the prompt; a family that declared weighting without applying it would hand
  the parens and the digits to its encoder as prose — worse than not weighting at all. Thirty-three ledgered
  families therefore still declare nothing, each with its reason recorded.
- **Weighting on Kandinsky5 is a no-op, and that IS parity.** SwarmUI selects the blend for it, but
  `Kandinsky5TEModel.encode_token_weights` returns the Qwen conditioning plus CLIP-L's pooled vector and discards
  the blended hidden states, so the weights never reach the model. Implementing the blend there would have broken
  parity rather than achieved it.
- **The refiner prepares the caller's prompt for its own family**, rather than inheriting whatever the base was
  left with. Preparation is destructive in both directions: a weighting base's `(text:N)` grammar would reach a
  refiner whose encoder reads parens as prose, and — the direction easier to miss — an unweighted base collapses
  `<weight[1.5]:x>` to `x` before a refiner that *can* weight ever sees it, so the emphasis is silently gone.
  Neither is recoverable from a prompt already resolved for someone else.

## alpha.98

- **A per-request setting was silently ignored for 53 of the engine's knobs.** `KnobScope.Runtime` declares a knob
  "read each generation; safe to override per request", and `KnobProfileScope` exists to carry a request's settings
  into generation — it is pushed per request by the image and video services, and uses `AsyncLocal` specifically so
  two engines generating on two devices cannot decide each other's numerics. But 56 knobs were read into
  `static readonly` fields, which bind once at type-initialization and are never re-read. For all of those the
  request override did nothing, and worse than nothing: a process-wide field means whichever request touched the
  type first decided for every later request and both GPUs, which is the exact failure the scope's `AsyncLocal` was
  chosen to prevent.

  The scope was a declaration nothing enforced. It is now read at the point of use — activation dtype, step-graph
  capture, KV-cache precision, orphan sweeping, INT8 row budgets, im2col band caps, every probe and dump — and
  `KnobScopeIsEnforcedTests` fails the build if a `Runtime` knob is frozen again. There is deliberately no allowlist
  file: the knob's own declared scope is the allowlist, so marking one `Construction` is how you say a value really
  is baked in, and the three that still are (Vulkan coopmat, profiling and submit-per-op, all decided when the
  device and its pipelines are built) are named by a test.

  Measured first, because the fix depends on it. `KnobResolutionBenchmarks` is kept so the resolve cost stays
  reproducible — roughly 57 ns with a request profile pushed, which is the production path. The other half of the
  number was a one-off: counting the per-op orphan sweep, the hottest reader, gave 9,741 calls for a 1024x1024
  8-step image, so about 0.56 ms across an eleven-second generation. Live reads, rather than a snapshot cache built
  for 0.005%.

- **Tests configured the engine through environment variables nothing reads.** 95 call sites across 24 files set a
  variable that was retired when settings moved to knobs, so they were measuring defaults under a name claiming
  otherwise. One comparison ran both of its arms in the same configuration. `TestsDoNotSetKnobEnvVarsTests` now
  fails on any test that sets a knob's legacy name, taking the list from the registry rather than a hardcoded copy;
  harness gates like `HARTSY_REQUIRE_REAL_WEIGHTS` are untouched, being real environment variables read by the
  tests themselves.

- **Wan-Animate-2's driving-cache policy named a dead environment variable** in its logs, its unrecognized-value
  warning and the note on its out-of-VRAM message, telling the reader to export something inert. It now names the
  setting id, which `--set` and the settings file accept.

## alpha.97

- **MiniMax-H3 opens every component through the container**, so a GGUF build of the DiT, either VAE or the text
  encoder loads like any other checkpoint. The planner learned to read a GGUF header in alpha.90, which left the
  recipe as the only thing between an H3 GGUF and a generation — and meant asset resolution had to keep pretending
  `.gguf` files were not there, since offering a candidate the recipe could not open only moved the failure later.
- Norm promotion no longer goes through `CastTo`, which refuses a quantized source by design: decoding a block
  layout is the dequantizer's job, and a GGUF build carries quantized norms.
- **Q2_K and Q3_K dequantize on the GPU.** These were the two shipped H3 quants CUDA had no kernel for, and the gap
  mattered more than the others: at 2.6 bits per weight a 6.7 GB Q2_K build expands roughly sixfold when the loader
  has to widen it on the host, putting a model that would fit a 12 GB card out of reach of a 24 GB one.
- **Planning no longer refuses an audio VAE it can actually load.** It demanded the unfused PyTorch weight-norm
  parametrization for every convolution, but both forms are in circulation — the vendor release carries it, the
  ComfyUI repack ships it collapsed — and the decoder and encoder load either. Eighteen tensors present under their
  fused names were reported missing. That is the worst shape of planning error: the refusal is authoritative and the
  capability it denies exists.
- H3's text encoder reads its embedding row scale from wherever the fold left it. Reading only the companion key
  made a container-opened checkpoint look like one with a missing scale, and the refusal below rejected the
  published int8 build.
- MiniMax-H3 asks only the devices that will execute its blocks before deciding whether a quantized DiT can stay
  packed. It never runs context-parallel or CFG-parallel — the recipe warns the operator about exactly that — so a
  peer with no packed-weight kernel used to widen the whole checkpoint on the host for a device that reads none of
  it, which turns a 6.7 GB Q2_K build into roughly 40 GB.
- A quantized `.gguf` component no longer displaces a proven dense one. Component ranking read only the ComfyUI
  markers (`fp8`, `int8`, `nvfp4`), so `…video_vae-Q4_K.gguf` landed in the dense class beside the FP16 build and
  the size tie-break then chose it — the silent substitution the video VAE's explicit-selection gate exists to
  prevent.

## alpha.96

- **YuE2 loads its own published `int8_convrot` repack.** It was refused by name, on the grounds that the per-layer
  quant reader did not exist; it does now, and it lives at the container, so the refusal's reason was stale. The
  refusal itself was not: with it simply deleted the file does not load, because YuE2 quantizes more than the
  refusal's comment assumed. Reading the published header rather than the doc showed 229 `int8_tensorwise` weights
  at ConvRot group 256 — and among them the **embedding table**, the **output head**, `llm2vae` and the timestep
  MLP, three of which this model reads on the host.

  So four things had to change before the refusal could go. The fused QKV and gate/up splits now narrow each part's
  per-row dequant scale to the rows it takes and size the copy through the block layout rather than
  `DType.SizeInBytes`, which is 0 for every block quant and copies nothing. The host-read entries are decoded
  instead of cast — raw int8 bytes in a Hadamard-rotated basis are a different weight, not a wrong magnitude.
  `GenericTransformer` stops byte-concatenating Q/K/V and gate/up into one fused dispatch when a part carries a row
  scale the concatenation renumbers away, and its host-gathered embedding table decodes such a weight rather than
  casting it — both apply to every model that loads a ComfyUI int8 checkpoint, not just this one. The AR stack's
  32,769-row semantic head is decoded once at load rather than kept packed, because that row count is not a
  multiple of four and the packed GEMM would otherwise dequantize the whole window once per token.

  Not yet run against the real 7.8 GB file: the numbers here come from the published header and from synthetic
  cases, and the song itself still needs a listen against the BF16 build.
- **YuE2 opens through `CheckpointSource`**, so a GGUF build loads as readily as a safetensors one, and
  `AudioLmQuant` selects a placed `q4_k`/`q8_0` GGUF when one exists. Nothing is quantized at load and nothing new
  is downloaded — the hub ships no GGUF YuE2, and writing one is the offline tooling's job.
- A LoRA sent to YuE2 now says so in the log. It reached the runner cache key but nothing applied it, so the
  request got a fresh runner that generated exactly the base model's song.

## alpha.95

- **Seventeen more image families open their checkpoints through the one container**, so each accepts a GGUF or a
  quantized repack rather than safetensors alone: Chroma and its Radiance and Zeta variants, Z-Image, Lumina-2,
  AuraFlow, Anima, Ideogram 4, ERNIE-Image, Krea 2, HiDream, Boogu, Mage-Flow's side models, Kandinsky 5, Lance,
  OmniGen 2 and F-Lite. Every image recipe now loads this way except SD1.5/SDXL, which need rank-4 quantized
  convolution support first, and Lens.
- **Chroma's four split helpers sized their copies from `DType.SizeInBytes`, which is 0 for every block quant.** On
  a GGUF they produced correctly-shaped, entirely zero projections while the dense path stayed byte-perfect — the
  same arithmetic that was fixed in the QKV splits, in four more places. They go through a new quant-aware
  `SplitRows`, which narrows each piece's row scale before the first allocation so a refusal cannot strand one.
- **F-Lite never folded its quantization companions at all**, so an fp8_scaled build ran every weight at `1/scale`.
- Zeta-Chroma's attention fusion concatenated Q/K/V and dropped every companion; it refuses a per-row-quantized
  weight by name instead, and likewise a build that stores Q, K and V in three different dtypes — the fused tensor
  can declare only one, and a K or V wider than Q used to be copied past the end of it. Krea 2's pre-rename
  companion carry is deleted, dead now that folding precedes renaming.
- Lumina-2, HiDream and OmniGen 2 widen to the dtype their transformer actually runs, rather than the F16 default
  that would have left a GGUF mixing dense F32 with widened F16.
- **A component published as `.gguf` inside a diffusers folder was invisible.** F-Lite, Lance, Kandinsky 5, Krea 2
  and Boogu discovered their components with a `*.safetensors` glob and threw before the container could be sniffed,
  so the GGUF path they now advertise was unreachable without renaming every file to a lie. Folder discovery reads
  the leading bytes instead, the way single-file loading always has, and a set that mixes the two containers is
  refused rather than merged into one dictionary.
- **Selecting a Lumina-2 GGUF stored beside the original sharded release loaded the release instead.** Any sibling
  `*.safetensors.index.json` used to expand the selection into every safetensors in the folder; the index now has to
  list the selected file before it expands anything, so a repack — or any second checkpoint parked there — loads as
  itself.

## alpha.94

- **A LoRA now applies to a block-quantized base without requantizing it.** The classic codecs a GGUF uses have no
  quantizer at all, and requantizing a merged result degrades both the base and the LoRA — so the delta rides on the
  weight as a low-rank adjunct and is accumulated inside the GEMM, which is what ComfyUI-GGUF does per forward.
- The adjunct is never written to the shared weight. Cached converted dictionaries, resident models and the
  identity-keyed device cache all hold that object, so a patched base would have leaked one request's LoRA into the
  next with nothing in the cache key to show for it. It rides a borrowed view that the stack owns and disposes.
- **Every GEMM entry a patched weight can reach either applies the adjunct or refuses by name.** A quiet miss reads
  as "the LoRA looks weak", never as an error, so `LinearImpl` was split and the accumulate moved into a wrapper
  around the dozen fused paths that return early. The fused GELU and head-gate entries un-fuse instead of adding
  afterwards, because the delta has to land before the activation.
- **One call site for LoRA merging.** `LoraApplier` is gone and all 36 recipe sites go through `RecipeLoraMerge`,
  which owns the merge-before-load invariant.
- **A text-encoder LoRA strength now does something.** `TencStrength` reached the cache key but never the merge, so
  the stack applied one strength to everything; CLIP-L, CLIP-G and the new `TextEncoder2` target take it properly.
- **A DoRA adapter is now actually decomposed on a dense base.** The magnitude vector reached `LoraDelta.DoraScale`
  and the decomposition was written and pinned against ComfyUI's reference, but nothing on the merge path called it —
  so a DoRA file merged as a plain LoRA with its magnitudes dropped, no warning anywhere. The quantized path already
  refused by name; only the dense one was silent, which is the exact failure the rest of this work exists to remove.
  A DoRA aimed at one slice of a fused projection is refused instead: its normalizer is defined over a whole weight,
  and on the input axis that is a column norm across every row, which a third of them cannot supply.
- A stacked LoRA carrying layers for a component the caller passed no dictionary for now warns by target and count.
  Adding `TextEncoder2` would otherwise have introduced exactly the silent partial merge this work is about — those
  keys used to be skipped as unrecognized and would now parse cleanly into a target nothing consumes.

## alpha.93

- **The shared residency cache now reclaims buffers an op displaces, and stops serving stale weights.** Four things
  CUDA's own cache had already solved, adopted before CUDA moves onto the shared one — so that migration is a port
  onto familiar ground rather than a rediscovery of the same bugs in a new place.

  A buffer displaced by a rebind used to stay owned by nothing until teardown. Whether it is the op's own input —
  whose cleanup will free it — or nobody's is not knowable at the moment of displacement, so it now parks: the
  caller's release claims it, and whatever is left is freed when the NEXT op starts, by which point every previous
  op's `finally` has provably run. Freeing at teardown instead was not a fix but a deferral; measured on CUDA before
  it had this, twelve `Linear` calls at a 563 MB output stranded 5942 MB.

  More seriously, a resident weight whose tensor an op bound to a new buffer stayed a weight, and lookups check
  weights first — so every later read returned the pre-op bytes and the device write was silently discarded. That is
  the same shape as the auto-promotion bug fixed in CUDA in August, and it was waiting in the shared base for the
  first backend to write through a weight. A tensor bound as an op's output is no longer a weight, whatever route
  made it one, and its cached dtype conversions go with it.

  `PromoteToWeight` gives a backend the seam it needs to promote a tensor it has seen uploaded twice — the buffer
  has to enter the weight cache and the owned set together, or the caller's own cleanup frees what the cache now
  points at — and plants the demotion binding that makes promotion safe. Promotion happens behind the caller's back,
  so host data stays authoritative: a later host write drops the device copy rather than syncing it back, because
  without that a write would leave the stale device bytes cached and every later lookup would serve them. An
  explicit preload deliberately plants nothing, since there the caller asked for residency and owns the lifetime.

  The callbacks a tensor fires on read or dispose now check disposal before touching anything, behind a gate a
  backend can hold closed while it retires: a binding outlives the cache that planted it, and on CUDA that path
  threw during a model swap.

## alpha.92

- **Two backends on the shared residency cache would have handed out the same binding key.** The counter was a
  static field inside the generic class, and such a field exists once per CLOSED type — so a cache over one buffer
  type and a cache over another each started at 1. That is the collision the key exists to prevent, moved up a
  level: a host tensor resident on two devices, which split placement makes ordinary, would carry both bindings
  under one key and their finalizer-cleanup buckets would collide, so one backend's drain runs the other's device
  cleanup. Latent until a second backend joined the base, which is the next step.

  Moving the counter off the generic class was only half of it: CUDA's state registry allocates binding keys from a
  private counter of its own, also starting at 1, so a CUDA cache and a Vulkan cache collide today — not after some
  future step — the first time one tensor is resident on both. The sequence now lives in Core beside the bindings it
  names, where every backend already reaches, and CUDA draws from it. A source test keeps it the only one: the bug
  has now been written twice, each copy correct alone, and neither was visible until a second backend existed.

## alpha.91

- **Vulkan's residency cache is now the shared one.** Its three caches, lookup order, tensor-binding lifecycle,
  weight-cast cache and offload policy come from `GpuResidencyCache<TBuffer>`; what stays behind is what is
  genuinely Vulkan — the ReBAR-or-staging upload, the non-coherent flush, deferred frees against the command
  stream's timeline, and the step-graph retain list. Output is byte-identical: SD1.5 and Krea2 both hash-match
  their pre-migration images exactly.
- **Vulkan drains its finalizer cleanup queue.** A tensor finalized rather than disposed cannot free its device
  buffer from the finalizer thread, so the work is queued for a safe point — and nothing on this backend ever ran
  it, so those buffers stayed allocated until the backend itself was torn down. The outermost op scope runs it now.
- Each cache instance takes its own binding key instead of every Vulkan device sharing `0`, so one device's
  teardown can no longer drop another's binding on a tensor resident on both.
- Three teardown bugs found and fixed while migrating, all recorded in the troubleshooting notes: a deferred free
  at teardown is never serviced, `public new` silently keeps the base implementation where `override` was meant,
  and a buffer orphaned by an in-place re-cache is reachable from the owned-buffer set but from no cache.

## alpha.90

- **Every checkpoint now opens through one container, whatever format it is in.** Quantized-checkpoint support
  had been wired architecture by architecture: a recipe that wanted GGUF grew a second constructor parameter and
  a second code path, so GGUF loaded for four image models and no video ones, and the ComfyUI fp8/int8 companion
  fold was copy-pasted into 24 of 47 checkpoint converters and simply missing from the rest.
- Nothing about a container is architecture-specific, which is the whole point. A diffusion GGUF is a repack —
  the publisher quantizes the released safetensors file and keeps its tensor names — so once the keys are mapped,
  the shapes relabelled and the quantization companions folded, a converter cannot tell the two apart.
  `CheckpointSource` does those three things once, sniffing the container from its magic bytes rather than its
  extension, because repacks are routinely published under the wrong one.
- **The fold runs before the converter, and that ordering is the fix.** A converter renames `.weight` and has no
  rule for `.weight_scale`, so folding afterwards pairs nothing and drops the scale — and a weight without its
  scale is not an error, it is a weight hundreds of times too large that renders as noise at the end of a
  generation. Folding at the container makes that class of bug unreachable.
- The same failure existed one level down: a `Tensor` built over another's bytes — a reshape, a dtype relabel, a
  same-device copy, a fused-projection split — started life with no scales at all. Those now carry the per-tensor
  fp8 factors always and the per-row companions whenever the row numbering survives, and refuse rather than pair
  each row with another row's scale when it does not.
- **A quantized weight the backend has no kernel for is now caught at load.** It used to fail inside the first
  GEMM, minutes into a generation, with a stack trace naming a kernel rather than a file; the loader asks the
  backend what it can hold packed and widens the rest on the host, saying which dtypes and why. A Q2_K or IQ4_NL
  diffusion GGUF — both routinely published — loads slowly instead of crashing.
- **Flux.1, SD3/SD3.5 and the Wan video family load GGUF too.** They worked on safetensors and simply could not
  open a quantized build; routing them through the container is the whole change. Verified with real generations
  from real community files: Flux.1-dev Q4_K_S (city96), SD3.5-medium Q8_0 (city96) and Wan 2.1 T2V 1.3B Q8_0.
- Two things that had to be fixed for those, both invisible until a published file was actually read. The SD3
  converter only routed keys under a `model.diffusion_model.` prefix, and city96's SD3.5 GGUF ships bare LDM keys,
  so every transformer tensor was dropped and the DiT loaded empty. And Wan's checkpoint is the first to carry
  tensors that are not matrices — 31 rank-3 modulation tables and a rank-5 Conv3d patch embed — which only arrive
  with the right shape because the container now reverses every ggml axis rather than just the two of a matrix.
- **A shipped checkpoint that rendered black now works.** Black Forest Labs' own `FLUX.2-klein-4b-fp8` carries 80
  fp8 weights and 160 companion scales, and the Flux.2 converter was one of the 23 that never folded them — so every
  fp8 weight ran at scale 1.0 instead of `stored x scale`, and the generation saturated to a fully black image. This
  is what the fold gap looks like when it lands on an official release, and why the fold belongs to the container
  rather than to a 24th converter.
- **A GGUF video checkpoint reaches planning.** The planner opened every checkpoint as safetensors, so a GGUF
  build died before any recipe was reached, and MiniMax-H3's component resolution could not even see a `.gguf`
  file. Both read the shared header now, and a component's format reports as `gguf-q4_k` and the like.
- **A fused projection can be read in windows when it is block-quantized.** This is what put a GGUF MiniMax-H3
  out of reach rather than merely making it slower: H3 reads its packed `qkv_proj` in two windows, that chunking
  is how the model runs at all, and the weight it chunks is the one the quantization applies to.
- bitsandbytes NF4 checkpoints load, through a decoder that had been written and never wired. Every quantity the
  file declares is reconciled against its actual byte counts first, so a layout misread refuses by name instead
  of decoding to plausible noise.
- Fourteen GGUF key mappers were the same class fourteen times — a family name, a recognition rule, and a method
  returning its argument — and are now one table of recognition rules. Writing them side by side surfaced that
  Flux.2 was asked after Flux although it keeps Flux's block naming, so a Flux.2 GGUF with no declared
  architecture detected as Flux.1.
- The GGUF writer emitted dimensions in the engine's order while every other tool reads ggml's, so a file the
  quantizer produced came back with every matrix transposed. Nothing consumed those files yet; they are now
  readable by other tools and by our own recipes.

## alpha.89

- **A shared GPU layer, `HartsyInference.Gpu`.** Nothing references it yet: the package is built and tested first so
  each backend can be moved onto it one at a time, with its own suite green at every step.
- `GpuResidencyCache<TBuffer>` is the device-residency cache both GPU backends had written separately — the same
  three caches, the same weight → activation → fresh-upload order, the same four-step activation bind. A backend
  supplies five operations that genuinely need an API (allocate, free, upload, download, make-current) and inherits
  the rest. Two drifts between the old copies are settled by having one: only one of them drained the finalizer
  cleanup queue, so tensors finalized rather than disposed leaked their device memory on the other; and one keyed
  every device's tensor binding as 0, which holds only until two of its devices are used at once.
- Because the cache decides *which* tensor is resident rather than doing any transfer itself, it is testable against
  a fake buffer with no GPU at all. Nine tests cover the cases that previously needed hardware to reach: re-caching a
  tensor in place, a weight surviving an activation sweep, pinning, per-device binding independence, arena-owned
  buffers, and a host read during capture.
- `OpProfile` gives both backends the per-op timing only one had, so the pipelines that already call
  `ResetOpProfile`/`DumpOpProfile` stop silently producing nothing on the other.

## alpha.88

- **The backend contract now describes a device instead of listing the two backends that exist.** `IsGpu` tested for
  CUDA-or-Vulkan by name, so a device added later would have read as a CPU to every caller that gates on it — same-
  device serialization, weight preloading, VRAM reclamation would each have skipped it silently. It asks whether the
  device is not a CPU, and `DeviceType` names ROCm and Metal so that question has real answers to give.
- `CacheWeightCasts` moved onto `IBackend`. Both GPU backends already had the property with the same name and the
  same meaning, and the one caller reached them through a type test naming each — so a third backend would have been
  skipped while the log line still claimed the flag had been applied. The recipe helper no longer references the CUDA
  or Vulkan packages at all.
- Backends report their vendor, device name and total VRAM. A VRAM tier can only be resolved from a number somebody
  publishes, and nothing published one. Vendor matters because several decisions are per-vendor rather than per-API —
  cooperative-matrix reliability above all — and a software rasterizer is its own vendor, since llvmpipe otherwise
  reports the silicon vendor of the host and would walk into a hardware comparison.
- Vulkan finally declares that its convolution bands its im2col workspace. It has done so since the Krea2 VAE-decode
  fix, but never said so, leaving the VAE planner to assume the naive blow-up on that backend.

## alpha.87

- **A mistyped command-line option now fails instead of being ignored.** Spectre collects an option no command
  declares as a "remaining" argument, and nothing reads those — so a typo did not fail, it ran the generation with
  a different setting than the one asked for and reported success. `--cfgscale 2` (the option is `--cfg`) quietly
  generated at the model's default guidance, and `--detect` (it is `--mode detect`) quietly fell through to CLIP
  and then died several layers down in a model loader complaining about a missing text-encoder weight. Both were
  found the hard way, drawing a wrong conclusion from a run that had not used the settings it was given.

## alpha.86

- **Vulkan convolves a batch.** `Conv2D` refused `batch > 1` outright, which is what made SDXL unusable on the
  Vulkan backend: its fused denoise loop runs one batch=2 UNet forward per step, with the positive and negative
  prompt concatenated for classifier-free guidance, so the first convolution of the first step threw. SD1.5 never
  hit it only because it runs CFG as two separate batch=1 passes. No kernel changed: the im2col shader already
  wrote each image's columns as its own block, and `matmul_tiled` has carried the `aOffset`/`bOffset`/`cOffset`
  push constants "for batched dispatch" all along — the convolution now walks them per image the way
  `BatchedMatMul` already did. The column-tile budget is divided across the batch, so the cap that exists to bound
  peak im2col memory keeps meaning what it says instead of being exceeded by a factor of the batch.
- **Three dtype fallbacks stopped calling themselves.** `((IBackend)this).X(...)` reads as "run the managed
  default", but the class method implicitly implements the interface member, so interface dispatch re-enters the
  override: `WanRmsNormChannel`, `GatedResidualLastDim` and `RopeApplyDecodeStep` each recursed until the stack
  overflowed, taking the process with them, for any input that took the fallback branch. This is a bug class the
  troubleshooting notes already describe and had already cost two earlier instances; these were three more. The
  first two now call a static reference, as that note prescribes, and the third throws — it is a device-position
  op whose interface default is an empty body, so "falling back" would have silently skipped the rotary embedding
  and returned a plausible wrong token rather than failing.
- `WanRmsNormChannel`'s reference states its F32-only contract instead of assuming it. It reads every operand as
  `float*`, so an F16 tensor reaching it would have been reinterpreted bit-for-bit into plausible garbage.

## alpha.85

- **`--backend vulkan` now reaches Vulkan for text generation.** `TextService` derived its device key as "not CPU,
  therefore CUDA", so every non-CPU selector became `cuda:{ordinal}`: a Vulkan engine silently ran its LLM on CUDA,
  and on a machine with no CUDA device it failed with a driver error instead of generating. The key now carries the
  resolved kind as well as the ordinal, and the slot builds its backend through `BackendFactory.Create` like every
  other modality — so an unknown device key reports the selectors that exist rather than naming CUDA and CPU as the
  only choices. Every Vulkan LLM measurement taken before this was a CUDA measurement.
- Same-device generation gating covers every device backend rather than CUDA alone. Two Vulkan generations on one
  card contend for its VRAM exactly as two CUDA ones do, and were running concurrently by default.
- **A registered recipe name is accepted as a family id.** The video planner's own error text advertises the
  drivable families, but several of them — the Wan compat classes — exist only as recipe names, with no catalog
  entry behind them. `-m wan-22-5b` was therefore listed as drivable, accepted, and then refused as family
  'unknown'. A name either registry knows now resolves as itself; a name neither knows still falls through to
  header detection, so an unregistered model keeps reporting what IS drivable.
- Video sparse attention is gated on what the backend can execute rather than on the spelling of the selector. The
  check compared the resolved selector against the string "cuda" before asking the capability, so a backend that
  implements the profile could never be reached through it. A backend without the native kernel is still refused,
  by name, with no dense fallback.
- `hartsy video` and `hartsy world` default to `--backend auto` and no longer describe themselves as CUDA-only.
  Neither ever enforced it: the restriction is per-profile (H3's sparse attention), not per-command.

## alpha.84

- Driving audio refuses the output timing edits that would slide the picture against it: a start trim, a boomerang,
  or an fps other than the native 24. Frame edits reach the frames only, and the soundtrack is trimmed at its end
  alone, so each of those quietly broke the lip sync the feature exists to provide.
- Video: **MiniMax-H3 can be driven by a soundtrack you supply** (`--driving-audio`, `VideoRequest.VideoAudioReference`).
  H3 has no audio-driven mode of its own and its reference audio is a soft exhibit the generated soundtrack can
  drift away from, which is no use when the words have to match. The mechanism that does drive video is the one
  long-form chaining already uses: hold every audio row fixed at the supplied track and let video denoise against
  it, so the model composes a picture around audio it cannot change. The mask is built inside the pipeline, so the
  request-level mask surface keeps its own release gate. A track longer than the clip is trimmed and a shorter one
  zero-padded, which is what the audio VAE's fixed-length encode already did. It is a planned feature
  (`VideoFeatures.DrivingAudio`), so a family that cannot consume it rejects the request during planning instead of
  accepting the option and ignoring it, and a sparse VSA profile — which is T2VA-only — is refused there rather
  than at the execution boundary.
- Video: **Wan-S2V declares `VideoFeatures.DrivingAudio`.** Classifying `VideoRequest.VideoAudioReference` as a
  planned feature reached every family that reads that field, not only H3 — and S2V reads it as the driving speech
  it cannot run without. Undeclared, the generic planner answered `video.feature.unsupported` before construction,
  so the documented speech-to-video path would have failed on its own mandatory input. The declaration is the whole
  fix; the gate itself is unchanged, and a family that does not consume a supplied track still refuses one.
- Measured on a 90-frame 512x288 pair, same seed and prompt, differing only in the locked track: the output audio
  correlates 0.9771 with the driving track through the VAE round-trip; a silence-driven run emits rms 0.00002
  rather than inventing a soundtrack; the two clips diverge at SSIM 0.556; and motion runs 2.36x higher while
  speech plays than after it stops, against 0.80x for the silence control.

## alpha.83

- Video: **an unchunked MiniMax-H3 geometry is no longer charged for its own buffers twice.** Below
  `MinChunkableRows` the forward runs whole, and then `Attention`'s qkv and head-major q/k/v ARE the full-sequence
  buffers the pass-1 term models — the floor added both, counting one allocation twice (about 257 MB at seq 4700).
  Chunked they are genuinely distinct, since kFull/vFull outlive each chunk in flight, so the correction applies
  only when the scratch spans the whole sequence.
- Sizing that scratch by the real sequence rather than a fixed 4,096 rows (alpha.80) made the over-count reachable:
  it added 99 MB at seq 4700, turning a 51 MB margin into a 48 MB deficit and refusing a 90-frame 512x288 clip on
  a 24 GB card that had generated the same geometry minutes earlier. Every calibrated estimate sits above the
  chunking threshold, so none of them covered this branch; the new test pins an unchunked floor against what the
  unchunked forward actually allocates.
- The floor now models attention per implementation rather than assuming one shape. `AttentionSparse` keeps a
  full-sequence gate and the token-major buffer it permutes from alongside qkv and head-major q/k/v, an 8x
  projection peak against the dense path's 6x — and it holds that at ANY length, because `ForwardNamedBlock`
  selects it before testing `seq > chunkRows` and there is no chunked sparse path. Charging the released VSA
  profile a chunk's worth would have approved a near-limit generation that then ran out of VRAM, which is the
  dangerous direction for a pre-flight. The MLP chunks in both modes and is sized separately. A caller that does
  not name the mode gets the larger sparse reservation, so an omitted argument cannot under-estimate; the
  calibrated boundaries name themselves dense, since they measured the fp8 FL2VA path.
- The unchunked peak also reserves the modulated attention/MLP input. `ForwardNamedBlock` disposes it only after
  the call returns, so it is a second `[seq, hidden]` buffer live beside the residual — about 168 MB just below the
  chunking threshold. `Modulate` can emit fp8, but not on a bf16 checkpoint or with `numerics.modulateEmitFp8` off,
  so the reservation is F32. Chunked, the kFull/vFull term covered it incidentally; unchunked nothing did.
- The activation-accounting tests move out of `SyntheticSmoke` into the unit lane. They are arithmetic only — no
  model, GPU, checkpoint or network — but the class trait meant the documented CPU command skipped every one of
  them. This accounting has regressed twice now; quarantining its guards is what let the first one through.

## alpha.82

- Audio: **SheetSage2 transcribes a recording into a score.** With the symbolic half from alpha.81, the model is
  now complete: the MERT2 Conformer encoder, the score decoder, the greedy decode under its grammar, and the
  pass that reads a song longer than the encoder's window in overlapping passes and stitches them onto one
  timeline. `hartsy transcribe -m sheetsage2` returns ABC rather than words.
- This is what makes covering an existing recording possible. Until now YuE2 could only edit scores it had
  written itself, because it has no audio input at all; a transcription gives it a melody it did not compose.
- **Checked end to end, not only stage by stage.** Each stage is pinned against a dump of that stage — the mel
  frontend, the encoder and its learned layer mix, the decode, the grammar mask, the window plan, the stitch and
  the serializer. On top of those, a real recording is run through the whole chain and the resulting score is
  compared as text against what the released implementation produces from the same file. A single wrong note,
  bar line or chord symbol fails it, and it is the only check that can catch an error living in a handover
  rather than in a part.
- The decode is **token-exact** against the reference on both the host and CUDA: reference and port both run
  float32 over the same weights, so exactness is a real criterion rather than a coincidence of dtype. The
  largest logit drift observed consumed an eighth of the margin between the top two candidates.
- One thing the gate had to be built carefully to prove: the checkpoint has largely internalised its own
  grammar, and on a real clip the unmasked argmax is already legal at every step — so a port that dropped the
  mask entirely would still have passed. The mask is therefore pinned on a case constructed to need it, where
  the model would otherwise write a fifth consecutive subbeat shift and the grammar forces a pitch instead.
- Both renderings of the score, with chord symbols and without, come from one decode. The decode is the entire
  cost — the encoder attends over a fixed five-minute window and the token loop is autoregressive — while
  serializing events already in hand is free, and the two renderings are not a substitution apart.
- The weights are cached under the YuE2 repo they ship in, so a machine that already generates with YuE2 does
  not fetch a second copy. They are CC BY-NC 4.0, unlike the engine.

## alpha.81

- Audio: **the SheetSage2 port's symbolic half is complete** — the events-to-ABC serializer and the sliding-window
  stitcher. Between them they turn a decoded token stream into a finished two-voice lead sheet, which is the
  artifact YuE2 edits and re-renders, so this is the half that makes covering an existing recording possible
  rather than only editing scores YuE2 wrote itself.
- The serializer infers the beat grid and meter from the decoded timestamps, spells accidentals against the key
  and the running bar, splits durations that no single ABC token can express, and pads the idle voice so both
  voices carry the same bar count in every parallel chunk. It is checked string-exact against the released
  implementation on ten cases and, separately, against a transcription the model itself produced: the port
  regenerates that score byte for byte from the model's own events.
- **Writing the score without chords is not the same as deleting the chords from one that has them.** A bar
  carrying a chord symbol cannot fold into a multi-bar rest, a chord change inside a held note splits it into
  tied parts, and the two spell rests within a bar differently — four of the ten gated cases differ, six happen
  to coincide. So the mode is a parameter on the serializer and both renderings are produced from one decode,
  rather than one being derived from the other after the fact.
- The stitcher resolves each window's own subbeat and second counts onto the whole clip's timeline and decides
  which window is trusted for each passage. Its seam tolerance is deliberately asymmetric: symmetrising it
  duplicates or drops a bar at every seam. The gate covers fourteen seams across seven clip lengths, including
  events landing exactly on an accept boundary, and was mutation-tested — twenty deliberate breakages, with
  every surviving mutant either turned into a new case or shown to be mathematically equivalent.
- Two reference behaviours are reproduced rather than tidied, each with a note saying so: an unclipped interior
  interpolation in the time map (unreachable in the shipped pipeline, since transcription always asks for the
  full window), and several guards that the surrounding code makes dead. Two are deliberate divergences that
  throw instead: a melody track index outside the two voices, which the reference silently routes to the wrong
  staff, and a beat period below the timestamp resolution, which the reference would expand into millions of
  synthesized beats.

## alpha.80

- Video: **a VRAM posture now reaches video at all.** `--vram-mode` existed only on the image command, so passing
  it to `video` was accepted and ignored rather than rejected; `Vram` was put on the request only in the image
  dispatch branch, so video, music, world, restore, mesh and speech dropped it even when supplied; and `Program`
  read `diagnostics.logLevel` before pushing the `--set` profile, so raising the log level from the command line
  could not work — which is what made the first two hard to see. Every tier request for video was silently
  becoming `Auto`.
- Video: with the tier arriving, `MiniMaxH3ChunkPolicy` honours its `ChunkScale` lever, and a long-form chain
  hands device memory back between segments. The conditioning encoders load ahead of the DiT *within* a segment,
  but across a chain the previous segment leaves weights cached and the next segment's mask-source encode competes
  with them. Gated on `PhaseUnload`, so `VramTier.Auto` stays byte-identical to an unchained run.
  `MiniMaxH3Recipe` declares `PhaseUnload | Chunking` only because both are now wired. The unload frees only the
  pipeline's own DiT tensors: video takes no device gate, so evicting the backend's shared caches could strand a
  concurrent generation on the same device.
- Verified on a 12 GB RTX 3060, which previously ran out of VRAM at the mask-source encode: a 3-segment chain at
  512x288 producing 192 frames and 8.00 s of audio, the unload firing twice, and colour drift of 0.045 per frame.

## alpha.79

- Audio: **the SheetSage2 port continues** — its event codec, which reads a decoded token stream as musical
  events and writes events back as tokens. SheetSage2 transcribes audio into a symbolic score, which is what
  gives YuE2 a melody to cover, so it is the piece that makes editing an existing recording possible rather than
  only editing scores YuE2 wrote itself. Each event keeps both its raw tokens and their read values: the values
  are what a serializer wants, the tokens are what a later window replays verbatim as context, and re-encoding
  from values would not reproduce them.
- The vocabulary, decode grammar and sliding-window plan landed in alpha.78 without a changelog note, so for the
  record: the vocabulary is 31,678 tokens whose ranges are laid out in one pass from offset 260, an id's meaning
  is decided by nothing but which range it falls in, and decoding is greedy argmax over logits masked by the
  grammar — so the grammar does not guard the result, it decides it. All of it is checked against the released
  implementation's own output, dumped by `tests/python-reference/dump_sheetsage2_reference.py`, which needs no
  weights.
- Nothing in the port is reachable yet: the encoder, decoder and the events-to-ABC serializer are still to come,
  along with their parity gates, which do need the checkpoint.

## alpha.78

- Audio: **YuE2 can write its score without rendering it.** The model composes in two passes — an autoregressive
  model writes an ABC score, then a second pass turns that score into sound — and the score is the only editable
  artifact it exposes. Planning it alone takes seconds where the full render takes minutes, so it is now its own
  operation: `IMusicService.PlanScoreAsync` returns the score plus what the context left for audio, and the score
  goes back in through `MusicRequest.Yue2Abc` once edited. `BudgetAsync` answers the budget question on its own,
  for a score a caller already has. Measured 2026-09-16 on a 4090: a 25 s request planned in 14 s and reported a
  25.0 s audio budget; rendering that same score back took 13 s and produced 23.9 s of audio, inside its budget.
- Neither rides `/v1/native/music`, which rejects a result with no audio — they are `POST /v1/native/music/score`
  and `POST /v1/native/music/budget`, and `hartsy music --score-only` writes the score out as a `.abc` file.
  `IMusicRunner` grew optional `PlanScore`/`Budget` members, in the shape `SttRunner.Timed` already uses, so a
  model with nothing symbolic to report simply leaves them null and the service reports it as unsupported.
- `Yue2Pipeline.Generate` now plans through `PlanScore` instead of a second copy of the same sampling call, so a
  score asked for on its own and a score planned on the way to audio cannot drift apart.
- CLI: fixed `hartsy music --help`, which rendered nothing but an error. The `--genre` help text names the
  `[verse]`/`[chorus]` lyric markers, and Spectre.Console read those as markup tags.

## alpha.77

- Video: **MiniMax-H3 long-form chaining is released.** Its real-generation gate passed on 2026-09-16 against the
  fp8 FL2VA base: 3 chained segments at 512x288x141f produced one 345-frame clip whose video and audio both ran
  exactly 14.375 s, with seams at the 2.0th and 14.5th percentile of the clip's own adjacent-frame SSIM
  distribution (0.8856 and 0.9202 against a 0.8627 minimum and 0.9492 median) — a busier-than-average step rather
  than a cut. Verified the same way through the CLI and the native HTTP API. Guides and AV denoise masks stay
  release-blocked: chaining builds its masks inside the pipeline and never sets those request objects, so it
  carries its own gate rather than theirs.
- Each segment VAE-encodes a full-length source clip on top of its own generation, so a chain needs more headroom
  than a single generation of the same segment length — 23.4 GB peak at 512x288x141f.

## alpha.76

- Video: **MiniMax-H3 can generate past one denoise, as a chain of segments.** `VideoRequest.ChainTotalFrames`
  splits a longer target across successive generations; each one after the first copies the previous segment's tail
  into its leading rows and masks those rows out of denoising, so they carry that segment's motion and soundtrack
  phase into the new frames' attention context. The protected head is context rather than output — it is dropped at
  assembly, which leaves one continuous sequence with no duplicated frames and no seam to blend. Lengths stay on
  H3's `17k+5` grid (`MiniMaxH3ChainPlanner`) so a protected head always covers whole latent tokens; a head ending
  inside a token would hold part of it fixed while denoising the rest, which the sampler's row masks cannot express.
  Trim and boomerang apply once, to the whole video. Exposed as `--chain-frames` / `--chain-seconds` /
  `--chain-context-frames` on the CLI and as `chainTotalFrames` / `chainContextFrames` over the native video API.
  **Release-gated**: `video.h3_expansion.release_blocked` refuses it in published builds until its
  operator-provided real-generation and output-inspection gate passes.
- Video: `VideoDenoiseMask` gained `MaskFrameValues` (one spatially-uniform value per latent frame, the video mirror
  of `AudioDenoiseMask.Values`) and `SourceFrames` (raw frames instead of an encoded clip). A binary temporal
  boundary cannot survive `MaskVideo`'s lossy codec — mask values quantize upward, so a smeared black partly
  denoises rows meant to be preserved — and a caller holding decoded frames no longer pays an encode/decode round
  trip to hand them back.

## alpha.75

- Image: **Kohya SD1.5/SDXL LoRAs that use LDM block names now load.** `LoraFormatDetector` recognized only the
  diffusers spellings (`lora_unet_down_blocks_` / `up_blocks_` / `mid_block_`), so a file whose UNet keys are
  `lora_unet_input_blocks_` / `output_blocks_` / `middle_block_` — what sd-scripts emits, and what the large
  majority of community SDXL LoRAs on CivitAI ship — fell through every arm and was refused at load as
  "Could not detect LoRA format". Detection alone was not enough: the loaded UNet dict is diffusers-named, so
  those keys are mapped through `SdxlCheckpointConverter`/`Sd15CheckpointConverter.ConvertUNetKey`, the same
  LDM→diffusers map the checkpoint itself goes through. The text-encoder halves (`lora_te1_`/`lora_te2_`) were
  already correct and are untouched. Verified against a stock CivitAI SDXL LoRA: all 986 modules (722 UNet,
  72 CLIP-L, 192 CLIP-G) resolve to keys the converted checkpoint holds, and the merged image is a large
  visible change from the base (SSIM 0.34) while strength 0 is inert to the pixel (SSIM 1.0).
- Image: LoCon/conv LoRAs for those families reach the resnets, whose LDM sub-keys (`in_layers`, `out_layers`,
  `emb_layers`, `skip_connection`) are compound names the underscore→dot pass used to split into keys that match
  nothing — a partial merge that no error reports, since the attention layers still merge.

## alpha.74

- Audio: YuE2 songs can run to **900 seconds**, up from a hard 360. Nothing in the checkpoint required the old
  cap — 25 tokens a second against a 24,576-token context holds about 980 s minus the prefix — but three
  separate constants enforced it, and the duration knob was inert above 360 s regardless because the release's
  9,000-token sampler preset won every `Math.Min` against it. `Yue2Protocol` now separates the release's
  default from the ceiling it will accept. Measured at 479.9 s of real music from extended lyrics and 843.1 s
  against a forced budget. Raising the ceiling does not make the model write longer songs — length is decided
  by the lyrics and the score — it stops one that wants to be longer from being cut off.
- Audio: a prompt that leaves less context than the requested duration now **shortens the song and says so**
  instead of throwing. The old check raised after the score had already been planned, so an over-long lyric
  cost ~20 s of planning and then failed; the budget is now fitted to whatever the prefix leaves (the longer of
  the two branches under guidance) and travels back as `Yue2Result.BudgetSeconds` and `meta.budgetSeconds`.
  This is a deliberate divergence from the release, whose sampler refuses the case outright with "no implicit
  truncation" — our caller-facing knob is a duration ceiling rather than a token count. A request that already
  fits is unaffected token for token, which the identical 360 s output confirms.
- Audio: the Oobleck VAE decodes in **bounded tiles**, as the release does by default and we did not — 1,024
  frames of core with a 16-frame halo. Every core sample carries its whole input support inside its own tile,
  so there is no crossfade and the samples are identical to a whole-song decode; what changes is that peak
  activation memory is set by the tile rather than by the song. Peak VRAM is 18.4 GB at 360 s and flat at
  ~22.0 GB past 480 s.
- Audio: **`OobleckConfig.DecodedLength`** — a decode is `frames × HopLength` only when every stride is even.
  A transpose conv here runs `k = 2s, padding = ceil(s/2)`, which emits `sL` for an even stride but `sL − 1`
  for an odd one, and YuE2's stride of 5 sits under a further 64× of upsampling: every YuE2 decode is exactly
  64 samples short of the round multiple. Tiling that assumed the multiple overran its last tile. The older
  all-even presets (Stable Audio Open, ACE-Step 1.5) are unaffected, which is why an all-even test config could
  not have caught this — the tiling tests now carry an odd-stride config for that reason.
- Audio: the score planner's repetition-penalty window is reachable from a request
  (`MusicRequest.Yue2AbcPenaltyWindow`), matching ComfyUI's PR 16293. Its 1-100 validation is unchanged: that
  mirrors the release's own bound, which their node does not re-check.
- Not ported from ComfyUI PR 16293: the `decode_buffer`/`rotary_buffers` work is CUDA-graph capture address
  stability, and graph decode is a measured 23% regression on YuE2.
- Perf note: alpha.74 measures 98.7 / 98.8 / 98.9 s on the standard 360 s request against 98.6 / 99.0 s for an
  unmodified alpha.73 tree in the same session — tiling costs nothing. The 97.1 s recorded for alpha.73 was
  taken in an earlier session; between-session spread is ~2% and a 1-2 s difference cannot be attributed to
  code across one.
- Tests: `Yue2BudgetTests` covers the duration/context arithmetic and the multi-chunk split with no checkpoint
  or GPU; `OobleckVaeTests` gains tiled-vs-whole-song equality (which is what validates the halo) across even
  and odd strides, and the exact-length rule.

## alpha.73

- CUDA: `ApplyRopeSingle` accepts an F16 activation against an F32 cos/sin table, matching the asymmetric
  contract `ApplyRope` already honours. `dit_rope_f16` is indexing-identical to its F32 twin and declares the
  table as `const float*`, so the earlier crash chasing this was an F16 TABLE being over-read by exactly 2x,
  not a layout mismatch and not a missing kernel. It also now rejects a non-rank-4 tensor: head count and head
  dim are read from `Shape[2]`/`Shape[3]`, so a head-major input silently rotated the wrong rows.
- Audio: YuE2's acoustic attention block runs at the key/value buffers' dtype end to end — the input norm, the
  q/k/v projections, the per-head qk-norm, RoPE and the permutes. Its weights are BF16, so an F32 activation
  was cast down on every projection, the same inefficiency alpha.69 fixed in the feed-forward; RoPE was the
  one stage blocking it. Q now reaches the fused attention entry already at its dtype, removing the per-layer
  cast alpha.71 added. The grouped K/V step back to F32 only for `KvCacheAppend`, which narrows from F32 and
  has no F16-source form. The acoustic pass goes 23.5s to 22.4s and a 236-second song 98.2s to 97.1s.
- Tests: `RopeSingleF16Tests` covers F16-vs-F32 rope at full and partial rotary, and pins both halves of the
  contract — an F16 table and a non-rank-4 input must both be rejected.

## alpha.72

- CUDA: causal PREFILL no longer runs on the decode-tuned flash kernel. `GenericTransformer` issues both
  shapes through `IBackend.FlashAttention`, but prefill is thousands of query rows rather than one against a
  long cache, and that kernel is tuned for the latter: measured at 4,096 tokens (D=128, 16q/8kv, RTX 4090)
  64.8 ms a layer against 9.5 ms through cuDNN's fused engine, and 6.74 s across 28 layers on YuE2's
  8,664-token prefill. The plain causal prefill now takes the fused engine, carrying the causal rule as an
  additive bias built on the device by a new `lm_causal_bias_mask_f32` kernel — a host fill is 75M floats at
  that length (~0.2-0.3 s), more than the attention it accelerates. Sliding windows fold into the mask; a
  soft-cap, attention sink or ALiBi has no bias-shaped equivalent and keeps the general kernel, as does a
  non-tight K/V buffer or a head dim cuDNN cannot take, and any cuDNN failure falls through.
- Audio: a 236-second YuE2 song generates in 98.2s, from 107.2s — 0.416 s per second of audio against the
  reference implementation's 0.473 on the same lyrics. The acoustic stage drops 29.8s to 23.5s, but that is
  the per-chunk AR prefill which the progress buckets count there: **the acoustic transformer itself is
  unchanged**. The semantic pass goes 105.0 to 108.6 tok/s and the ABC planner 20.0s to 19.3s, both from
  their own prefixes' prefill.
- Tests: `CudaFlashAttentionTests.CausalPrefill_MatchesCpuReference` checks the fused path against
  `AttentionReference` at 512 tokens with and without a query offset, GQA and MHA, and with a sliding window
  — the pre-existing prefill case runs at 7 keys with no offset, which cannot reach the path and whose oracle
  (SDPA plus an explicit mask) is what the path itself uses. It also asserts cuDNN actually engaged, since a
  gate that silently fell back would satisfy every numeric assertion without running the new code.
- Kernels: `Kernels/lm/build.sh` compiles the two hand-tuned kernels but no longer INSTALLS them unless
  `--install-tuned` is passed. `lm_f32` and `mul_mat_vec_q6k_q8_1` are LLM-decode hot paths held at a tuned
  register allocation, and a routine rebuild to add an unrelated kernel to that domain silently overwrote
  both with ~1,550 lines of different codegen — caught here only because the artifacts showed up in
  `git status`. They still compile, so the drift check still covers them.
- Tests: `CudaKernelDriftTests` rebuilds all 9 kernel domains from source and compares against the committed
  PTX. `dotnet build` never compiles a `.cu` — MSBuild only copies the artifacts and there is no nvrtc
  fallback in the runtime — so an edited kernel whose PTX was not regenerated keeps running the old code
  silently, which shipped 8 stale kernels once already. Skips rather than fails when the local toolchain
  cannot reproduce the artifacts.

## alpha.71

- Audio: YuE2's acoustic pass holds one chunk's attention keys and values for the whole ODE solve instead of
  rebuilding them per velocity evaluation. The AR prefix's K/V are the same on all 64 evaluations of a 32-step
  midpoint solve, but were being re-concatenated and re-widened to full heads every time — ~8,700 identical rows
  a layer. They are now written once per chunk, each evaluation appends only its own rows at the tail, and the
  buffers are stored at F16 where the backend has F16 kernels, which lets the attention take cuDNN's native-F16
  entry and cast nothing (the F32 route re-narrowed the whole key and value on every call).
- Audio: YuE2's acoustic RoPE tables are built once per chunk rather than per velocity evaluation. They depend
  only on the chunk's token span, and building them is a host loop over `tokens * head_dim/2` Pow/Cos/Sin triples
  — 378,000 of them per evaluation, 64 times a chunk, for identical values.
- Together the acoustic pass goes 32.0s to 29.8s on a 236-second song and the whole generate 108.9s to 107.2s.
  The six parity gates still pass, including the acoustic velocity and the full 32-step solve against the
  reference, so the F16 key/value storage is within the gates' bar.

## alpha.70

- Audio: YuE2's autoregressive passes project and sample only the contiguous id window their phase can draw from.
  Every id outside a phase's span is masked to -inf before sampling, so the semantic pass was spending a
  184,704-wide head GEMM, a 739 KB device-to-host copy and ~8 full-vocabulary host scans per token to choose among
  32,769 candidates. The semantic head is now an owned BF16 row-slice (134 MB) covering `[MusicEnd, CodecOffset +
  CodecSize)` and the sampler works in window coordinates, so all three costs shrink 5.6x. Measured on a
  236-second song: the semantic pass goes 90.1 to 105.3 tok/s (65.5s to 56.1s) and the whole generate 119.0s to
  108.9s. The ABC phase keeps the full projection — its span is not contiguous — but shortens its read-back.
- Audio: the six YuE2 parity gates now check the semantic phase over its head window rather than the full
  vocabulary, which is every logit that phase's sampler can reach; A3's greedy tokens still match the reference.

### Measured and rejected

- CUDA-graph decode for YuE2's AR is a **23% regression** (105.3 to 81.1 tok/s) and was not kept. Capture succeeds,
  but the win it targets is not there: eager decode's ~500 per-token kernel launches already overlap with GPU
  execution, so collapsing them buys nothing, while the device-position graph ops are slower than the eager
  kernels for this geometry. The 6.5 us/launch cost measured from a tiny-kernel loop only bites when the GPU is
  idle — it is not a per-token tax on a decode step that keeps the device busy.
- An F16 KV cache for the AR changes the per-token body cost by 0.8% (9.076 to 9.001 ms), so YuE2's attention is
  not KV-bandwidth-bound at song-length contexts and `KvCaches.F16Enabled` is not a lever here.

## alpha.69

- Audio: YuE2's acoustic feed-forward runs its activations at F16 where the backend has F16 kernels. Its weights
  were already BF16, so an F32 activation was being cast down on every one of the three projections; at F16 the
  GEMM stays 16-bit end to end. Worth ~1.2s on a full-length song's acoustic pass (33.0s to 31.8s) — less than an
  isolated op benchmark predicted, which is worth recording: op timings that reuse warm tensors overstate what the
  same change does in situ. Parity improved rather than degraded (velocity corr 0.999809 to 0.999825, nrms 1.96%
  to 1.88%), as expected — the release runs the whole transformer in bfloat16, so this moves toward its numerics.
  The residual stays F32; it accumulates over 28 layers and rejoining it costs one 0.08 ms cast.

## alpha.68

- Audio: YuE2's acoustic stack ran its attention through `IBackend.FlashAttention`, whose kernel is tuned for LLM
  decode — one query row against a long cache. The acoustic pass is the opposite shape: every frame of the song
  is a query row, attending bidirectionally over the whole AR prefix. Routing it through
  `ScaledDotProductAttention`, which reaches cuDNN's fused engine, is **26x** on that op (74.6 ms a layer to 2.8 ms
  at 2308 frames on a 4090), and attention was ~89% of the stack. A full-length song's acoustic pass drops from
  ~816s to 33s and a 3.9-minute song end to end from **17.8 minutes to 119 seconds**. The fused engine is MHA-only,
  so the grouped KV is widened to full heads first; that copy costs ~0.2% of what it buys. All six parity gates
  hold (velocity corr 0.999809, 32-step solve 0.999939 — unchanged to five decimal places), which is expected:
  F16 attention ingest matches the release, which runs the whole transformer in bfloat16.

## alpha.67

- Audio: YuE2 — lyrics- and style-conditioned song generation at 48 kHz stereo, up to six minutes. Despite the
  name it shares no architecture with YuE v1: a Qwen3-geometry autoregressive LM plans an editable ABC score and
  emits one semantic codec token per 25 Hz frame, then a second stack of identical geometry but its own weights
  flow-matches 64-channel acoustic latents while attending to the first model's per-layer KV cache as a
  bidirectional prefix, and an Oobleck VAE decodes those to audio. Loads the Comfy-Org single-file repack, which
  carries both stacks, the VAE and the tokenizer in one checkpoint, so there is no subfolder fetching and no side
  assets. Weights are CC BY-NC 4.0. The `int8_convrot` repack is refused by name rather than silently falling back
  to a different precision than the caller asked for.
- Engine: `MusicRequest` gains YuE2's own knobs — `Yue2Cot` (planning mode), `Yue2Abc` (render a score verbatim,
  skipping the planning pass), a separate sampler block for the score planner, which samples far cooler than the
  semantic pass (`Yue2AbcTemperature`/`TopP`/`TopK`/`RepetitionPenalty`/`MaxTokens`), plus `Yue2PenaltyWindow` and
  `Yue2MinTokens`.
- CLI: `hartsy music` was building its request from four fields, so `--steps`, `--cfg-scale`, `--temperature`,
  `--top-k`, `--top-p` and `--repetition-penalty` were unreachable from the command line for *every* music model.
  All six now reach the engine, alongside the new `--cot`, `--abc` (a file path or inline notation),
  `--abc-temperature`, `--abc-top-p`, `--abc-top-k`, `--abc-repetition-penalty`, `--abc-max-tokens`,
  `--penalty-window` and `--min-tokens`.
- API: the audio endpoints serialise `AudioResult.Meta`, which the response shape already promised to mirror but
  silently dropped. Music generations carry the model, seed and channel count, and YuE2 adds its planned score and
  two truncation flags — a request for a duration the token budget cannot cover previously returned a song that
  stopped mid-phrase with the only warning in a server-side log.

## alpha.66

- Images: Qwen-Image-Edit accepts more than one reference image. `ImageRequest.ReferenceImages` carries the extra
  references, presented after `Img2Img.InitImage`, so a prompt that says "the first image" / "the second image" is
  addressing that order; up to three are consumed, matching the slots the edit-plus template was trained with. The
  Qwen-Image recipe now also builds the text encoder's Qwen2.5-VL vision tower and conditions on the full
  `Picture N: <|vision_start|>…<|vision_end|>` edit template, and a 2511 checkpoint's `index_timestep_zero`
  reference method is honoured instead of being left off. References keep their own aspect-preserving rescales
  (~1 MP to the VAE, ~384² to the vision tower) rather than being squashed to the output size, which changes the
  single-reference path too. Reference images with an explicit denoise mode are refused by name instead of being
  silently dropped.
- CLI: `hartsy image --init-image-mode denoise|reference|auto` selects how an init image is consumed, and a
  repeatable `--reference-image` adds the extra edit references.

## alpha.65

- Engine: `ImageRequest.RemoveBackground` cuts the subject out of a finished generation. RMBG-1.4's matte lands
  in the new `ImageResult.Alpha` plane and the generated RGB is left byte-for-byte untouched, so a consumer
  composites the partial-coverage edge exactly once instead of receiving pixels already blended toward a
  background. The stage runs last — after the refiner and segment-refinement passes, over the pixels the caller
  actually receives — and is family-independent, so no recipe has to declare it. It shares `VisionService`'s
  weight cache rather than loading a second copy of the net, and it is not gated on denoise strength: a
  strength-0 img2img request still gets its cutout.
- CLI/API: `hartsy image --remove-background`, and `removeBackground` on `/v1/native/images`. Both encode
  color-type-6 (RGBA) PNGs when a matte is present and color-type-2 otherwise.

## alpha.64

- Audio: MiniMax Music 3's one-time GGUF quant caches (language model and depth decoder) are written beside the
  weights they derive from, under `AudioModelCache.CacheRoot`, instead of a hardcoded `~/.cache/hartsyinference/`.
  They are multi-gigabyte files and were landing on the boot drive regardless of how the cache was relocated;
  they now follow the same root resolution the weights themselves use (`paths.modelCacheRoot`, else
  `paths.modelsRoot/audio`, else the user cache directory). An existing cache re-quantizes once on the next
  `:q8`/`:q4` run, after which the old directory can be deleted.

## alpha.63

- Krea 2: the recipe honors the `ImageRequest.Components` Qwen text-encoder and VAE picks instead of always
  loading its own pinned side models, resolving them through `ModelFileLocator.Require` so an unresolvable pick
  is refused by name. The pinned fallbacks auto-download when absent, the contract `LtxVideo2Recipe` already
  uses and what SwarmUI's own `RequireClipModel`/`DoVaeLoader` do. First of the `TODO(E-IMG-4)` recipes closed.
- Models: `SideModels.Qwen3VL_4B` moves to the flat `text_encoders/qwen3vl_4b.safetensors` that SwarmUI's Comfy
  backend downloads under the identical SHA-256, so the two share one file instead of fetching 5 GB twice. The
  `Krea2/` subdir it previously used matched nothing on either side.
- Engine: `RecipeContext.Cancel` carries the originating request's cancellation token into recipe construction, and
  Krea 2 honors it when acquiring a side model. The fallback download it newly enables can transfer several
  gigabytes, so an HTTP client disconnect would otherwise have left it running with the request already gone.
- Models: `ModelAsset.LegacyTargetNames` records the names an asset was saved under before its canonical name
  changed, and `ModelDownloader.TargetPath` resolves to an existing legacy file when the canonical one is absent.
  Without it, renaming a shared asset costs every upgrading install a multi-gigabyte re-download and makes the
  recipes that resolve it through the strict non-downloading overload (Mage-Flow shares Krea 2's encoder) fail as
  though the file were missing. A fresh install still downloads to the canonical name.

## alpha.62

- Diffusion: Z-Image generates again. Since alpha.42 every generation threw a `NullReferenceException` on its
  first denoise step: the new per-step preview snapshotted the transformer's fixed CUDA-graph latent, which
  only exists on the step-graph route, and that route is default-off for Z-Image. The preview now unpatchifies
  the loop's own packed tokens through the backend op, so it works on both routes and keeps the tokens
  device-resident instead of draining them to the host every step. `SnapshotGraphLatent` now names the unmet
  precondition instead of dereferencing null.

## alpha.61

- Prompting: SwarmUI's 2026-09-01 parser update hands every backend Swarm tags in place of Comfy-native prompt
  syntax — `(word:1.5)` arrives as `<weight[1.5]:word>`, `[a|b]` as `<alternate:a,b>`, `[a:b:N]` as
  `<fromto[N]:a,b>`. The engine read those as prose, so SD1.5/SDXL prompt weighting was silently inert and the
  tag text was tokenized into the conditioning. `PromptTagFlattening` converts the weight tag back to the
  `(text:N)` grammar `PromptWeighting` implements and collapses `alternate`/`fromto` to their step-0 value,
  running in `ImagesService`/`VideoService`/`MusicService` ahead of every pipeline and of region/segment
  parsing; every other tag passes through byte-for-byte.
- Diffusion: real per-step prompt scheduling. SD1.5 and SDXL declare `ImageFeatures.PromptScheduling` and keep
  the scheduling tags raw, so `PromptTagScheduling` and `WeightedConditioning.Build{Single,Dual}ClipScheduled`
  build a multi-variant `ConditioningSchedule` — one encode per distinct (positive, negative) variant pair that
  occurs, not the full cross product. `fromto` thresholds are a 1:1 port of the reference `SwarmText.py` float
  comparison, so a fraction lands on the same step and a `when` above 1 stays an absolute step index. The old
  `PromptScheduling` (Comfy bracket grammar, never wired into a pipeline) is removed.
- Diffusion: SDXL's pooled/ADM conditioning follows the prompt schedule. The denoise loop switched hidden states
  per step but kept passing the single pooled encode to every UNet and ControlNet call, pairing a later variant's
  hidden states with variant 0's ADM vector; `ConditioningSchedule.PooledVariants` now carries one pooled tensor
  per variant. Null keeps the single encode, which is the unscheduled path and the only option for SD1.5.
- Prompting: `<weight[N]:text>` collapses to its inner text for architectures without per-token weighting rather
  than becoming `(text:N)`. The parens form is only meaningful where a tokenizer applies the weight; emitting it
  to an LLM-conditioned DiT handed the literal digits to Qwen/T5/Gemma as prose. `ImageFeatures.PromptWeighting`
  gates it and only SDXL/SD1.5 declare it; video and music strip unconditionally. The weight itself is still
  unimplemented for LLM encoders, so it is dropped rather than applied for them.
- Diffusion: a scheduled conditioning build that fails partway — an OOM on the third variant, say — disposes the
  tensors it already encoded instead of leaking them, since the schedule that would own them is never returned.
- Diffusion: `WeightedConditioning.HasWeightingSyntax` no longer treats a bare `[` as weighting syntax.
  Brackets carry no grammar now, and counting them put bracket-bearing prose on the schedule path — which
  forfeits SD1.5's fused Euler loop and made any non-default sampler selection fail outright.

## alpha.59

- Benchmarks: a standalone `hartsy-bench` runner produces reproducible community evidence from frozen,
  hash-pinned text and image workloads. Each session runs in its own process with immutable attempts and a
  resumable journal, and every trial retains its raw timings, native token trace and saved output. `validate`
  checks protocol and evidence rather than trusting submitted numbers, `export`/`extract` move data-only
  bundles carrying a full hash inventory, and `publish` builds the static explorer from reviewed evidence
  only. Trusted workflows validate results PRs without executing contributor code. `--cache` resolves as
  `--cache`, then `$HARTSY_BENCH_CACHE`, then `~/.cache/hartsy-bench`; the content-addressed
  `<cache>/<sha256>/<file>` layout lets a checkpoint already stored elsewhere be hard-linked in under its
  pinned hash instead of downloaded again.
- Engine: optional generation diagnostics. `EngineOptions.Diagnostics` is null by default, so
  `StartDiagnostics` returns 0 and `ReportDiagnostic` returns on its first comparison, and the per-token hook
  is only built when an observer is active. An observer that throws is disabled once rather than being allowed
  to alter inference.
- Core: a knob profile that pins a nullable knob to `null` now keeps that null instead of falling through to
  the machine's `HARTSY_*` override. This also corrects `--profile reference`, which pins
  `numerics.cfgInterval` and `numerics.sagePv` to null and until now silently inherited either variable from
  the environment, defeating the profile's stated purpose. Value-type knobs are unaffected.

## alpha.58

- Restore: SeedVR2 runs on the CPU backend. Its kernels are F32-only (`Linear` casts a half weight on the fly but
  not the bias, and the VAE's 3-D convs do not cast at all), so the fp16 checkpoints stopped at the first GEMM.
  `RestoreService` now casts the DiT and VAE tensors to F32 once at load when the backend is CPU (about 13.5 GB for
  the 3B model) and releases the copies with the pipeline; CUDA still consumes the half weights directly.

## alpha.57

- Vision: Real-ESRGAN x2plus produced 4× output. BasicSR's 2× RRDBNet is the same ×4 network fed a 2× pixel-unshuffled
  12-channel input, and `RealEsrganConverter.InferConfig` read the factor from the presence of `conv_up2`, which every
  checkpoint has. The factor now comes from `conv_first`'s input channels, `UpscalePipeline` unshuffles on the host
  before tiling (odd edges replicated and cropped back), and the network always runs both upsample stages.
- Restore: the SeedVR2 catalog's positive embedding is now the upstream `ByteDance-Seed/SeedVR2-3B/pos_emb.pt` (public,
  hash-pinned) instead of a private Hartsy-hosted safetensors that answered 401, so a fresh install's first-use download
  completes. `RestoreService` reads that bare-tensor pickle directly; a `*emb*.safetensors` sibling still works.
- Catalog: `briaai/RMBG-1.4` is no longer gated on HuggingFace; the entry's comment says so and names a byte-identical
  ungated mirror in case that changes.

## alpha.56

- Vision: `VisionMode.Upscale` runs the Real-ESRGAN generator that `HartsyInference.Vision/Upscale` has carried
  without a caller. Three catalog ids fetch the official BasicSR checkpoints on first use — `real-esrgan-x4plus`,
  `real-esrgan-x2plus`, `real-esrgan-anime6b` (`Models/Vision/Upscale/`) — and the request's new
  `TargetWidth`/`TargetHeight` fit the result: enough passes to cover the target (at most two), then a bicubic
  downsize, never a stretch past what the last pass produced (`UpscalePlan`). Tiled at 256 px input.
- Vision: `ImageData.Alpha`, an optional 8-bit straight coverage plane. `VisionMode.BackgroundRemoval` now fills
  it with the RMBG-1.4 matte beside the gray composite the image→3D preprocessors keep reading, so a caller can
  build a real cutout. `PngEncoder.Encode(ImageData)` writes colour type 6 when the plane is present; the
  `/v1/native/vision` route and `hartsy vision --mode removebg` both return RGBA for it.
- CLI: `hartsy vision --mode upscale --width/--height`; the mode is inferred from an `esrgan`/`upscale` model id.

## alpha.55

- Wake: a satellite may declare `"width": 1` in its `hello` and send G.711 µ-law, one byte a sample instead of
  two. Not about a link's average throughput — a device on a marginal link loses audio in stalls, where one
  dropped packet costs a retransmission timeout of about a second and everything queued behind it is dropped.
  A send buffer covers a fixed number of bytes, so halving the bytes doubles the seconds it covers. µ-law
  rather than 8-bit linear because a satellite microphone can sit at a few hundred counts out of 32768, which
  linear truncation would quantize to two or three levels. Width 2 remains the default and is unchanged.

## alpha.54

- Wake: `WakeServiceOptions.HostHandlesTurns`, and a settable `WakeService.HostHandlesTurns` to match, put
  `"handled":true` on every `transcript` frame. A satellite reads it as "do not answer this yourself". Without
  it a host that answers turns has no way to tell a device to stand down in time: `Detected` is raised after
  the transcript frame is written, so by the time a subscriber could send anything the device has already
  started its own assistant call, and two replies end up sharing one audio ring. Off by default; the frame is
  byte-identical to before when it is off.
- Wake: `WakeService.BeginAudio` returns a `WakeAudioStream` — one spoken reply, written in as many pieces as
  it arrives in. A reply synthesized sentence by sentence used to go out as one `SendAudioAsync` call per
  sentence, and each call numbered its frames from zero and marked its last one final, so the device read every
  sentence after the first as a new reply, reset its playback ring and cut off the one before it. The stream
  also holds back a piece's ragged tail rather than sending a frame with an odd number of bytes, and closes
  itself on the way out of an abandoned turn. `SendAudioAsync` still works, and is now one of these.

## [Unreleased]

### Added
- **Spoken audio can travel on the wake socket.** `WakeFrameCodec.WriteAsync` gained an overload that writes a
  header plus raw bytes — the same header-then-payload shape a satellite has always used to send audio, now in
  the other direction — and `WakeService.SendAudioAsync` pushes a reply through it in 40 ms frames.

  Speech reaches a device over HTTP today, which costs it a second connection and a second protocol per turn.
  The socket it is already holding open can carry it.

  The server does the pacing, which is the part that is not obvious. A device paces an HTTP body by withholding
  TCP acknowledgements, but doing that here would also stall the `ping` and `detection` frames queued behind
  the audio on the same connection, and a satellite that stops answering pings is dropped after twenty seconds.
  So the server writes a little ahead of real time — 400 ms — and never further.

  Header and payload go out under one lock and in one pair of writes. A header promising bytes that never
  arrive desynchronizes the stream permanently, because the reader then takes the next frame's header as
  payload; that exact failure has been seen on this protocol in the other direction and it cost a night to
  find. The round trip is tested, not just the bytes.
- **`BackendFactory.ProbeCuda`** runs a real matmul on the GPU and checks the answer, instead of asking whether
  a GPU exists. `CudaContext.IsAvailable` answers the second question, and everything between it and a working
  backend is untested by it: kernels built for another architecture, a PTX directory that did not ship, a card
  with no free memory, a driver and toolkit that disagree. Each of those says available and then throws on the
  first real operation, in the middle of somebody's request.

  Found on the first run, on the machine it was written on: `IsAvailable` returns true, `Resolve("auto")`
  returns `cuda`, and the shipped kernels target sm_80 against an sm_75 card, so the driver's JIT refuses every
  one of them. `ResolveProbed("auto")` returns `cpu` there, with the reason.

  A wrong answer counts as a failure too, not just a throw — a GPU that computes the wrong thing is worse than
  one that stops, because nothing downstream notices. Cached after the first call, and opt-in: `Resolve` is
  unchanged and stays cheap.
- **`status` frames on the wake socket.** A voice turn is several seconds spread across transcription,
  generation and synthesis, and from a satellite's side all of it looks the same: it sent audio and nothing has
  come back. So its light stayed on one colour for the whole wait, and a user with no screen could not tell
  "still working" from "did not hear you" — and would repeat themselves into the middle of a reply.

  `WakeService` now emits `captured` (the speaker can stop talking) and `transcribing` as it reaches them, plus
  `error` and `done`, and `SendStatusAsync` is public so whoever owns the turn after the transcript leaves can
  send `thinking` and `speaking` too. `WakeStatus` names the states and builds the frame's data object, so the
  wire contract a C++ satellite parses is one function with a test on its exact bytes rather than an
  interpolation at four call sites.

  Advisory throughout: a device that does not know a state ignores it, an unreachable device is not an error,
  and no turn fails because a light could not be updated.
- **Piper streams a sentence at a time.** Piper is a whole-utterance model — nothing comes out until every
  phoneme of the text has been through the decoder — so a spoken reply used to begin only once all of it had
  been synthesized. It now implements `IStreamingTtsRunner` by splitting on sentence boundaries and yielding
  one `AudioChunk` per sentence, so the first one can be played while the rest is still being made.

  Measured on the four-sentence passage from the latency audit, 8-core CPU: **8.315 s** to synthesize whole,
  **1.797 s** to the first sentence, 22% of the wait. Total synthesis also fell to 4.058 s, because VITS cost
  grows faster than linearly in sequence length — so this is a throughput win as well as a latency one, which
  was not the point but is not unwelcome.

  The audio is not sample-identical to the whole-text call: each sentence gets its own prosody contour. The
  non-streaming `Synthesize` path is unchanged and still passes the text through in one piece.
- **`SentenceSplitter`** (Audio, `Frontends`) cuts a passage into sentences for exactly that. Cutting where a
  sentence does not end is the expensive mistake — the halves are voiced separately and the seam is audible —
  so it declines to cut on abbreviations, initials, decimal points, and anything not followed by something that
  looks like a new sentence, and it merges a fragment shorter than 24 characters into what follows it.
- **End-of-speech detection on the wake path.** After a wake word fired, the service waited a fixed three
  seconds and then transcribed the preceding eight — so a question longer than three seconds was cut off
  mid-word, and a two-word command still cost the full three seconds. Measured on a real satellite before this
  change: "Hey Jarvis, can you tell me what the weather is going to be like this afternoon and whether I should
  bring an umbrella…" came back as "…like this after".

  `SileroVadStream` has been implemented and parity-tested since it was written but was wired to nothing. It
  now runs per session on the wake worker's own thread and backend, alongside the detection pipeline and the
  denoiser, and `WakeService` ends the utterance on `EndOfSpeechSilenceMs` (500 ms) of silence instead of a
  fixed wait — capped at `UtteranceSeconds`, so someone who never stops talking still gets an answer.
  `UseEndOfSpeech` turns it off; without VAD weights installed the old fixed wait is still the fallback.
- **`WakeEvent.Command`, and `command` on the device frame.** Transcription covers the wake word and the
  command together, so the transcript begins with the words that woke the device — "Hey Jarvis, what time is
  it?" — and a small model will answer the greeting instead of the question. The engine knows which head fired,
  so it now reports the command separately. The full transcript is still sent; nothing is silently edited.
- **`OnnxWeightLoader.SubgraphConstants`** reads the float `Constant` values of a nested graph, and the ONNX
  parser now walks node attributes for subgraphs. Silero VAD keeps its weights this way — inside the
  `then_branch` of an `If` on sample rate rather than as graph initializers — so an initializer-only reader
  finds nothing in the file at all. `WakeModelSet` can therefore load `vad/silero_vad.onnx` straight from its
  canonical MIT source with no conversion step and nobody hosting a repacked copy; `vad/silero_vad_16k.safetensors`
  is still read when present. Verified by scoring both formats through the service's own loader and requiring
  identical probabilities.
- **`tools/convert_silero_onnx.py`** emits the safetensors form and, with `--verify`, checks the export against
  onnxruntime end to end (1.25e-6 max abs over 343 chunks).


### Fixed
- **End-of-speech waited for the silence window twice.** The utterance clock was driven by
  `SileroVadStream.InSpeech`, which describes a segment rather than a moment: it stays true for the stream's
  own `minSilenceMs` after the speaker stops. `WakeService` then waited its own `EndOfSpeechSilenceMs` on top,
  so with both set to 500 ms every command paid a full second of silence before transcription even started.
  The stream now exposes `LastChunkWasSpeech`, the per-chunk verdict with hysteresis applied, and the session
  times silence from that. Measured on a real satellite before this change: detection to transcript 3.95 s on
  "what time is it?", of which a second was this double wait.
- **A question longer than 6.5 s was still truncated.** The end-of-speech wait was capped at
  `UtteranceSeconds - LeadInSeconds`, which quietly spent the lead-in allowance out of the speaker's time
  rather than out of the buffer it actually comes from. The cap is now the whole of `UtteranceSeconds`, and
  that default moves from 8 s to 12 s — a spoken question runs longer than it reads, and the capture buffer
  holds fifteen. Measured before: an 8.6 s clip came back as "…tomorrow afternoon and", losing its last four
  words. After: transcribed whole.

### Changed
- **The CPU kernels now use every core.** `HartsyInference.Cpu` contained no threading of any kind, so a
  synthesis or a transcription ran on one core of whatever machine it was given. Four kernels now fan out
  through a new `CpuParallel` helper in `HartsyInference.Core.Numerics`, and `Conv1d` gained a vectorized
  stride-1 path:
  - `Conv1dKernels.Conv1d` splits over (batch, output channel) and, when a layer has too few output channels
    to fill the machine, over the time axis as well. A vocoder's last convolution has a single output channel
    and tens of thousands of samples, so splitting on channels alone would have left exactly the widest layer
    serial. With unit stride — every 1x1 projection and every dilated k3 residual in a VITS graph — the loops
    are also restructured from a per-element gather into a contiguous vector accumulate.
  - `Conv1dKernels.ConvTranspose1d` is re-expressed output-channel-first. The arithmetic and the accumulation
    order are unchanged, but with the input channel outermost every channel in a group accumulated into shared
    output rows, which no work split could have been made safe.
  - `MatMulKernels.MatMul` and `LinearTransB` split over (row tile x column tile) rather than rows alone,
    because an LLM decoding one token at a time calls them with M = 1.
  - `AttentionKernels.ScaledDotProductAttention` splits over (batch, head, query slice); its one shared scratch
    row, the only thing preventing this, is now allocated per worker.

  Measured on an 8-core dev box, Release, same commands before and after:

  | Benchmark | Before | After |
  |---|---|---|
  | `Bench_Piper` (3.2 s utterance) | 9.872 s, RTF 3.115 | 0.658 s, RTF 0.208 |
  | `Bench_WhisperBase` (3 s clip) | 7.240 s, RTF 2.413 | 2.112 s, RTF 0.704 |

  Small calls still run serially: dispatch costs more than the work below `CpuParallel.MinWorkForParallel`.

### Added
- **`HARTSY_CPU_THREADS`** (`numerics.cpuThreads`) caps the workers those kernels use; 0, the default, means one
  per logical core. It exists because the CPU device gate is a no-op, so an LLM decode and a TTS synthesis
  genuinely overlap on the same box and would otherwise each claim every core. The cap is enforced through a
  shared task scheduler rather than `MaxDegreeOfParallelism` alone, which bounds only one call at a time and
  would let two concurrent generations oversubscribe between them.
- **`SttBenchTests.Bench_WhisperBase`**, the transcription counterpart to the existing TTS benchmarks. Speech
  recognition sits on the critical path of a voice turn and had no benchmark at all, so kernel work could speed
  up synthesis and leave it untouched unnoticed.
- Above-threshold correctness tests for all four kernels (`Conv1dParallelKernelTests`,
  `MatMulParallelKernelTests`). Every pre-existing kernel test runs at 32x32 or smaller, which is below the
  parallel threshold — the whole suite exercised only the serial branch and would have passed regardless of
  what the parallel one did.

### Fixed
- **A missing checkpoint is now one contract: HTTP 400, naming the model the caller asked for.** Selecting a
  model that is neither in the catalog nor on disk reported `Model path not found: ` — an empty path, and no
  mention of the selection that failed. `InferenceEngine.ResolveFamilyId` is reached through
  `SupportedFeatures`/`DefaultsFor` before either construction guard, and handed its null `LocalPath` straight
  to `ModelLayoutResolver`, which could only echo the empty path it was given. Three engine guards now share
  one helper, and every modality service was brought onto the same contract:
  - Text, embedding, mesh, world and restore name the requested model instead of a generic noun.
  - Vision's embed path and its per-annotator `RequirePath` raised `InvalidOperationException`, which no
    `GenerationErrors` arm maps — so an absent checkpoint surfaced as **HTTP 500 `server_error`**. They now
    raise `FileNotFoundException` (400), and `RequirePath` names both the caller's id and the family it
    resolved to rather than a hardcoded label the caller may never have typed.
  - The four vision annotator "not installed" errors (CLIPSeg, YOLO, Grounding DINO, RT-DETR) were 500s for
    the same reason and are now 400s.
  - An empty model id resolved `Path.Combine(root, "Vision", "")` to the modality *folder*, which exists — so
    it passed every null guard and failed deep inside a loader, naming a server path the caller never sent.
    `ModelResolver` now declines a blank id.
  - `WorldService` checked for a checkpoint before its "catalogued but not loadable" cases, so `matrix-game-2`
    and `matrix-game-3` reported a missing file instead of the explanation that no file would help.

### Changed
- **LTX-2.5 defaults are template-faithful.** Distilled checkpoints now run the shipped ComfyUI workflow by
  default: 8-step fixed-sigma base pass at the half grid, learned x2 latent upsample (auto-downloaded side
  model), 3-step refine (`HARTSY_LTX2_TWO_STAGE=0` restores single-pass). A `distilled`-named checkpoint
  selected under the plain `ltx-2.5`/`ltx-2` id auto-routes to that contract with a log line, which also
  makes it reachable from SwarmUI. Default geometry for both LTX-2.5 families is now 1280x736 (two-stage
  decodes 1280x704), 121 frames @ 24 fps — the template's 5-second clip — and the dev family's default
  sampling moved from the never-measured 50 steps / cfg 3.0 to the measured parity profile 20 steps /
  cfg 4.0. Default generations are minutes, not seconds; pass explicit `--width/--height/--frames/--steps`
  for quick turnarounds. The 2.3-lineage ids share the dev defaults.

### Removed
- **`VideoRequest.VideoExtendModel`** — never consumed by any recipe on either side of the contract since the
  DTO's introduction (video extension was explicitly out of scope in the extension's own plan). Breaking
  record-shape change for transports that set it; the SwarmUI extension's mapping was removed in the same pass.

### Added
- **ComfyUI `int8_tensorwise` quantization, resident (± the `convrot` Hadamard rotation).** This is the format
  the *official* Lightricks LTX 2.5 and Comfy-Org MiniMax-H3 quantized releases ship in, and it was previously
  rejected by name at load. Weights now stay int8 on the device at 1 byte/param instead of expanding to BF16,
  so a 21 GB DiT fits a 24 GB card. Format notes:
  [`docs/Research/QUANTIZATION_COMFY_FORMATS.md`](docs/Research/QUANTIZATION_COMFY_FORMATS.md).
  - **`Tensor.QuantInfo`** (`QuantWeightInfo`) carries a packed weight's companions — per-output-row scale,
    ConvRot group size, `full_precision_matrix_mult` — the way `Fp8ScaleFactor` already carried fp8's scalar.
    Attaching them to the weight rather than to a per-model linear wrapper is what lets one backend branch
    serve every model that loads such a checkpoint, with no change to any model's code.
  - **`CudaBackend.Linear`** gained a resident-int8 branch reusing the existing W8A8 chain: activation ConvRot
    (`convrot.ptx`, a radix-4 butterfly — `H` is `kron(h4, …)`, so no matrix is materialized), per-row dynamic
    int8 quant, cuBLASLt IMMA, then the `rowScale·wScale + bias` epilogue. Chunked over rows against live free
    VRAM, because the int32 accumulator is 4 bytes per output element and the weights have already filled the
    card. Short sequences pad to the 32-row IMMA granularity rather than falling back, matching comfy-kitchen.
  - **`Int8ConvRotCodec`** provides the un-rotating dequant for the CPU/Vulkan backends and for layers tagged
    `full_precision_matrix_mult`. Verified against comfy-kitchen's eager reference at **relL2 5.1e-8–2.7e-7**
    with F32 activations; `H` itself matches bit-exactly.
  - **`ComfyQuantDescriptor`** replaces three divergent private copies of the `.comfy_quant` blob parser. The
    per-layer blob is now authoritative over the file-level `_quantization_metadata` mirror, which re-quants of
    the same model disagree with.
  - MiniMax-H3's `int8_convrot` rejection is deleted; `MiniMaxH3Assets` no longer sinks `convrot` filenames.
- **ComfyUI `nvfp4` weights stay resident too.** They were unpacked to BF16 at load, which turned the official
  18.72 GB LTX-2.5 distilled nvfp4 DiT into 42 GB. `Nvfp4Codec.TryAttachResident` now relabels the packed
  weight to `DType.F4E2M1 [N, K]` — the dtype already existed for exactly this — and `CudaBackend` dequantizes
  it in-kernel per GEMM under the existing `CacheWeightCasts` budget. Bit-exact against the host reference on
  real `qwen3vl_32b_minimax_h3_nvfp4_awq` layers. A **VRAM** win only: no consumer GPU here has FP4 tensor
  cores, so the GEMM still runs in F16. Opt-in per caller, since the eager unpack is what CPU/Vulkan need and
  AWQ layers with `pre_quant_scale` must take it regardless.
- **`Tensor.ReinterpretAs`** — a byte-count-validated, keep-alive-rooted dtype/shape view. `Reshape` could not
  serve: it holds element count fixed, and the whole point here is that one U8 byte becomes two F4E2M1 elements.
- **LTX-2.5 pipeline wiring — the Gemma 4 tower is now driven, not just built.** `Gemma4TextEncoder` and
  `Gemma4Tokenizer` existed and were parity-checked but had no consumers; `LtxVideo2Recipe` still constructed the
  Gemma-3 tower and refused a 2.5 bundle with a targeted error.
  - **`ILtx2TextTower` / `ILtx2PromptTokenizer`** name the contract the pipeline was already relying on
    structurally. Both encoders already exposed `EncodeMultiLayer`/`EnumerateWeights`/`NumLayers` with identical
    signatures, so neither implementation changed. `LtxVideo2Pipeline`'s 3840 caption channels and 49 harvested
    states hold for both families.
  - The recipe branches on `model.layers.0.layer_scalar` — **not** on a missing `v_proj`, which would misclassify
    every Gemma 4 checkpoint, since layer 0 is a sliding layer and has one. Gemma 4 conditions at 1024 tokens
    against Gemma 3's 256; that length is part of the conditioning, because the connector replaces learnable
    registers positionally.
  - The tokenizer is built from the `tokenizer_json` U8 tensor LTX-2.5 embeds **inside** its text encoder — there
    is no side file anywhere to fall back to.
  - **Converter routing**: a standalone Gemma 4 tower ships bare `model.layers.*` keys with no `text_encoder.`
    prefix and was falling through to the DiT mapper. Those now route to the text-encoder bucket, checked after
    the connector rule because `text_embedding_projection.*` lives in the same file but belongs to the connectors.
    Packager-embedded `hf_asset__*` side files (chat template, tokenizer/processor config) are dropped — they are
    not weights. Verified on the real 15.37 GB checkpoint: 328 packed int8 tower weights, zero leaking into the DiT.
  - Real-weight verified: the int8-convrot Gemma 4 encoder produces finite, deterministic, prompt-discriminating
    conditioning on a 4090 — identical tokens give bit-identical states, differing tokens diverge.
  - Unchanged on purpose: the diffusion-decoder guard (decode through the conv VAE — the diffusion decoder is
    managed-only), and both documented divergences from upstream (the 49th state's final norm, and padding side),
    which also affect the shipping 2.3 path.
- **LTX-2.5 support (components complete; end-to-end generation not yet wired).** Every piece is ported and
  checked against the reference, but the LTX-2 pipeline still constructs the Gemma-3 tower and the
  convolutional decoder, so a full 2.5 bundle is now refused with a targeted error instead of being
  mis-decoded. Details and the two open questions are in
  [`docs/Research/LTX_2_5.md`](docs/Research/LTX_2_5.md).
  - **Variant detection** replaces the hardcoded `LtxVideo2Config.V23`: `SafeTensorsLoader` exposes a file's
    `__metadata__`, and `LtxVideo2VariantDetector` resolves a config from it with tensor-key probes as the
    fallback — probing whenever the architecture config did not state a value, since repacks routinely ship
    metadata with no LTX config, and letting key presence win for the keyframe marker as ComfyUI does.
  - **Transformer**: `keyframes_abs_pos_embedding` applied to the first latent frame's tokens in all three
    forward paths including the captured step graph, plus a load-time cross-check that rejects a checkpoint
    contradicting the detected variant. The 2.3→2.5 architecture diff is only two config keys.
  - **`IBackend.Na3d`** — 3D neighborhood attention (NATTEN window semantics: the window slides inward at
    borders rather than truncating), verified against the reference at relL2 < 1e-6.
  - **`LtxVideo25DiffusionDecoder`** — the new diffusion video decoder, relL2 2.3e-7 on final pixels and a real
    decode of the shipped checkpoint. Managed-only, so it is a numerical reference rather than a fast path.
  - **`Gemma4TextEncoder` + `Gemma4Tokenizer`** — per-layer alternating geometry (global layers carry one KV
    head at a different head dim, no `v_proj`, and a 25% partial rotary), RMS weights stored directly rather
    than as Gemma 3's `1+w`, and a rank-merge BPE tokenizer that is bit-exact against the HuggingFace library
    on the real 262k vocab.
  - **Distilled sampling**: a checkpoint's baked-in sigma schedule replaces the dynamic flow-match shift, and
    the unconditional branch is skipped when both guidance scales are 1.
  - **Catalog**: `ltx-2.5` and `ltx-2.5-distilled` as separate ids, because the two checkpoints are
    byte-indistinguishable and only the id can carry which schedule was intended.
- **Device-resident CFG+Euler for the image denoise loops** (CUDA image bring-up): Lance, Lumina2, HiDream,
  F-Lite, Kandinsky5, SD3, and Z-Image's fast path now run guidance + the Euler update in-place on device via
  `CfgEulerStep` and the new fused `CfgRenormEulerStep` (Lance renorm), `CfgNormalizedEulerStep` (Lumina2
  `cfg_normalization`), and `AffineMix`/`MaskedAffineMixInPlace` (SD3 img2img noise + masked-inpaint blend/
  recomposite) — replacing per-step scalar host loops over `DataPointer`. New `MixContract`/`PatchTokenContract`/
  `SplitContract` validate the op geometry identically across backends. BF16 elementwise dispatch (Gelu, Clamp,
  GeGlu, RepeatKvHeads) no longer silently falls through to the F32 kernel — GeGlu gained a real BF16 kernel,
  RepeatKvHeads a 16-bit bit-copy launcher, and unsupported dtypes now throw.
- **Z-Image lifecycle hardening**: Base/Turbo checkpoint-variant detection from the filename, two independent
  prompt-cache layers, a Qwen3 tokenizer rewrite (byte-level encoding gap, `<think>` handling, tokenization-
  boundary fix) golden-tested against HF tokenizers 0.22.2 output, and a CUDA-graph-captured denoise step for
  the packed fast path. Scope note: the packed device-resident loop covers t2i, img2img, and CFG; masked
  inpaint and regional conditioning still run the host-stepped loop (per-step `ApplyZImageCfg`/`scheduler.Step`
  on the CPU) — porting those onto the SD3 `MaskedAffineMixInPlace` pattern is tracked follow-up work.
- **Kernel build reproducibility**: `conv/`, `vision/`, and `wan/` gain the same `build.sh` (nvcc, or the
  committed `nvrtc_compile` fallback) the other kernel domains already had — their shipped PTX previously had
  no scripted rebuild path at all; `dequant/build.sh` now covers `w8a8` (sm_75) and `fp8_quant` (sm_80, per its
  shipped target) and gains the nvrtc fallback; `dit/build.sh` covers `mg3_action`. 13 of 16 artifacts
  reproduce bit-identically from source; `stepcache`/`w8a8`/`wan_vae_norm` are regenerated with the current
  pinned toolchain (all kernel-family GPU tests pass against the regenerated artifacts).

- **Wan 2.2 A14B dual-expert swap through the native contract** (regression restore): `VideoRequest.VideoSwapModel`
  + `VideoSwapPercent` (fraction of steps for the low-noise expert; null = official 0.875/0.9 boundary) →
  `WanVideoRecipe` loads the second DiT and warps the fraction through the flow shift
  (`boundary = s·p/(1+(s−1)·p)`); swap-aware pipeline cache key; CLI `--swap-model`/`--swap-percent`.
- **FLUX.1 Redux through the native contract** (regression restore): `redux.stylemodel`/`redux.multiply`/
  `redux.merge` (number 0..1)/`redux.apply_start` Extra keys drive `ReduxResolver` from `Flux1RecipePipeline`;
  prompt images ride `IpAdapter.PromptImages`; Flux declares `ImageFeatures.IpAdapter` (Redux only — real
  IP-Adapter checkpoints are refused with a clear message). CLI `--style-model` + redux knobs.
- **Wan-Animate driving video**: `VideoRequest.DrivingVideo`/`DrivingPoseVideo`/`DrivingFaceVideo`/
  `DrivingAutoPreprocess` with in-engine YOLO11-pose skeleton render + face crop (ported from the extension's
  dead preprocessors); single-still tiling kept as fallback. CLI `--driving-video`/`--pose-video`/`--face-video`/
  `--no-auto-preprocess`.
- **`VideoFeatures.ReferenceImages/ReferenceVideos/ReferenceAudios/DrivingVideo`** gating bits — closes the
  silent drop of reference conditioning on families that never consumed it.
- **`ImageRequest.InstructPix2PixCfg` wired** to OmniGen2 (default 2.0) and Boogu (default 1.0) dual-CFG edit
  paths; CLI `--ip2p-cfg`.
- **Multi-GPU sharding, placement & parallelism — the full opt-in feature set** (`PlacementConfig` /
  `EngineOptions.Placement`; all-defaults is byte-identical to single-GPU). Works over plain PCIe with
  no P2P/NVLink (host-staged boundaries; P2P used when available). User guide: `docs/MULTI_GPU.md`.
  - **LLM layer split** (`ShardDevices`, or a `"cuda:0+cuda:1"` composite device key / `--device` in the
    CLI): N-way transformer layer split planned from live free VRAM (or explicit `ShardRatios`), per-stage
    asymmetric weight preload, KV cache per-layer on its stage's card, logits/sampler on the last stage.
    Verified: Llama-3.2-1B split = exact token parity vs single-GPU; **Qwen3-32B Q4_K_M (19.8 GB) OOMs a
    24 GB 4090 alone and runs at ~12.1 tok/s split across 4090+3060**. Exclusions: SSM, Gemma-4 PLE, VLM
    sidecars (warned + skipped); CUDA-graph/speculative decode disabled while staged.
  - **DiT block sharding** (`EnableDitSharding`, exactly 2 devices; CLI `--dit-shard-gpu`, extension
    `DitShardGpuId`): diffusion transformer block-range split with pooled (never replicated) weights.
    Verified on real weights: Krea2, Qwen-Image 20B, Flux.1 (plain generations; ControlNet/Kontext/
    inpaint/regional auto-fall-back), Chroma, HunyuanImage 2.1, MiniMax-H3 fp8. Disables step-graph/
    step-cache/block-streaming while sharded; mutually exclusive with CFG-parallel.
  - **Audio-LM layer split + precision policy** (CLI `--lm-shard-gpu`, extension `LmShardGpuId`): YuE's
    7B Stage-1 rides the same layer-split machinery; the load-time Q4_K quantization became a policy
    (`HARTSY_AUDIO_LM_QUANT=q4k|q8|off`) defaulting to **un-quantized bf16 when sharded** — pooled at
    8.7 + 4.3 GB across 4090+3060. `MusicLoadContext` carries shard backends + precision to loaders.
  - **TE/VAE component placement** (`TextEncoderDevice`/`VaeDevice`; CLI `--te-gpu`/`--vae-gpu`): Wan
    TI2V-5B 43.7 s → 32.7 s with umT5 on the second card; SDXL SSIM 0.9998; Flux/Qwen-Image/Chroma/
    HunyuanImage/LTX-1/LTX-2 wired. Composes with DiT sharding.
  - **CFG-branch parallelism** (`CfgParallelDevice`; CLI `--cfg-parallel-gpu`): negative branch runs
    concurrently on a second card with replicated weights (~1.8-1.9× per-step concurrency; Wan +
    Flux true-CFG), observably falling back to sequential when the replica doesn't fit.
  - **Same-GPU dual backends**: two engine instances share one physical GPU with isolated
    streams/caches/mempools; serialized per-ordinal by default (`HARTSY_SAME_GPU_CONCURRENT=1` opts into
    concurrent mode, which has a known allocator issue near VRAM capacity — left off).
  - Verification: `tests/run-multigpu-campaign.sh` (real-weight, fail-on-missing-checkpoint) covering
    every mode; measured tables in `benchmarks/results/2026-08-05_multigpu_speeds.md`.
- **YuE full-quality pipeline activated**: Stage-2 (m-a-p/YuE-s2-1B-general, cb0 → all 8 codebooks) and
  the per-stem 44.1 kHz Vocos vocoders are now weights-catalog entries (auto-download); the vocoder
  torch checkpoints auto-convert to safetensors on first load (`EnsureVocoders`, same pattern as
  x-codec). Without them YuE silently degraded to the vocal-cb0-only 16 kHz draft — the "garbled" mode
  (Whisper transcribes it as nothing; the full pipeline transcribes supplied lyrics near-verbatim).
  CLI `hartsy music` gained `-g|--genre` (YuE's prompt is the LYRICS — `[verse]`/`[chorus]` markers —
  and genre carries the style tags).
- **MiniMax-H3 ("Hailuo 03") — full port, real-weight video + audio verified on a 12 GB RTX 3060.** Single-stream
  packed-token DiT (`[text | cond | audio | video]`, hidden 5376, 50 blocks) denoising 24-channel video and 32-channel
  40 Hz stereo audio jointly, with a ViT3D video VAE, a DAC/BigVGAN audio VAE, and an NVFP4-AWQ Qwen3-VL text encoder.
  Verified end-to-end: 512×288 / 39 frames / 30 steps produces a tracking shot of a dog splashing through a stream with
  a 1.625 s stereo soundtrack at −12.5 dBFS peak. The 66 GB bf16 DiT is mmap-backed and loads at 943 MB RSS.
  Layout, dual-shift schedule, audio row packing and final-layer modulation rows are byte-identical to upstream
  ComfyUI master; the audio VAE decoder matches the reference module shipped inside the checkpoint at relL2 4.9e-6
  (CPU) / 1.3e-3 (CUDA), covered by the new `MiniMaxH3AudioVaeParityTests` + `tests/python-reference/
  minimax_h3_audio_vae_ref.py`.

- **SeedVR2-7B support (v1 NaDiT port).** The 7B checkpoint is the V1 architecture (`models/dit`, not the
  3B's `dit_v2`) — same windowing/attention/AdaSingle, but a different RoPE (pixel-basis freqs
  `linspace(1,128,10)·π` matching the checkpoint's `rope.rope.freqs`, 60 of 128 head dims rotated,
  positions normalized to `linspace(−1,1)` over each window axis, applied to video only — text is never
  rotated), plain GELU-tanh MLP with biases, all 36 blocks fully split (no `mm_layers`), and no tail
  norm/ada or last-layer text shortcut. `SeedVr2Config.Detect` now configures all of this from the
  plain-MLP+no-tail signature instead of throwing. Parity: a v1 tiny-config dump against ByteDance's own
  `models.dit.nadit` (`seedvr2_transformer_v1_parity_dump.py`) passes at blocks ≤9.1e-4 / output 8.96e-4
  (`Dit_TinyConfigV1_ForwardMatchesReference_PerBlock`).
- **SeedVR2 BF16 VAE activations** (CUDA default; `HARTSY_SEEDVR2_VAE_F32=1` reverts): the fp32
  whole-clip activation peak that OOM'd 24 GB at 720p-area is halved — 720p-area now restores a full
  25-frame clip at a measured 13.3 GB peak, 960×540-area at 9.0 GB, and **the 12 GB 3060 runs 960×540-area
  end-to-end (peak 7.8 GB)**. Pixel/latent boundaries stay F32; the mid-block attention runs F32 (the
  known F16-attention precision class). BF16 variants of the five `wan_vae` glue kernels; BF16 admitted
  through `SliceRows`/`Permute0213`. Output vs the f32 path: SSIM 0.9998. 1080p-area still exceeds 24 GB —
  tiled/sliced VAE remains open.

- **LTX-2.3 audio guidance rescale — the near-silent soundtrack is ~14 dB louder.** Root cause measured:
  CFG over-disperses the audio latent (σ 0.89 at guidance 1 → 2.22 at 3 → 2.69 at 7, against the
  checkpoint's own 1.17) and the decoded level falls with it; the audio VAE is not at fault (fed a
  training-distribution latent it decodes to a healthy level). `AudioGuidanceRescale` (default 1.0) applies
  diffusers' `rescale_noise_cfg` to the audio stream, restoring σ to 1.141. Real 20-step generation
  (512×320×25f, seed 42): **peak −43.9 → −28.2 dBFS, RMS −59.4 → −45.4 dBFS**, video unchanged.
  Implemented as an affine transform so only four scalars reach the host and the step stays on-device.
  Also adds `AudioGuidanceScale` (null = follow the video scale, the reference default) with
  `HARTSY_LTX2_AUDIO_CFG` / `HARTSY_LTX2_AUDIO_RESCALE` overrides. **Still ~20 dB below a healthy
  soundtrack** — the reference's STG + modality-isolation guidance remain unimplemented; see
  MODEL_STATUS_VIDEO. Note for anyone tempted: raising audio guidance to the authors' recommended 7.0
  *alone* makes it ~17 dB worse, because that recommendation assumes the rescale/STG stack.
- `HARTSY_LTX2_PROBE=1` now also dumps the audio stages (latent pre/post-denorm, VAE log-mel, vocoder
  waveform) via a shared `ProbeTensor` helper.

### Changed
- **`LlamaStyleEncoder` attention glue is device-resident** — the per-layer CPU reshape/RoPE/GQA-repeat/merge
  `float*` loops are replaced with the existing `Permute0213`/`ApplyRopeSingleHeadMajor`/`RepeatKvHeads`
  kernels (this encoder is shared by Qwen-Image, Z-Image, Krea2, Boogu, Flux.2, Ideogram 4, Lumina2, OmniGen2
  and others); tests assert the D2H sync count, not just numerics.
- **Lumina2 sampling schedule corrected — output images change.** The scheduler previously applied Flux-style
  dynamic shifting derived from the image token count (an experiment its own comment marked VALIDATION-PENDING,
  using Flux's base/max-shift constants); the official Alpha-VLLM/Lumina-Image-2.0 `scheduler_config.json` is
  `shift: 6.0` with `use_dynamic_shifting: false`, so the pipeline now uses the checkpoint's static shift. Same
  seed produces a (correctly) different image than prior releases.
- **SD3 patchify/final-layer/masked-mix run on device**; `flash_attn_v2_tf32` rejects partial query tiles
  (OOB read) and zero-fills its shared-memory K/V tail (stale-value poisoning); MaxPool distinguishes an empty
  window from a valid all-−Inf one; MSDA uses true −Inf softmax init and 64-bit index products; the cuDNN SDPA
  plan cache is keyed by attention scale; the step-cache treats a zero-denominator relative distance as
  Infinity (was a false cache HIT); Sage's F32→F16 V-narrowing is opt-in (`HARTSY_SAGE_UNSAFE_F32_V_NARROW`).
- **SeedVR2 DiT is device-resident** — the bring-up host-math forward (window gather/scatter, rope,
  qk-norm, AdaSingle on CPU spans; ~200 stream drains per forward) is replaced with backend-op
  composition: fused `QkvSplitNorm`, `RowGather`/`RowScatterAdd` window packing over cached per-geometry
  index tensors, `WanRopeInterleaved` with per-token identity-padded tables, and modulation vectors
  precombined per timestep (constant 1000) and cached across chunks (`SeedVr2DevicePlan`). No new
  kernels beyond GPU-resident `SeedVr2PixelShuffle`/`SeedVr2PadBottomRight` (the VAE upsampler/downsampler
  host loops, previously a multi-GB D2H+H2D round trip each). **Measured e2e: 960×540-area 14.5 → 2.7
  s/frame (362 → 68 s); 720p-area 25.7 → 8.4 s/frame (4090, BBB 25 f). The 3060 runs the same clip in
  169 s.** Existing tiny-config parity numbers are unchanged to the printed digit; per-chunk phase timing
  is logged at Debug level.

### Fixed
- **LTX-2.5 ignored the prompt entirely — `prompt_adaln` was driven by the raw flow sigma.** Output was sharp
  and temporally coherent but followed the seed, not the text: two unrelated prompts at one seed differed by
  1.28% of pixel range, guidance 1 and guidance 10 behaved identically, and zeroing the *entire* 1024-row
  conditioning changed nothing. `LtxVideo2Transformer` passed the unscaled sigma (0..1) to
  `prompt_adaln_single` / `audio_prompt_adaln_single`, where the reference passes the same ×1000-scaled
  timestep every other modulator gets. Those modules emit the `shift_kv`/`scale_kv` that modulate the **text
  keys and values** into every block's cross-attention, so evaluating a sinusoidal timestep embedding at t≈1
  instead of t≈1000 left the cross-attention with the right magnitude but no ability to discriminate between
  prompts. A code comment stated the wrong convention as fact, which is what kept it alive. Fixed at all six
  call sites (video + audio); per-block prompt sensitivity against ComfyUI 0.32 went from 8–13× too weak to
  **ratio 1.00**. LTX-2.5 now generates prompt-faithful 704×480×25f clips with a soundtrack in ~80 s on a 4090.
  The whole text path was verified against the reference on the real `int8_lean_convrot` checkpoints on the way
  (tokenizer ids byte-exact, Gemma-4 tower cosine 0.9999–1.0000 per layer, connector within 0.2–0.6%, `attn2`
  within 0.07%) and needed no changes; two research items previously flagged as open — the 49-state stack's
  final norm and the left-vs-right padding side — were both settled as **not** defects, the first of which also
  clears the shipping LTX-2.3 Gemma-3 path.
- **Z-Image Base checkpoints were silently corrupted when the filename carried no variant token.** The official
  Base release ships under the bare family name (`z_image_bf16.safetensors`); variant detection fell through to
  Turbo's policy, whose F16 attention narrowing overflows Base's >83k value-projection range into Inf — a
  garbage image with no error. Bare family naming now positively detects Base, and genuinely ambiguous
  filenames default to the numerically safe Base policy (F32 attention, shift 6) with a loud warning — a
  misfiled Turbo merely runs slower, instead of a misfiled Base corrupting. Verified with a real generation
  from the official Comfy-Org single-file on the exact previously-corrupted filename.
- **Z-Image `ReleaseDeviceCache` could leave the captured denoise step graph pointing at freed memory** — the
  graph bakes the caption-pin and RoPE-table device addresses it frees, so a later same-signature forward
  could replay against freed allocations (CUDA 700 context poison). The release path now invalidates the
  graph first, keeps an invalidation failure as the first error, and continues the rest of cleanup.
- **`CfgNormalizedEulerStep`/`ApplyCfgNormalized` produced NaN for `eps=0` with an all-zero guided row**
  (0/0 in the norm ratio; NaN then poisons `z` through `0·NaN`). A zero denominator now resolves the ratio to
  0 — exact, since an all-zero row contributes nothing — identically in the IBackend fallback, the CUDA
  kernel (`dit_f32.ptx` regenerated), and the host helper; new CPU+CUDA regression test.
- **cuDNN auto-fetch 404'd for CUDA < 12** (NVIDIA publishes no cuDNN 9.21 redist there) — now refused
  up-front with manual-install guidance instead of attempting the download.
- **`Qwen3Tokenizer` hardcoded its chat special-token ids** (`<think>`, `<|im_start|>`, pad/EOS), which only
  fit the embedded artifact — a caller-supplied `tokenizer.json` now has them resolved from its added-token
  table, with a logged fallback when absent.
- **Z-Image rejected legitimate solid-color output** — a uniformly black/white frame now only fails the
  generation when the decoded F32 tensor is actually non-finite; a finite solid frame (valid prompt outcome,
  inpaint over a solid source) is accepted with a log line.
- **CUDA BF16/F16 GroupNorm mis-read non-F32 affine weights.** `CastAffineDownIfF32` only converted an
  F32 affine down to the kernel dtype; an F16-checkpoint affine (e.g. the numz SeedVR2 VAE) was passed
  raw to the BF16 kernel — F16 bits reinterpreted as BF16 → garbage scale/shift and flat-gray output.
  Any affine dtype is now converted to the kernel's dtype. Caught by the SeedVR2 BF16-VAE bring-up: the
  isolated parity test passed against the f32 checkpoint while the pipeline (fp16 catalog checkpoint)
  produced uniform gray.

- **Borrowed views passed as an in-place op's OUTPUT silently discard the write on CUDA** — a hazard class, found via
  MiniMax-H3. The backend binds the result to the borrowed `View`/`RowView` and the dispose callback skips the D2H, so
  the store never lands; the CPU path is unaffected, which is why unit tests passed. In H3 this made RoPE, adaLN
  modulation and the gated residual no-ops across all 50 blocks — `h` never left the patch embedding and every frame
  decoded to a regular-grid mosaic. Fixed by forcing the read-back at the three sites; CPU-vs-CUDA parity went
  0.246 → 4.75e-4 and step-0 velocity rms 7.90 → 2.24. The rest of the repo was swept for the same pattern and is
  clean: the only other borrowed views outside `HartsyInference.Core` feed a host `float*` loop (video-VAE tile
  blending) or are read-only GEMM inputs (LLM stacked-weight slicers).
- **MiniMax-H3 now loads SwarmUI/Comfy's flat checkpoint layout, not just the vendor folder tree.** The vendor
  publishes `transformer/` + `video_vae/` + `audio_vae/` + `text_encoder/` folders; Comfy-Org repackages the same
  weights as one file per component under `diffusion_models/`, `vae/` and `text_encoders/`, and that is what SwarmUI's
  native H3 support downloads — so a Swarm-driven load previously failed looking for a `transformer/` subfolder that
  does not exist. New `MiniMaxH3Assets` resolves both layouts, walking up from the DiT to find components and ranking
  variants so an unloadable `int8_convrot` file never beats a loadable sibling. Nothing is downloaded: re-fetching
  under the engine's own model directory would duplicate the ~5.8 GB of VAEs Swarm already has. Falls back to the
  embedded Qwen BPE and to `MiniMaxH3VideoVaeConfig.Detect` since the flat repack ships no tokenizer or `config.json`.
- **MiniMax-H3 `pruned_fp8_scaled` checkpoints load, and are ~10x faster than bf16.** Two defects blocked them.
  (1) `ThrowIfInt8Convrot` rejected on the *presence* of Comfy's quantization companions, but Comfy tags every
  quantized build with the same `.weight_scale`/`.input_scale`/`.comfy_quant` suffixes — only the `.comfy_quant`
  descriptor distinguishes them, and the fp8 build says `{"format": "float8_e4m3fn"}`. The guard now reads the
  descriptor and rejects only genuine int8-convrot (`MiniMaxH3QuantGuardTests`; an absent/unreadable descriptor still
  rejects conservatively). (2) The converter never called the shared `CheckpointConvertUtils.ApplyFp8ScaledDequant`,
  so the scale companions were routed as unknown weights. **Verified on the real 21 GB
  `minimax_h3_fl2va_pruned_fp8_scaled` checkpoint:** 22 frames at 512x288 / 20 steps produces a coherent tracking shot
  with matched 0.9167 s audio at -21.5 dBFS peak, at **8.6 s/step and 22.5 GB VRAM, fully resident on a 24 GB 4090** —
  versus ~90 s/step for the 66 GB bf16 build, which cannot stay resident and re-reads most of itself from NVMe every
  step. This run is also the first exercise of the *pruned* checkpoint's `adaln_t_table` curve path (`curves=True`).

- **MiniMax-H3 is 25.9x faster: 50.2 -> 1.94 s/step** (512x288, 141 frames, RTX 4090; a full 30-step clip went
  1602 s -> 129 s). ComfyUI does the identical work at 1.67 s/step, so this closes a ~30x gap to ~1.16x.
  The cause was host round-trips, not weight residency or GEMM selection. `View`/`RowView` were built as
  `new Tensor((void*)t.DataPointer, ...)`, and `GpuTransferHelper`'s activation cache is keyed by Tensor object
  reference, so a view can never alias its parent's device buffer — and merely CONSTRUCTING one calls
  `DataPointer` -> `EnsureCpuData` -> cache-evict + `cuStreamSynchronize` + device-to-host copy. The worst
  offender was the QKV split, which host-copied a `[seq, 21504]` tensor three times per attention (~473 MB each
  way per block, ~47 GB/step). Restructured so views are never needed: `SliceLastDim` for the QKV and adaln
  splits, q/k allocated 4-D up front so `RmsNorm` runs in place and `ApplyRopeSingle` consumes them directly,
  the residual stream shaped `[seq, 1, hidden]` so `AffineBroadcastLastDim`/`GatedResidualLastDim` modulate the
  whole packed sequence in a single launch driven by a `RowGather`ed table, and `Concat` for segment assembly.
  **No new kernels.** Acceptance metric: D2H syncs per forward **74 -> 0** (`IBackend.GetD2hSyncCount`, whose own
  doc states a fully GPU-resident denoise loop must stay at ~0). Numerics unchanged — parity holds at video
  relL2 4.752E-004 / audio 6.555E-004.
- **MiniMax-H3 text encoder: ~2x less PCIe traffic per prompt.** The nvfp4 tower dequantized every layer into a
  full F32 weight and uploaded it per call (~97 GB per encode). It now narrows to BF16 inside the dequant loop
  and reuses one shared host scratch buffer instead of ~350 short-lived 200-500 MB allocations, and drops a
  redundant per-call `Sync()` (`FreeWeights` already syncs). BF16 rather than F16 because F16 overflows on the
  SwiGLU gated tensor. Qwen3-VL is BF16-trained, so this is closer to the reference than the old F32/TF32 path.

- **MiniMax-H3 geometry was wrong at its own declared defaults.** Three grids were mis-derived: frame counts must
  snap up onto `17k+5`, video latent frames are `(frames-5)/17*5 + 2` rather than `frames/4`, and pixel axes round to
  32 rather than 16 (a multiple of 16 that is not a multiple of 32 gives an odd latent axis, and the 2x2 patchifier
  silently drops its last row/column). At the shipped defaults `1360x768x121f` that meant 1344x768 output and 102
  delivered frames sized against ~5.0 s of audio — roughly 0.8 s of soundtrack generated and then trimmed away. The
  reference grids now live in `MiniMaxH3Geometry` and are pinned by `MiniMaxH3GeometryTests`, including a round-trip
  asserting the latent count re-expands to exactly the requested frames. Defaults corrected to 1344x768x124f.
- **`CudaBackend` now logs the device name at construction.** `CUDA_VISIBLE_DEVICES` defaults to fastest-first
  ordering, so it does not agree with `nvidia-smi` indices — every perf and VRAM figure from an H3 bring-up run was
  initially attributed to the wrong GPU because only the ordinal was logged.

## [2.0.0-alpha.8] — 2026-08-01

### Fixed
- **A short soundtrack silently dropped trailing video frames.** Muxers cut to the shorter stream
  (ffmpeg `-shortest`), and LTX-2.3's audio-latent count rounds down: a real 25-frame @24fps clip
  (1.0417s) came back with 1.010s of audio, so the muxed mp4 contained **24 frames, not 25**.
  `VideoAudioResolver` now fits the track to the clip in both directions — trim if long, silence-pad
  (`AudioBuffer.PadTo`) if short — so frame count is preserved; a shortfall over 0.25s still warns,
  since that indicates the wrong track rather than latent rounding. Verified on a real LTX-2.3
  generation: audio 1.0417s, muxed mp4 keeps all 25 frames, and the generated samples are
  bit-identical to the pre-fix run with the padding appended as pure silence.
  Found by the e2e run after alpha.7 was cut, hence the separate version.

### Note
- `2.0.0-alpha.7` was tagged but never appeared on nuget.org (both the flat-container and registration
  indexes still topped out at alpha.6 more than 30 minutes after publish). Consume alpha.8 instead.

## [2.0.0-alpha.7] — 2026-08-01

Video gets its sound back: generated audio now reaches the caller (closes `TODO(E-IMG-4/5)`), plus the
LTX-2 split-checkpoint decode fix.

### Added
- **`AudioBuffer`** (`Engine.Requests`) — engine-native raw planar-float PCM, the decoded counterpart to
  `AudioClip` (encoded in) and `AudioResult` (encoded out). Mono/stereo conversion + duration trim; the
  shared currency for moving a waveform between components in any modality.
- **`VideoGenerationResult`** — frames plus the soundtrack that belongs with them.
- **`VideoAudioResolver`** — one place that decides which track ships with a generation: what the pipeline
  attached beats `VideoRequest.VideoAudioInput` pass-through, then the track is trimmed to video length.
  `VideoAudioReference` is deliberately not a fallback (it is conditioning; a family that means it to be
  heard attaches it itself).
- `AudioClipCodec` is now public and gained `DecodeNative` (native rate/channels, no resample) and an
  `EncodeWav(AudioBuffer)` overload.
- REST `/v1/native/video/stream` emits an `audio` SSE event (base64 WAV + rate/channels).

### Fixed
- **LTX-2 split-checkpoint output was checkerboard garbage** (the known-broken `hartsy video -m ltx-2`
  path, which SwarmUI also hits). The split VAE file ships bare keys, and the converter's bare-key router
  only recognized `decoder.`/`encoder.`/`latents_` as VAE keys — `per_channel_statistics.{mean-of-means,
  std-of-means}` fell into the Transformer bucket, so latent denormalization silently became an identity
  no-op. With std-of-means as low as 0.074, the decoder received channels up to ~13× too hot; the up-stack
  amplified that to ±943 and the RGB clamp saturated to checkerboard. One added route in
  `LtxVideo2CheckpointConverter.RouteKey` fixes it: decode now lands in [-1,1] and the catalog path produces
  coherent frames (verified 512×320×25f, seed 42; transformer Sha256 pinned). Bundled single-file
  checkpoints were never affected.
- **LTX-2.3's generated soundtrack was dropped**, not muxed — `LtxVideo2RecipePipeline` logged a warning
  and discarded it because the pipeline contract carried frames only. It is now attached and muxed.
- **Wan2.2-S2V's driving speech was not muxed either.** The mux moved to the Engine when the extension was
  thinned to a wrapper, but was never implemented there; `VideoRequest.VideoAudioInput` was documented as a
  mux track with no consumer. S2V now attaches the speech it consumed, at source rate rather than the 16 kHz
  mono conditioning downmix.
- The SwarmUI extension's ffmpeg audio mux (`VideoOutputEncoder.AudioTrack`, `FormatSupportsAudio`) was
  unreachable dead code — never constructed, never passed. Reconnected, with a warning when the chosen
  container (gif/webp) cannot carry a track.

### Changed
- **Breaking:** `IVideoRecipePipeline.Generate` returns `VideoGenerationResult` instead of
  `IReadOnlyList<VideoFrame>`, and `IVideoService.GenerateAsync` returns `Task<VideoGenerationResult>`
  instead of `IAsyncEnumerable<VideoFrame>`. The enumerable never streamed — it awaited the full frame list
  before yielding — so no delivery behaviour is lost. Replaced rather than added alongside: a second
  frame-only overload would silently drop audio, which is the bug being fixed.
- `hartsy video` writes `audio.wav` beside the frame directory when a generation produces sound.

## [2.0.0-alpha.6] — 2026-08-01

SeedVR2 video/image restoration — a new modality, end to end.

### Added
- **SeedVR2 one-step video restoration** (`Modality.Restore`, catalog ids `seedvr2-3b`/`seedvr2-7b`):
  NaDiT windowed MM-DiT + s8c16t4 causal video VAE ported to pure C#, every stage parity-gated against
  the ByteDance reference — window partition **exact** (2,490 slices), preprocessing maxAbs **2.3e-6**,
  VAE relL2 **≤2.9e-6** vs real weights, full-model E2E **SSIM 0.99950 / 56.6 dB PSNR** vs the Python
  pipeline with injected reference noises. Surfaces: `hartsy restore <video|image>` (PNG frames + H.264
  MP4 out), `--restore` chain on `hartsy video`, REPL `/mode restore`, `POST /v1/native/restore[/stream]`,
  and the SwarmUI extension's "Video Restore" param group. 7-clip real-footage matrix verified on the
  4090 (USIA Reagan '87, NASA Apollo 11, JFK '61, Steamboat Willie, Prelinger '62, Big Buck Bunny
  ground-truth, still-image t==1 branch): 25-frame clips at 960×540-area, **~14 s/frame, 17.1 GB peak,
  zero OOM**. Ground-truth profile matches the paper: pixel metrics prefer bicubic (SSIM −0.05) but
  **LPIPS improves 26–28%** (0.735→0.541 extreme; 0.448→0.324 mild) — it repaints, it doesn't
  reconstruct; `--strength` guards oversharpening.
- **`FfmpegProcessDecoder`** (ffmpeg/ffprobe child processes) — first video-INPUT path in the engine;
  `VideoClip`/`RestoreRequest` DTOs; `TorchResize` (torchvision-exact antialiased bicubic, a=−0.5
  float32 weights — two silent-divergence bugs caught by parity, see PARITY_VERIFICATION).
- **Reference quirks ported deliberately** (SEEDVR2_ARCHITECTURE.md §2.5): the tail `vid_out_ada`
  cache-collision (uses the ATTN emb slice — the code as written is dimensionally impossible), last-layer
  `vid_only` semantics incl. the txt self-residual doubling, per-frame VAE GroupNorm stats, asymmetric
  (0,1,0,1) downsampler padding, MAGViT `(x y z c)` pixel-shuffle dropping output frame index 1.

### Known limitations
- fp32 whole-clip VAE activations cap restoration at ~960×540-area on 24 GB (5-frame chunks); 720p+
  needs bf16 activations or tiled VAE — tracked in MODEL_STATUS_VIDEO remaining work.
- DiT window gather/scatter and RoPE run host-side (bring-up shape): ~14 s/frame. Residency/CUDA-graph
  optimization is the follow-up perf pass.
- Catalog DiT + VAE download from the community safetensors mirror `numz/SeedVR2_comfyUI` (verbatim
  original state-dict keys, fp16; Sha256 pinned from a verified download → convert → restore run, and
  the fp16 output is visually equivalent to fp32 — remaining delta is generative high-frequency repaint).
  Only the 1.2 MB frozen pos/neg embeddings ship from `HartsyAI/SeedVR2-safetensors` (upstream has them
  as torch-pickle `.pt` only); until published, place `seedvr2_embeddings.safetensors` under
  `Models/Video/SeedVr2/`.
- **seedvr2-7b is catalog-registered but BLOCKED**: its smoke run revealed the 7B is the **v1 NaDiT**
  (`models/dit`, `qk_rope`/`shared_qkv`) whose state-dict keys coincide with v2 — it loaded and produced
  plausible-but-wrong mud (GT SSIM 0.71 vs 3B's 0.88). `SeedVr2Config.Detect` now throws on the v1
  signature instead of running it; the v1 port is tracked in MODEL_STATUS_VIDEO.

## [2.0.0-alpha.5] — 2026-07-27

Low-VRAM generation, a GPU-memory leak fix, and selectable devices.

### Added
- **Low-VRAM weight streaming across the image fleet** (`HARTSY_LOWVRAM`, three-state: `auto` default /
  `on` / `off`). The sliding-window machinery (`BlockStreamingController`) already existed but only one of
  ~25 image pipelines used it. **Four models that could not run on a 12 GB card now do**, all 1024²,
  quality-gate clean: **HunyuanImage-2.1** (19.7 s), **Ideogram 4** (205 s — a *pair* of 9.3 GB DiTs
  needing 19.7 GB against 9.2 GB available), **Qwen-Image** (231 s, 20B MMDiT), **Krea2** (71 s).
  `off` is a real escape hatch: the same request succeeds under `on` and raises `OutOfVramException`
  under `off`.
- **`VramPlanner`** — one place that decides resident-vs-streamed per generation phase, carries the
  `HARTSY_KEEP_MODELS` residency short-circuit, and logs the **weights-vs-activations split** (streaming
  can only move the weight term, so a phase dominated by activations needs a smaller working set, not a
  sliding window).
- **Selectable CUDA device**: `cuda:1`-style backend selectors, `InferenceEngine(selector, ordinal)`, and
  a real `GPU_ID` in the SwarmUI backend — previously logged and ignored. Verified by memory delta.
  Note the ordinal is CUDA's (fastest-first), which need not match `nvidia-smi`'s PCI order.
- **SD3.5 modular component loading** — CLIP-L / CLIP-G / T5-XXL / VAE each resolve independently when the
  checkpoint does not bundle them, which is the standard SD3.x distribution format. SD3.5-Medium now
  generates end-to-end; it previously threw before any sampling.

### Fixed
- **GPU memory leak on OOM.** `CudaBackend.PreloadWeights` had no exception path, so a mid-load OOM left
  already-uploaded weights registered against a model that would never finish — unreachable, therefore
  unfreeable. The process held ~11.5 GB with nothing running and **starved other processes on the same
  card**, including a separate ComfyUI. Now: typed `OutOfVramException`, per-batch rollback, and reclaim at
  both the generate and construct boundaries. An OOM'd process now holds **152 MiB instead of ~11.5 GB**,
  and a sequential multi-model sweep survives an OOM (3/3 models succeeded after one).
- **Streaming was inert for every GGUF model.** `DType.Q4_K.SizeInBytes` is 0 (a K-quant has no
  per-element size), so `ElementCount * SizeInBytes` totalled block weights to **zero bytes** and the
  "fits resident?" test was always trivially true. Fixed in four block implementations.
- **Lens rendered solid black** (16/16). SageAttention's INT8 path materializes V as F16; Lens does not
  RMS-norm V, and `max|V|` crossed F16's 65504 mid-generation. Verified against ComfyUI's own reference
  implementation on the same checkpoint — an engine bug, not a port bug.
- **Anima was 19-63× slower than ComfyUI**: 792 host round-trips per denoise step (14 per block × 28
  blocks × 2 CFG passes). Now **3**. Warm step 15,279 ms → 519 ms. Its documented "1024² hangs" was never
  a hang.
- **The VRAM planner under-reported free memory by ~4.6 GB**, because `cuMemGetInfo` counts the
  stream-ordered pool's reservations as used. The error is asymmetric — it biases toward streaming, which
  costs 5-8× — so a large card could silently take the slow path for a model that fits.
- `TextService.PrimaryDeviceKey()` hardcoded `"cuda:0"`, so a `cuda:1` engine would have rendered images on
  one GPU while its LLM landed on another.
- Lumina2's on-disk checkpoint was the wrong variant (`cap_embedder.*` naming vs the diffusers
  `time_caption_embed.*` the converter expects). Correct weights now load and generate — though this
  revealed a **separate, previously unreachable conditioning bug**: output is coherent but off-prompt.

### Changed
- `Chroma` checkpoint conversion is now streaming per tensor (removes a GC-timing dependence from the
  peak). **The documented "host RAM OOM" does not reproduce** — it peaks at 9.1 GB anon and completes;
  the reported 25 GB was total RSS including reclaimable file-backed page cache.

## [1.0.0-alpha.48]

Production-readiness push: closes the throughput gap toward python inference stacks (vLLM/TGI-class) and
adds the serving infrastructure a real deployment needs. Full technical detail in
[`docs/Checklists/LLM_DECODE_PERF_GRIND.md`](docs/Checklists/ROADMAP.md)'s dated status
updates; this is the release-notes-level summary.

### Added
- **Fused GEMV kernels for Q4_0 and Q5_K** quantization formats — the last two of the six original
  quant types without a fused decode kernel; both previously fell to the ~10-20x-slower
  dequant-to-F16-then-cuBLAS path.
- **On-device repetition penalty for CUDA-graph decode.** Graph decode was previously greedy-only with a
  raw unpenalized argmax — a request with `RepetitionPenalty > 1.0` and graph decode enabled silently
  ignored the penalty. Fixed with two new device-resident kernels chained into the existing captured graph.
- **`/v1/chat/completions`** (OpenAI-compatible, streaming and non-streaming) on `HartsyInference.Server` —
  the server previously had no LLM chat endpoint at all (image generation only). Includes structured
  request logging (queue depth, prompt/completion tokens, latency, tokens/sec) and real cancellation that
  stops in-flight generation, not just the HTTP connection.
- **Paged KV cache** (`PagedKvPool`/`PagedKvCache`) — replaces the single-sequence `FixedKvCache` (hard
  `batch=1` restriction) with pages allocated on demand from a pool shared across sequences.
- **True continuous batching** (`DynamicBatchScheduler`/`IBatchScheduler`) — requests admit dynamically at
  any time and batch together into shared decode rounds; each sequence evicts the instant it
  finishes/stops/cancels. Replaces the old static-batch `ContinuousBatchScheduler` (fixed request list up
  front, zero production callers, removed). Backend-exclusivity is preserved via an injected gate so LLM
  batching never races with diffusion image generation on the shared GPU backend instance.
- **JSON-mode constrained decoding** (`response_format: {"type":"json_object"}`) — masks every candidate
  token so generation can only produce syntactically valid JSON. The richer `json_schema` mode is not
  implemented and is rejected with a clear 400 rather than silently ignored.
- Server integration test suite (`ChatCompletionsIntegrationTests`, in-process via `WebApplicationFactory`)
  covering chat-completions request validation — previously zero automated coverage on this HTTP surface.

### Changed
- `IBackend.SliceTimeRange` — new primitive (host default + CUDA kernel) extracting a contiguous
  time-range from a KV-shaped tensor; used by the paged KV cache.
- `GenericTransformer.ForwardBatchDecode`'s cache parameter widened from `FixedKvCache[]` to `IKvCache[]`.
- Chat-completions request validation now checks pure request-shape issues (empty messages, unsupported
  `response_format`) before consulting server state (is the model loaded) — fails fast on a malformed
  request regardless of what's currently loaded.

### Fixed
- Two real bugs in the new JSON-grammar state machine, both caught by unit tests before ever touching a
  live model: object keys didn't set the post-string parse transition (would have broken any JSON with a
  key — i.e. almost all real JSON); the state's `Clone()` was missing two fields added after it was first
  written (every candidate-token check clones the state, so this would have corrupted the container stack
  on every single trial in production).
- `ModelManager`'s diffusion-vs-LLM checkpoint routing no longer speculatively attempts the LLM loader on
  an unrecognized GGUF — a prior version of this logic (try-LLM-then-catch-fallback) fully materialized a
  multi-GB diffusion checkpoint's tensors before the fallback path could fire, causing a real OOM.
- Paged KV cache's VRAM footprint is now sized from a configurable byte budget
  (`HartsyInferenceServerOptions.KvPoolBytesBudget`, default 512MB) scaled to each loaded model's actual KV
  dimensions, replacing a fixed page count that comfortably fit a narrow-KV-dim model but eagerly
  pre-allocated several GB for a wider one — caught loading gemma-3 during a broader architecture sweep.

### Deferred (explicitly, not attempted)
- Prefix/prompt caching (share identical-content KV pages across sequences) — real additional scope (page
  reference-counting, prefix hashing, copy-on-write on divergence).
- Speculative decoding — a true stretch item, orthogonal to everything else in this release.
- `json_schema`-constrained decoding (schema-aware, not just syntax-valid JSON).
- Wider quant kernel coverage (Q2_K/Q3_K/IQx formats) — no template to adapt from, genuinely new kernel
  design (lookup-table dequant for IQx specifically).

## [1.0.0-alpha.47] and earlier

Not individually itemized here — see `git log` for the full history prior to this changelog's introduction.
