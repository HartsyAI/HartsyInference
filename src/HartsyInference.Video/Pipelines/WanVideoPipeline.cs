using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Runtime;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Video.Pipelines;

/// <summary>Wan-Video (Wan-AI, Apache-2.0) text-to-video pipeline — Wan2.2 TI2V-5B. Maximum reuse: the DiT denoises directly in VAE-latent space <c>[1,48,T,H,W]</c> (the transformer patchifies/unpatchifies internally), and the VAE is the **already-built <see cref="Wan22VaeDecoder"/>** (z=48, 16×/4×, streaming). UniPC multistep (flow sigmas) + 2-way text CFG; reuses <see cref="LancePipelineCommon"/> + frame streaming.
///
/// <para>Takes pre-computed umT5 features (encode upstream with the shared T5 encoder). <c>T_lat = (num_frames−1)/4 + 1</c>; latent <c>H/16 × W/16</c>. <b>Status: built, first-run validation pending</b> — the flow-match shift (5.0/3.0), scheduler (UniPC vs Euler), and DiT timestep scaling are validation-gated.</para></summary>
public sealed unsafe class WanVideoPipeline : DiffusionPipelineBase
{
    private readonly WanVideoTransformer _transformer;
    private readonly WanVideoTransformer? _transformer2;   // low-noise expert (Wan2.2 A14B MoE)
    private readonly IWanVaeDecoder _vae;
    private readonly IWanVaeEncoder? _encoder;
    private readonly WanVideoConfig _config;

    /// <summary><paramref name="encoder"/> is optional and only needed for RGB-input I2V (<see cref="EncodeFirstFrame"/>);
    /// it loads from the same VAE weight dict as the decoder. <paramref name="transformer2"/> is the low-noise expert
    /// for the Wan2.2 A14B MoE (the constructor's <paramref name="transformer"/> is then the high-noise expert); leave
    /// it null for single-expert variants.</summary>
    public WanVideoPipeline(IBackend backend, WanVideoTransformer transformer, IWanVaeDecoder vae, WanVideoConfig config,
        IWanVaeEncoder? encoder = null, WanVideoTransformer? transformer2 = null)
        : base(backend)
    {
        _transformer = transformer;
        _vae = vae;
        _config = config;
        _encoder = encoder;
        _transformer2 = transformer2;
    }

    /// <summary>Selects the MoE expert for a given (×1000) timestep: high-noise expert while <c>tEmb ≥ boundary·1000</c>,
    /// else the low-noise expert. Single-expert variants always return the primary transformer.</summary>
    private WanVideoTransformer Expert(float tEmb) =>
        (_config.IsMixtureOfExperts && _transformer2 is not null && tEmb < _config.BoundaryRatio * 1000f)
            ? _transformer2 : _transformer;

    // Tracks which MoE expert's weights are currently GPU-resident. Two 14B experts (2×14 GB fp8) don't fit in 24 GB,
    // so we keep only the active one loaded and swap once at the boundary crossing (high→low noise).
    private WanVideoTransformer? _loadedExpert;

    /// <summary>Standard-profile residency (HARTSY_KEEP_MODELS, default on): the single-expert DiT stays GPU-resident across generations so the next gen's preload is a cache-hit no-op; every VAE phase beside it is gated on measured free VRAM (evict when short).</summary>
    private static readonly bool KeepModelsResident = EnvSwitch.IsEnabled("HARTSY_KEEP_MODELS", defaultOn: true);

    // Cross-generation I2V conditioning cache: the [mask, cond-latent] tensor is a deterministic function of
    // (init frame, last frame, geometry), tiny (~1.4 MB host), and its whole-padded-clip VAE encode is the ONE
    // phase whose conv-activation peak (~7.5 GB at 25f 512×320, measured) can never run beside the resident
    // 14B DiT — a same-image repeat skips the encode, the DiT eviction, AND the DiT re-upload.
    private Tensor? _cachedCondition;
    private string? _cachedConditionKey;

    /// <summary>Ensures <paramref name="expert"/> is the only DiT resident: frees the previously-loaded expert (if
    /// different) and preloads this one. No-op for the single-transformer (non-MoE) case handled by the callers.</summary>
    private void SwapToExpert(WanVideoTransformer expert)
    {
        if (ReferenceEquals(_loadedExpert, expert)) return;
        if (_loadedExpert is not null) Backend.FreeWeights(_loadedExpert.EnumerateWeights());
        Backend.PreloadWeights(expert.EnumerateWeights());
        _loadedExpert = expert;
    }

    // Env-gated (HARTSY_WAN_DEBUG=1) per-step numerical diagnostic for the all-zero-output hunt. Reads host data only
    // (scheduler writes latents host-side; velocity is host-coherent per the working Flux/Lance pattern). Identical
    // velocity stats across steps ⇒ the GPU is re-reading a stale (frozen) latent input; NaN/inf or exploding
    // magnitudes ⇒ transformer/scheduler math. See memory wan22-video-first-run-state.
    private static readonly bool WanDebug = Environment.GetEnvironmentVariable("HARTSY_WAN_DEBUG") == "1";
    private static void DumpStats(string tag, Tensor t)
    {
        if (!WanDebug) return;
        ReadOnlySpan<float> s = t.AsReadOnlySpan<float>();
        double mn = double.MaxValue, mx = double.MinValue, sum = 0, sumAbs = 0;
        long n = s.Length, bad = 0;
        for (long i = 0; i < n; i++)
        {
            float v = s[(int)i];
            if (float.IsNaN(v) || float.IsInfinity(v)) { bad++; continue; }
            if (v < mn) mn = v; if (v > mx) mx = v; sum += v; sumAbs += Math.Abs(v);
        }
        long ok = n - bad;
        Logs.Info($"[WANDBG] {tag}: n={n} min={(ok > 0 ? mn : 0):F5} max={(ok > 0 ? mx : 0):F5} " +
            $"mean={(ok > 0 ? sum / ok : 0):F5} meanAbs={(ok > 0 ? sumAbs / ok : 0):F5} nan/inf={bad}");
    }

    /// <summary>Encodes an interleaved-RGB24 conditioning frame to the normalized first-frame latent for the TI2V
    /// I2V path — pass the result as <c>firstFrameLatent</c> to <see cref="GenerateFromEmbeddings"/> /
    /// <see cref="GenerateFramesAsync"/>. The caller owns (disposes) the returned tensor. Requires the pipeline to be
    /// constructed with a <see cref="Wan22VaeEncoder"/>.</summary>
    public Tensor EncodeFirstFrame(ReadOnlySpan<byte> rgb24, int width, int height)
    {
        ThrowIfDisposed();
        if (_encoder is null)
            throw new InvalidOperationException("RGB-input I2V needs a Wan22VaeEncoder — construct the pipeline with one (it loads from the same VAE weights).");
        return _encoder.EncodeRgbFrame(VaeBackend, rgb24, width, height);
    }

    /// <summary>Generates frames from pre-computed umT5 features <c>[L, textDim]</c>. Returns one interleaved-RGB <c>byte[]</c> per frame.
    /// <para><paramref name="firstFrameLatent"/> and/or <paramref name="lastFrameLatent"/> switch to the TI2V
    /// image-to-video path (diffusers <c>expand_timesteps</c>): a <c>[1, 48, 1, H/16, W/16]</c> VAE-encoded <b>and
    /// latent-normalized</b> (<see cref="Wan22VaeLatentNorm.Normalize"/>) frame that is re-imposed into the model
    /// input each step at per-frame timestep 0 while the remaining frames denoise. Both set together is Wan's
    /// first-last-frame (FLF2V) mode; either alone conditions only that end. The Wan2.2 VAE <i>encoder</i> is not
    /// built yet — produce the conditioning latent offline (validation-gated).</para></summary>
    public (byte[][] frames, int width, int height, int seed) GenerateFromEmbeddings(
        Tensor promptEmbeds, Tensor negativeEmbeds, TextToImageRequest request, int numFrames,
        Action<GenerationProgress>? onProgress = null, Tensor? firstFrameLatent = null, Tensor? lastFrameLatent = null)
    {
        Tensor latent = RunDenoise(promptEmbeds, negativeEmbeds, request, numFrames, onProgress, firstFrameLatent, lastFrameLatent, out int seed);
        // LOAD-BEARING for VaeDevice: RunDenoise's latents are already host-current (FlowUniPCMultistepScheduler.Step
        // writes sample.DataPointer host-side every iteration), so no explicit Sync()+DataPointer read is needed
        // before handing off to a VaeBackend on another device — unlike the device-resident-latent pipelines.
        VaeBackend.PreloadWeights(_vae.EnumerateWeights());
        Tensor rgb;
        try { rgb = _vae.Decode(VaeBackend, latent); }
        finally { latent.Dispose(); }
        if (!ReferenceEquals(VaeBackend, Backend)) VaeBackend.Sync();

        int f = (int)rgb.Shape[2];
        byte[][] frames = new byte[f][];
        for (int i = 0; i < f; i++) frames[i] = FrameToBytes(rgb, i);
        rgb.Dispose();
        Logs.Info($"Wan-Video T2V complete ({frames.Length} frames, seed={seed})");
        return (frames, request.Width ?? 832, request.Height ?? 480, seed);
    }

    /// <summary>Streams decoded frames (pull-based → memory bounded; pair with an <c>IVideoEncoder</c>).
    /// <paramref name="firstFrameLatent"/>/<paramref name="lastFrameLatent"/> enable the TI2V image-to-video path —
    /// see <see cref="GenerateFromEmbeddings"/>.</summary>
    public async IAsyncEnumerable<VideoFrame> GenerateFramesAsync(
        Tensor promptEmbeds, Tensor negativeEmbeds, TextToImageRequest request, int numFrames,
        Action<GenerationProgress>? onProgress = null, Tensor? firstFrameLatent = null, Tensor? lastFrameLatent = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        Tensor latent = RunDenoise(promptEmbeds, negativeEmbeds, request, numFrames, onProgress, firstFrameLatent, lastFrameLatent, out _);
        // Preload the VAE decoder weights onto the GPU. Without this every conv/norm in the decode is a weight
        // cache-miss → SyncStream + re-upload, serializing the whole decode (GPU idle ~79%, ~8 min). RunDenoise
        // ended with ReleaseOrKeepTransformer, which guarantees decode headroom (kept the DiT only if it fits).
        // Runs on VaeBackend (defaults to Backend): RunDenoise's latents are host-current (see GenerateFromEmbeddings),
        // so a VaeBackend on another device just uploads from there.
        VaeBackend.PreloadWeights(_vae.EnumerateWeights());
        if (_vae is Wan22VaeDecoder w22)
        {
            try
            {
                int idx = 0;
                foreach (Tensor group in w22.DecodeStreaming(VaeBackend, latent))   // [1,3,groupT,H,W] per latent frame
                {
                    int gT = (int)group.Shape[2], h = (int)group.Shape[3], w = (int)group.Shape[4];
                    for (int gi = 0; gi < gT; gi++)
                        yield return new VideoFrame(idx++, w, h, FrameToBytesGroup(group, gi));
                    group.Dispose();
                    cancellationToken.ThrowIfCancellationRequested();
                    await Task.Yield();
                }
            }
            finally { latent.Dispose(); }
            yield break;
        }
        // Non-streaming VAE (Wan2.1): full decode, then emit each frame.
        Tensor rgb;
        try { rgb = _vae.Decode(VaeBackend, latent); }
        finally { latent.Dispose(); }
        try
        {
            int f = (int)rgb.Shape[2], w = (int)rgb.Shape[4], h = (int)rgb.Shape[3];
            for (int i = 0; i < f; i++)
            {
                yield return new VideoFrame(i, w, h, FrameToBytes(rgb, i));
                cancellationToken.ThrowIfCancellationRequested();
                await Task.Yield();
            }
        }
        finally { rgb.Dispose(); }
    }

    /// <summary>Runs the flow-match denoise loop in (normalized) latent space and returns <c>[1,48,T_lat,H_lat,W_lat]</c>.
    /// With <paramref name="firstFrameLatent"/> and/or <paramref name="lastFrameLatent"/> set, follows the diffusers
    /// <c>expand_timesteps</c> I2V path: the model input gets each set condition imposed on its latent-frame index
    /// (0 for first, <c>T_lat-1</c> for last) with that frame's per-frame timestep pinned to 0 each step, the
    /// evolving latents are stepped freely, and the condition(s) are re-imposed once after the loop.</summary>
    private Tensor RunDenoise(Tensor promptEmbeds, Tensor negativeEmbeds, TextToImageRequest request, int numFrames,
        Action<GenerationProgress>? onProgress, Tensor? firstFrameLatent, Tensor? lastFrameLatent, out int seed)
    {
        ThrowIfDisposed();
        seed = request.Seed ?? SeedGenerator.RandomSeed();
        int width = request.Width ?? 832, height = request.Height ?? 480;
        int sp = _config.VaeSpatialCompression, tp = _config.VaeTemporalCompression;
        if (width % sp != 0 || height % sp != 0)
            throw new ArgumentException($"Width/height must be divisible by {sp} for Wan-Video.");
        if (numFrames < 1 || (numFrames - 1) % tp != 0)
            throw new ArgumentException($"num_frames must satisfy (num_frames-1) % {tp} == 0; got {numFrames}.");

        int tLat = (numFrames - 1) / tp + 1;
        int hLat = height / sp, wLat = width / sp;
        if (firstFrameLatent is not null &&
            (firstFrameLatent.Shape.Rank != 5 || firstFrameLatent.Shape[0] != 1 || firstFrameLatent.Shape[1] != _config.InChannels
             || firstFrameLatent.Shape[2] != 1 || firstFrameLatent.Shape[3] != hLat || firstFrameLatent.Shape[4] != wLat))
            throw new ArgumentException($"firstFrameLatent must be [1,{_config.InChannels},1,{hLat},{wLat}]; got {firstFrameLatent.Shape}.", nameof(firstFrameLatent));
        if (lastFrameLatent is not null &&
            (lastFrameLatent.Shape.Rank != 5 || lastFrameLatent.Shape[0] != 1 || lastFrameLatent.Shape[1] != _config.InChannels
             || lastFrameLatent.Shape[2] != 1 || lastFrameLatent.Shape[3] != hLat || lastFrameLatent.Shape[4] != wLat))
            throw new ArgumentException($"lastFrameLatent must be [1,{_config.InChannels},1,{hLat},{wLat}]; got {lastFrameLatent.Shape}.", nameof(lastFrameLatent));
        if (lastFrameLatent is not null && tLat < 2)
            throw new ArgumentException($"lastFrameLatent needs at least 2 latent frames (numFrames={numFrames} → T_lat={tLat}); a single-frame clip has no distinct end.", nameof(lastFrameLatent));

        int steps = request.Steps ?? _config.NumInferenceSteps;
        float guidance = request.CfgScale ?? _config.GuidanceScale;
        float shift = (request as VideoGenerationRequest)?.FlowShift ?? _config.FlowShift;

        string mode = firstFrameLatent is not null && lastFrameLatent is not null ? "FLF2V"
            : firstFrameLatent is not null ? "I2V"
            : lastFrameLatent is not null ? "EndFrame-only (unverified — Wan's real usage always pairs an init image)"
            : "T2V";
        Logs.Info($"Wan-Video {mode}: {numFrames}f {width}x{height}, {steps} steps, cfg={guidance}, seed={seed} (latent {_config.InChannels}x{tLat}x{hLat}x{wLat}, shift={shift})");
        Logs.Warning("Wan-Video pipeline is first-run-validation pending — numerics unverified vs the reference checkpoint.");

        // CFG-branch parallelism (ROADMAP.md §1): uncond runs concurrently on a second backend instead of after
        // cond on this one. MoE (A14B) is excluded from v1 — SwapToExpert below keeps only one expert resident on
        // Backend, and mirroring that swap onto a second backend under VRAM pressure is unbuilt; single-expert
        // variants (the on-disk TI2V-5B checkpoint) always qualify. Preloading BOTH backends' weight caches here,
        // sequentially before the loop, is load-bearing for correctness, not just perf: once both caches are warm,
        // every in-loop weight read is a per-backend cache-hit dictionary lookup that never touches the shared
        // weight tensor's mutable GPU-binding state, which is what makes concurrent cond/uncond reads of the same
        // weight tensors safe (see CfgBranchRunner's doc comment). Only this loop opts in — the CLIP-I2V and
        // video-to-video loops below stay sequential.
        bool cfgParallelEnabled = CfgParallelBackend is not null && _transformer2 is null;
        LastCfgParallelDecision = null;
        if (CfgParallelBackend is not null && _transformer2 is not null)
        {
            RecordCfgParallelDecision("fell-back(moe-two-expert)");
        }
        if (_transformer2 is null) Backend.PreloadWeights(_transformer.EnumerateWeights());
        if (cfgParallelEnabled)
        {
            // Replicating ~the whole DiT on the second card can genuinely not fit (TI2V-5B fp16 needs ~10 GB
            // per backend); a failed preload must degrade to sequential CFG, not kill the generation.
            try
            {
                CfgParallelBackend!.PreloadWeights(_transformer.EnumerateWeights());
                RecordCfgParallelDecision("active");
            }
            catch (Exception ex)
            {
                Logs.Warning($"Wan CFG-parallel: couldn't preload the DiT onto the second backend (falling back "
                    + $"to sequential CFG this generation): {ex.Message}");
                RecordCfgParallelDecision($"fell-back(preload-failed: {ex.Message})");
                cfgParallelEnabled = false;
            }
        }
        // MoE (A14B): SwapToExpert in the loop keeps only the active expert resident (2×14 GB won't co-reside in 24 GB).
        // T2V/TI2V denoise in VAE-latent space; the latent channel count is the VAE z, not the (possibly larger,
        // I2V-concat) transformer in_channels.
        int latentCh = _config.VaeLatentChannels;
        Tensor latents = SeedGenerator.CreateNoise(new TensorShape([1L, latentCh, tLat, hLat, wLat]), seed);
        // VALIDATION-PENDING: Wan 2.2 ships UniPCMultistepScheduler (solver_order=2, bh2, predict_x0=true,
        // use_flow_sigmas=true, time_shift_type="exponential"); verify the UniPC sigma grid + bh2 update-coefficients
        // against diffusers WanPipeline at the configured step count (e.g. 50).
        FlowUniPCMultistepScheduler scheduler = new(solverOrder: int.TryParse(Environment.GetEnvironmentVariable("WAN_SOLVER_ORDER"), out int _so) && _so > 0 ? _so : 2);
        scheduler.SetTimesteps(steps, shift);
        float[]? frameTs = (firstFrameLatent is null && lastFrameLatent is null) ? null : new float[tLat];

        // Default-off perf knobs (Qwen-Image is the reference wiring — INFERENCE_ACCEL_GRIND §H1.5/H2.3):
        // HARTSY_STEP_CACHE = First-Block cache per CFG stream; HARTSY_CFG_INTERVAL = skip the uncond forward
        // outside the normalized-σ band (NOTE the Qwen finding: early-step skipping changes image identity —
        // late-only bands like "0.15,1" are the quality-safe shape). Unset ⇒ byte-identical baseline.
        GuidanceInterval cfgInterval = GuidanceInterval.FromEnvironment();
        float stepCacheThreshold = StepCacheEnv.ReadThreshold();
        DeviceFeatureCache? condCache = null;
        DeviceFeatureCache? uncondCache = null;
        if (stepCacheThreshold > 0f)
        {
            if (Backend.SupportsDeviceStepCacheGate)
            {
                int stepCacheCap = StepCacheEnv.ReadCap();
                condCache = new DeviceFeatureCache(stepCacheThreshold, stepCacheCap, StepCacheEnv.ReadPoly(), StepCacheEnv.ReadCalibFile());
                uncondCache = new DeviceFeatureCache(stepCacheThreshold, stepCacheCap, StepCacheEnv.ReadPoly(), StepCacheEnv.ReadCalibFile());
                Logs.Info($"Step cache ON: threshold={stepCacheThreshold}, maxConsecutiveReuse={stepCacheCap}");
            }
            else
            {
                Logs.Warning("HARTSY_STEP_CACHE set but the backend lacks a device-side gate " +
                    "(stepcache.ptx not compiled?) — running uncached.");
            }
        }
        if (!cfgInterval.IsAlways)
            Logs.Info($"CFG interval ON: guidance applies at normalized t in [{cfgInterval.Start}, {cfgInterval.End}]");

        // Context parallelism (sequence split, weights replicated on both ranks): cond and uncond stay sequential,
        // but each forward forks into rank 0 (this thread, Backend) + rank 1 (worker, CpBackends[1]) over
        // frame-aligned token ranges, with per-block self-attn K/V exchanged via CpKvExchange. v1 scope: 2 ranks,
        // single-expert (MoE swap can't be mirrored under VRAM pressure), no step cache (residuals are
        // full-sequence), and never composed with CFG-parallel (ValidatePlacement rejects the config — the guard
        // here is belt-and-braces for direct pipeline construction).
        LastCpDecision = null;
        bool cpEnabled = false;
        if (CpBackends is { Count: >= 2 })
        {
            if (!ReferenceEquals(CpBackends[0], Backend)) RecordCpDecision("fell-back(rank0-not-primary)");
            else if (CpBackends.Count != 2) RecordCpDecision($"fell-back({CpBackends.Count}-ranks-unsupported)");
            else if (_transformer2 is not null) RecordCpDecision("fell-back(moe-two-expert)");
            else if (cfgParallelEnabled) RecordCpDecision("fell-back(cfg-parallel-configured)");
            else if (condCache is not null) RecordCpDecision("fell-back(step-cache)");
            else if (tLat / _config.PatchSize.T < 2) RecordCpDecision("fell-back(single-latent-frame)");
            else cpEnabled = true;
        }
        if (cpEnabled)
        {
            // Replicating the whole DiT on rank 1 can genuinely not fit; degrade to single-GPU, not a dead generation.
            try
            {
                CpBackends![1].PreloadWeights(_transformer.EnumerateWeights());
            }
            catch (Exception ex)
            {
                Logs.Warning($"Wan context-parallel: couldn't preload the DiT onto rank 1 (falling back to "
                    + $"single-GPU this generation): {ex.Message}");
                RecordCpDecision($"fell-back(preload-failed: {ex.Message})");
                cpEnabled = false;
            }
        }
        CpSequencePlan? cpPlan = null;
        CpKvExchange? cpExchange = null;
        if (cpEnabled)
        {
            (int cpt, int cph, int cpw) = _config.PatchSize;
            int gt = tLat / cpt, gh = hLat / cph, gw = wLat / cpw;
            // Frame split proportional to post-preload free VRAM (activations scale with the rank's token count).
            (long free0, _) = Backend.GetVramInfo();
            (long free1, _) = CpBackends![1].GetVramInfo();
            cpPlan = CpSequencePlan.Create(gt, gh * gw, [free0, free1]);
            cpExchange = new CpKvExchange(cpPlan);
            // First-touch of the shared rope/context caches must happen before the rank threads fork.
            _transformer.PrewarmSequenceCaches(Backend, gt, gh, gw, promptEmbeds, negativeEmbeds);
            RecordCpDecision($"active(frames {cpPlan.Ranks[0].FrameCount}+{cpPlan.Ranks[1].FrameCount})");
        }

        // One rank's forward: local [Sr, outVec] rows, host-materialized on the owning thread. ANY rank failure
        // aborts the exchange first — the peer may be blocked at a K/V barrier and must throw, not deadlock.
        Tensor CpRankForward(int rank, IBackend rankBackend, Tensor input, Tensor embeds, float[]? fts, float tE)
        {
            try
            {
                CpForwardContext ctx = new() { Rank = rank, Plan = cpPlan!, Exchange = cpExchange! };
                Tensor local = fts is null
                    ? _transformer.Forward(rankBackend, input, embeds, tE, null, ctx)
                    : _transformer.Forward(rankBackend, input, embeds, fts, null, ctx);
                _ = local.DataPointer;
                return local;
            }
            catch
            {
                cpExchange!.Abort();
                throw;
            }
        }

        // Full CP forward: fork rank 1 onto a worker (CfgBranchRunner's dedicated-thread ambient-binding shape),
        // run rank 0 here, then gather rows in global token order and unpatchify — the callers' host-side
        // scheduler/CFG path continues unchanged.
        Tensor CpForkedForward(Tensor input, Tensor embeds, float[]? fts, float tE)
        {
            Tensor rank1Input = CloneLatents(input);   // never share the mutable input tensor across rank threads
            Tensor local0, local1;
            try
            {
                (local0, local1) = CfgBranchRunner.Run(
                    () => CpRankForward(0, Backend, input, embeds, fts, tE),
                    () => CpRankForward(1, CpBackends![1], rank1Input, embeds, fts, tE));
            }
            finally { rank1Input.Dispose(); }
            (int cpt, int cph, int cpw) = _config.PatchSize;
            Tensor full = WanDitOps.ConcatRows(local0, local1, _config.OutChannels * cpt * cph * cpw);
            local0.Dispose();
            local1.Dispose();
            Tensor v = WanDitOps.Unpatchify(full, _config.OutChannels, tLat / cpt, hLat / cph, wLat / cpw, _config.PatchSize);
            full.Dispose();
            return v;
        }

        int cfgSkippedSteps = 0;
        int cacheComputes = 0, cacheReuses = 0, uncondComputes = 0, uncondReuses = 0;
        WanVideoTransformer? cacheExpert = null;   // A14B MoE: residuals are expert-specific — reset on swap

        for (int k = 0; k < steps; k++)
        {
            Stopwatch sw = Stopwatch.StartNew();
            float tEmb = scheduler.Timesteps[k];   // sigma·1000 (DiT timestep scaling, validation-gated)
            WanVideoTransformer expert = Expert(tEmb);
            if (_transformer2 is not null) SwapToExpert(expert);
            if (condCache is not null && !ReferenceEquals(expert, cacheExpert))
            {
                // A residual cached from one expert is meaningless under the other (same shapes — nothing
                // else would catch it). Accumulate stats, then hard-reset both streams at the boundary.
                if (cacheExpert is not null)
                {
                    cacheComputes += condCache.Computes; cacheReuses += condCache.Reuses;
                    uncondComputes += uncondCache!.Computes; uncondReuses += uncondCache.Reuses;
                    Logs.Info("Step cache: expert boundary — caches reset");
                }
                condCache.Reset();
                uncondCache!.Reset();
                cacheExpert = expert;
            }
            bool cfgThisStep = cfgInterval.Applies(tEmb / 1000f);
            if (!cfgThisStep) cfgSkippedSteps++;
            Tensor vCond;
            Tensor? vUncond = null;
            // Clone the shared latent BEFORE forking, on this (the main) thread — cloning inside the uncond
            // thunk would still race the main thread's own read of the same source tensor. Weights are safe to
            // share across the two threads (both backends' caches are warm from the preload above), but the
            // per-step latent mutates every iteration and is the one tensor both branches would otherwise touch.
            bool cfgParallel = cfgThisStep && cfgParallelEnabled;
            if (firstFrameLatent is null && lastFrameLatent is null)
            {
                if (cpEnabled)
                {
                    vCond = CpForkedForward(latents, promptEmbeds, null, tEmb);
                    if (cfgThisStep) vUncond = CpForkedForward(latents, negativeEmbeds, null, tEmb);
                }
                else if (cfgParallel)
                {
                    Tensor uncondLatents = CloneLatents(latents);
                    try
                    {
                        (vCond, vUncond) = CfgBranchRunner.Run(
                            () => expert.Forward(Backend, latents, promptEmbeds, tEmb, condCache),
                            () => expert.Forward(CfgParallelBackend!, uncondLatents, negativeEmbeds, tEmb, uncondCache));
                    }
                    finally { uncondLatents.Dispose(); }
                }
                else
                {
                    vCond = expert.Forward(Backend, latents, promptEmbeds, tEmb, condCache);
                    if (cfgThisStep) vUncond = expert.Forward(Backend, latents, negativeEmbeds, tEmb, uncondCache);
                }
            }
            else
            {
                // Model input: condition on frame 0 and/or frame T_lat-1 (timestep 0 each), evolving noise
                // elsewhere (timestep t). frameTsLocal is a non-null local so the loop body below narrows
                // cleanly — the null-forgiving `frameTs!` inside a `for` body doesn't otherwise persist the
                // non-null flow state to the Forward() calls after it.
                Tensor modelInput = CloneLatents(latents);
                if (firstFrameLatent is not null) WriteFirstFrame(modelInput, firstFrameLatent);
                if (lastFrameLatent is not null) WriteLastFrame(modelInput, lastFrameLatent);
                float[] frameTsLocal = frameTs!;
                for (int f = 0; f < tLat; f++) frameTsLocal[f] = tEmb;
                if (firstFrameLatent is not null) frameTsLocal[0] = 0f;
                if (lastFrameLatent is not null) frameTsLocal[tLat - 1] = 0f;
                if (cpEnabled)
                {
                    vCond = CpForkedForward(modelInput, promptEmbeds, frameTsLocal, tEmb);
                    if (cfgThisStep) vUncond = CpForkedForward(modelInput, negativeEmbeds, frameTsLocal, tEmb);
                }
                else if (cfgParallel)
                {
                    Tensor uncondModelInput = CloneLatents(modelInput);
                    try
                    {
                        (vCond, vUncond) = CfgBranchRunner.Run(
                            () => expert.Forward(Backend, modelInput, promptEmbeds, frameTsLocal, condCache),
                            () => expert.Forward(CfgParallelBackend!, uncondModelInput, negativeEmbeds, frameTsLocal, uncondCache));
                    }
                    finally { uncondModelInput.Dispose(); }
                }
                else
                {
                    vCond = expert.Forward(Backend, modelInput, promptEmbeds, frameTsLocal, condCache);
                    if (cfgThisStep) vUncond = expert.Forward(Backend, modelInput, negativeEmbeds, frameTsLocal, uncondCache);
                }
                modelInput.Dispose();
            }
            // Fold CFG into vCond, then take a UniPC predictor/corrector step in place. Outside the guidance
            // band the step runs cond-only (vCond used raw).
            if (vUncond is not null)
                LancePipelineCommon.CfgCombineRenormInPlace(vCond, vUncond, guidance, _config.CfgRescale);
            DumpStats($"step {k} tEmb={tEmb:F2} velocity(cfg)", vCond);
            scheduler.Step(latents, vCond);
            DumpStats($"step {k} latent(post-step)", latents);
            vCond.Dispose();
            vUncond?.Dispose();
            sw.Stop();
            // Latent is a borrowed reference for preview encoders (latent2rgb decodes the middle frame).
            onProgress?.Invoke(new GenerationProgress(k + 1, steps, sw.Elapsed.TotalMilliseconds)
            {
                Latent = latents,
                LatentArch = LatentArchitecture.Wan,
            });
            // Reclaim GPU-resident activation buffers between steps: the DiT keeps intermediates on-device, and any
            // not read-back or disposed would linger until GC and accumulate to OOM over many steps. The latent is
            // host-side, so nothing cross-step is lost. With CFG-parallel active, the uncond branch accumulates its
            // own activations on CfgParallelBackend — free both, or the second card OOMs a few steps in.
            Backend.FreeActivations();
            if (cfgParallelEnabled) CfgParallelBackend!.FreeActivations();
            if (cpEnabled) CpBackends![1].FreeActivations();
        }
        cpExchange?.Dispose();

        if (firstFrameLatent is not null) WriteFirstFrame(latents, firstFrameLatent);
        if (lastFrameLatent is not null) WriteLastFrame(latents, lastFrameLatent);

        // Perf-knob accounting for benchmark records (totals include any pre-expert-boundary segments).
        if (condCache is not null)
        {
            cacheComputes += condCache.Computes; cacheReuses += condCache.Reuses;
            uncondComputes += uncondCache!.Computes; uncondReuses += uncondCache.Reuses;
            Logs.Info($"Step cache: cond {cacheComputes} computes / {cacheReuses} reuses; " +
                $"uncond {uncondComputes} computes / {uncondReuses} reuses");
            condCache.Dispose();
            uncondCache.Dispose();
        }
        if (cfgSkippedSteps > 0)
            Logs.Info($"CFG interval: uncond forward skipped on {cfgSkippedSteps}/{steps} steps");

        Backend.Sync();
        ReleaseOrKeepTransformer(numFrames, width, height);
        return latents;
    }

    /// <summary>Wan concat-conditioned I2V: denoises a z-channel latent while the model input each step is the
    /// concat of <c>[noisy(z), mask(tp), cond-latent(z)]</c> (e.g. 16+4+16 = 36). Serves both Wan2.1 I2V-14B (pass
    /// <paramref name="imageEmbeds"/> = the CLIP-ViT-H penultimate hidden state <c>[seqImg, imageDim]</c> for the
    /// per-block image cross-attention) and Wan2.2 I2V-A14B (no CLIP — pass <paramref name="imageEmbeds"/> = null;
    /// conditioning is purely the concatenated cond-latent + mask). <paramref name="condRgb24"/> is the interleaved-RGB
    /// first frame. Honors the MoE expert switch when a second expert was supplied.</summary>
    public (byte[][] frames, int width, int height, int seed) GenerateImageToVideoConcat(
        Tensor promptEmbeds, Tensor negativeEmbeds, Tensor? imageEmbeds, ReadOnlySpan<byte> condRgb24,
        TextToImageRequest request, int numFrames, Action<GenerationProgress>? onProgress = null,
        byte[]? lastRgb24 = null)
    {
        ThrowIfDisposed();
        if (_encoder is null)
            throw new InvalidOperationException("Wan I2V needs a Wan VAE encoder for the conditioning latent.");
        int latentCh = _config.VaeLatentChannels;
        int sp = _config.VaeSpatialCompression, tp = _config.VaeTemporalCompression;
        if (_config.InChannels != 2 * latentCh + tp)
            throw new InvalidOperationException(
                $"Concat I2V expects InChannels == 2·z + tp ({2 * latentCh + tp}); got {_config.InChannels}. "
                + "Use a Wan I2V config preset (e.g. I2V_14B_480p / I2V_A14B), or the TI2V firstFrameLatent path.");
        if (_config.HasImageConditioning && imageEmbeds is null)
            throw new ArgumentException("This variant has CLIP image conditioning — imageEmbeds is required.", nameof(imageEmbeds));
        int width = request.Width ?? 832, height = request.Height ?? 480;
        if (width % sp != 0 || height % sp != 0)
            throw new ArgumentException($"Width/height must be divisible by {sp} for Wan-Video.");
        if (numFrames < 1 || (numFrames - 1) % tp != 0)
            throw new ArgumentException($"num_frames must satisfy (num_frames-1) % {tp} == 0; got {numFrames}.");

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int tLat = (numFrames - 1) / tp + 1;
        int hLat = height / sp, wLat = width / sp;
        int steps = request.Steps ?? _config.NumInferenceSteps;
        float guidance = request.CfgScale ?? _config.GuidanceScale;
        float shift = (request as VideoGenerationRequest)?.FlowShift ?? _config.FlowShift;

        Logs.Info($"Wan-Video I2V ({(imageEmbeds is null ? "concat" : "CLIP")}): {numFrames}f {width}x{height}, " +
            $"{steps} steps, cfg={guidance}, seed={seed} (latent {latentCh}x{tLat}x{hLat}x{wLat}, shift={shift})");
        Logs.Warning("Wan-Video I2V pipeline is first-run-validation pending — numerics unverified vs the reference checkpoint.");

        // Build the fixed [1, 20, T, H, W] conditioning: [mask(4), cond-latent(16)]. Reference construction
        // (diffusers WanImageToVideoPipeline / Comfy WAN21_I2V): VAE-encode the WHOLE padded pixel clip — the init
        // frame followed by mid-gray (0 in [-1,1]) frames (+ the last frame for FLF2V) — through the causal VAE, so
        // EVERY latent frame of the conditioning is the VAE's encoding of that padding. Encoding only the first
        // frame and zero-filling latent frames 1+ (the pre-2026-07-08 behavior) feeds a large off-distribution
        // constant on 16 of the model's 36 input channels and deterministically destroys every non-anchored frame.
        // The whole-clip encode's conv activations scale with numFrames·H·W — with a KEEP_MODELS-resident DiT from a
        // prior gen still on-device this OOMs (seen at 33f × native res: 5.25 GB requested, 82 MB free). Evict the
        // transformer first when headroom is short; the denoise loop re-uploads it right after.
        Stopwatch phase = Stopwatch.StartNew();
        string condKey = $"{width}x{height}x{numFrames}:" +
            Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(condRgb24)) + ":" +
            (lastRgb24 is null ? "-" : Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(lastRgb24)));
        Tensor condition;
        if (_cachedCondition is not null && condKey == _cachedConditionKey)
        {
            condition = _cachedCondition;
            Logs.Info("[wan-phase] I2V conditioning cache HIT — cond VAE encode skipped.");
        }
        else
        {
            EnsureVaeEncodeHeadroom(numFrames, width, height);
            // Runs on VaeBackend (defaults to Backend): BuildI2VCondition below host-materializes the result, so
            // a VaeBackend on another device never contends with the DiT's VRAM on Backend.
            VaeBackend.PreloadWeights(_encoder.EnumerateWeights());
            Tensor condClip = BuildCondClip(condRgb24, lastRgb24, width, height, numFrames);
            Tensor condLatent = _encoder.Encode(VaeBackend, condClip);   // [1, z, tLat, hLat, wLat], normalized
            condClip.Dispose();
            VaeBackend.Sync();
            VaeBackend.FreeWeights(_encoder.EnumerateWeights());
            WanVideoDebugDump.Dump("i2v_cond_latent", condLatent);
            condition = BuildI2VCondition(condLatent, lastRgb24 is not null, latentCh, tp, tLat, hLat, wLat);   // [1,tp+z,tLat,hLat,wLat]
            condLatent.Dispose();
            WanVideoDebugDump.Dump("i2v_condition", condition);
            _cachedCondition?.Dispose();
            _cachedCondition = condition;
            _cachedConditionKey = condKey;
            Logs.Info($"[wan-phase] I2V cond encode MISS (evict+VAE encode+free): {phase.ElapsedMilliseconds} ms");
        }

        phase.Restart();
        if (_transformer2 is null) Backend.PreloadWeights(_transformer.EnumerateWeights());
        Logs.Info($"[wan-phase] DiT preload: {phase.ElapsedMilliseconds} ms");
        // MoE (A14B): SwapToExpert in the loop keeps only the active expert resident (2×14 GB won't co-reside in 24 GB).
        Tensor latents = SeedGenerator.CreateNoise(new TensorShape([1L, latentCh, tLat, hLat, wLat]), seed);
        // VALIDATION-PENDING: Wan 2.2 UniPC scheduler (solver_order=2, bh2, predict_x0, flow sigmas, exponential
        // shift) — verify the I2V concat path's UniPC trajectory vs diffusers WanImageToVideoPipeline.
        FlowUniPCMultistepScheduler scheduler = new(solverOrder: int.TryParse(Environment.GetEnvironmentVariable("WAN_SOLVER_ORDER"), out int _so) && _so > 0 ? _so : 2);
        scheduler.SetTimesteps(steps, shift);

        for (int k = 0; k < steps; k++)
        {
            Stopwatch sw = Stopwatch.StartNew();
            float tEmb = scheduler.Timesteps[k];
            WanVideoTransformer expert = Expert(tEmb);
            if (_transformer2 is not null) SwapToExpert(expert);
            Tensor modelInput = ConcatChannels(latents, condition);   // [1, 2z+tp, tLat, hLat, wLat]
            Tensor vCond, vUncond;
            WanVideoDebugDump.SetTag("cond");
            if (imageEmbeds is not null)
            {
                vCond = expert.Forward(Backend, modelInput, promptEmbeds, [tEmb], imageEmbeds);
                WanVideoDebugDump.SetTag("uncond");
                vUncond = expert.Forward(Backend, modelInput, negativeEmbeds, [tEmb], imageEmbeds);
            }
            else
            {
                vCond = expert.Forward(Backend, modelInput, promptEmbeds, [tEmb]);
                WanVideoDebugDump.SetTag("uncond");
                vUncond = expert.Forward(Backend, modelInput, negativeEmbeds, [tEmb]);
            }
            WanVideoDebugDump.SetTag(null);
            modelInput.Dispose();
            LancePipelineCommon.CfgCombineRenormInPlace(vCond, vUncond, guidance, _config.CfgRescale);
            scheduler.Step(latents, vCond);
            vCond.Dispose(); vUncond.Dispose();
            sw.Stop();
            Logs.Verbose($"[wan-phase] I2V step {k + 1}/{steps}: {sw.Elapsed.TotalMilliseconds:F0} ms");
            onProgress?.Invoke(new GenerationProgress(k + 1, steps, sw.Elapsed.TotalMilliseconds)
            {
                Latent = latents,
                LatentArch = LatentArchitecture.Wan,
            });
            // Reclaim GPU-resident activation buffers between steps (matches the T2V loop): the DiT keeps
            // intermediates on-device, and any not read-back or disposed linger until GC and accumulate across
            // steps. The latent/condition are host-side, and the rope/context caches are host-materialized, so
            // nothing cross-step is lost.
            Backend.FreeActivations();
        }

        Backend.Sync();
        ReleaseOrKeepTransformer(numFrames, width, height);
        // condition is NOT disposed — it is the cross-generation conditioning cache (host tensor, ~1.4 MB),
        // freed on the next cache miss or in DisposeCore.

        phase.Restart();
        // LOAD-BEARING for VaeDevice: the denoise loop above steps `latents` through the same host-side
        // scheduler write pattern as GenerateFromEmbeddings, so the final latents are already host-current.
        VaeBackend.PreloadWeights(_vae.EnumerateWeights());
        Tensor rgb;
        try { rgb = _vae.Decode(VaeBackend, latents); }
        finally { latents.Dispose(); }
        if (!ReferenceEquals(VaeBackend, Backend)) VaeBackend.Sync();
        Logs.Info($"[wan-phase] I2V VAE decode: {phase.ElapsedMilliseconds} ms");
        int f = (int)rgb.Shape[2];
        byte[][] frames = new byte[f][];
        for (int i = 0; i < f; i++) frames[i] = FrameToBytes(rgb, i);
        rgb.Dispose();
        Logs.Info($"Wan-Video I2V complete ({frames.Length} frames, seed={seed})");
        return (frames, width, height, seed);
    }

    /// <summary>Video-to-video: VAE-encodes <paramref name="rgbClip"/> <c>[1, 3, T, H, W]</c>, noises it to the
    /// flow-match start determined by <paramref name="strength"/> (1 = full re-generation, &lt;1 = stay closer to the
    /// input), and runs the standard T2V denoise from that step (img2img-for-video). Honors the MoE expert switch.</summary>
    public (byte[][] frames, int width, int height, int seed) GenerateVideoToVideo(
        Tensor promptEmbeds, Tensor negativeEmbeds, Tensor rgbClip, float strength, TextToImageRequest request,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        if (_encoder is null)
            throw new InvalidOperationException("Video2Video needs a Wan VAE encoder.");
        if (rgbClip.Shape.Rank != 5 || rgbClip.Shape[1] != 3)
            throw new ArgumentException($"rgbClip must be [1,3,T,H,W]; got {rgbClip.Shape}.", nameof(rgbClip));

        int sp = _config.VaeSpatialCompression, tp = _config.VaeTemporalCompression;
        int pixT = (int)rgbClip.Shape[2], pixH = (int)rgbClip.Shape[3], pixW = (int)rgbClip.Shape[4];
        if (pixH % sp != 0 || pixW % sp != 0)
            throw new ArgumentException($"clip H/W must be divisible by {sp}.");
        if ((pixT - 1) % tp != 0)
            throw new ArgumentException($"clip frame count must satisfy (T-1) % {tp} == 0; got {pixT}.");
        strength = Math.Clamp(strength, 0f, 1f);

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int tLat = (pixT - 1) / tp + 1, hLat = pixH / sp, wLat = pixW / sp, latentCh = _config.VaeLatentChannels;
        int steps = request.Steps ?? _config.NumInferenceSteps;
        float guidance = request.CfgScale ?? _config.GuidanceScale;
        float shift = (request as VideoGenerationRequest)?.FlowShift ?? _config.FlowShift;
        int startStep = Math.Clamp((int)Math.Round((1f - strength) * steps), 0, steps - 1);

        Logs.Info($"Wan-Video V2V: {pixT}f {pixW}x{pixH}, strength={strength}, start step {startStep}/{steps}, " +
            $"cfg={guidance}, seed={seed} (latent {latentCh}x{tLat}x{hLat}x{wLat}, shift={shift})");
        Logs.Warning("Wan-Video V2V pipeline is first-run-validation pending — numerics unverified vs the reference checkpoint.");

        EnsureVaeEncodeHeadroom(pixT, pixW, pixH);
        VaeBackend.PreloadWeights(_encoder.EnumerateWeights());
        Tensor real = _encoder.Encode(VaeBackend, rgbClip);               // [1, z, tLat, hLat, wLat]
        VaeBackend.Sync();
        VaeBackend.FreeWeights(_encoder.EnumerateWeights());

        // Use the UniPC sigma grid (shift-warped flow sigmas) for the noising start so the V2V init matches the
        // scheduler the denoise loop runs on.
        // VALIDATION-PENDING: Wan V2V still Euler-steps here. UniPC is a stateful predictor/corrector whose multistep
        // history starts at step 0; a mid-trajectory start (startStep > 0) needs the diffusers
        // begin_index / init-timestep handling to seed the history correctly. Match WanVideoPipeline.RunDenoise once
        // the partial-trajectory UniPC start is designed + verified vs diffusers WanVideoToVideoPipeline.
        FlowUniPCMultistepScheduler scheduler = new(solverOrder: int.TryParse(Environment.GetEnvironmentVariable("WAN_SOLVER_ORDER"), out int _so) && _so > 0 ? _so : 2);
        scheduler.SetTimesteps(steps, shift);
        float sigma0 = scheduler.Sigmas[startStep];
        Tensor noise = SeedGenerator.CreateNoise(new TensorShape([1L, latentCh, tLat, hLat, wLat]), seed);
        Tensor latents = new Tensor(real.Shape, DType.F32);
        long n = real.Shape.ElementCount;
        float* rp = (float*)real.DataPointer, np = (float*)noise.DataPointer, lp = (float*)latents.DataPointer;
        for (long i = 0; i < n; i++) lp[i] = (1f - sigma0) * rp[i] + sigma0 * np[i];   // flow-match noising
        real.Dispose(); noise.Dispose();

        if (_transformer2 is null) Backend.PreloadWeights(_transformer.EnumerateWeights());
        // MoE (A14B): SwapToExpert in the loop keeps only the active expert resident (2×14 GB won't co-reside in 24 GB).
        for (int k = startStep; k < steps; k++)
        {
            Stopwatch sw = Stopwatch.StartNew();
            float t = scheduler.Sigmas[k], dt = t - scheduler.Sigmas[k + 1], tEmb = scheduler.Timesteps[k];
            WanVideoTransformer expert = Expert(tEmb);
            if (_transformer2 is not null) SwapToExpert(expert);
            Tensor vCond = expert.Forward(Backend, latents, promptEmbeds, tEmb);
            Tensor vUncond = expert.Forward(Backend, latents, negativeEmbeds, tEmb);
            LancePipelineCommon.EulerCfgStep(latents, vCond, vUncond, guidance, dt);
            vCond.Dispose(); vUncond.Dispose();
            sw.Stop();
            onProgress?.Invoke(new GenerationProgress(k + 1, steps, sw.Elapsed.TotalMilliseconds)
            {
                Latent = latents,
                LatentArch = LatentArchitecture.Wan,
            });
        }
        Backend.Sync();
        if (_transformer2 is null) Backend.FreeWeights(_transformer.EnumerateWeights());
        else if (_loadedExpert is not null) { Backend.FreeWeights(_loadedExpert.EnumerateWeights()); _loadedExpert = null; }

        // LOAD-BEARING for VaeDevice: the loop above steps `latents` via EulerCfgStep, a host-side unsafe-pointer
        // loop, so the final latents are already host-current — same guarantee as the other Wan denoise loops.
        VaeBackend.PreloadWeights(_vae.EnumerateWeights());
        Tensor rgb;
        try { rgb = _vae.Decode(VaeBackend, latents); }
        finally { latents.Dispose(); }
        if (!ReferenceEquals(VaeBackend, Backend)) VaeBackend.Sync();
        int f = (int)rgb.Shape[2];
        byte[][] frames = new byte[f][];
        for (int i = 0; i < f; i++) frames[i] = FrameToBytes(rgb, i);
        rgb.Dispose();
        Logs.Info($"Wan-Video V2V complete ({frames.Length} frames, seed={seed})");
        return (frames, pixW, pixH, seed);
    }

    /// <summary>Builds the <c>[1, tp+z, T, H, W]</c> I2V conditioning <c>[mask(tp), cond-latent(z)]</c>: the VAE-encoded
    /// first frame occupies latent frame 0 (and <paramref name="frameLast"/>, when given, the last latent frame for
    /// first-last-frame I2V), the rest are zero; the mask channels are 1 on the known frame(s) and 0 elsewhere — a
    /// structural stand-in for diffusers' temporal mask interleave (validation-gated).</summary>
    /// <summary>Padded conditioning pixel clip <c>[1, 3, numFrames, H, W]</c> in [-1, 1]: frame 0 = the init image,
    /// middle frames = mid-gray (0), last frame = <paramref name="lastRgb24"/> when present (FLF2V).</summary>
    private static Tensor BuildCondClip(ReadOnlySpan<byte> condRgb24, byte[]? lastRgb24, int width, int height, int numFrames)
    {
        Tensor clip = new Tensor(new TensorShape([1L, 3, numFrames, height, width]), DType.F32);
        float* cp = (float*)clip.DataPointer;
        long frame = (long)height * width;
        long perChannel = (long)numFrames * frame;
        new Span<float>(cp, (int)(3 * perChannel)).Clear();
        for (long pix = 0; pix < frame; pix++)
            for (int c = 0; c < 3; c++)
                cp[c * perChannel + pix] = condRgb24[(int)(pix * 3 + c)] / 127.5f - 1f;
        if (lastRgb24 is not null)
        {
            long lastOff = (long)(numFrames - 1) * frame;
            for (long pix = 0; pix < frame; pix++)
                for (int c = 0; c < 3; c++)
                    cp[c * perChannel + lastOff + pix] = lastRgb24[(int)(pix * 3 + c)] / 127.5f - 1f;
        }
        return clip;
    }

    private static Tensor BuildI2VCondition(Tensor condLatent, bool hasLastFrame, int latentCh, int temporalFactor,
        int tLat, int hLat, int wLat)
    {
        int maskCh = temporalFactor;
        int condCh = maskCh + latentCh;
        Tensor o = new Tensor(new TensorShape([1L, condCh, tLat, hLat, wLat]), DType.F32);
        float* op = (float*)o.DataPointer;
        long frame = (long)hLat * wLat;
        long perChannel = (long)tLat * frame;
        new Span<float>(op, (int)((long)condCh * perChannel)).Clear();
        int lastFrame = tLat - 1;
        // Mask channels: 1 at latent frame 0 (and the last latent frame for FLF2V) — matches the diffusers
        // repeat/view/transpose construction for the plain first-frame case.
        for (int m = 0; m < maskCh; m++)
        {
            for (long p = 0; p < frame; p++) op[(long)m * perChannel + p] = 1f;
            if (hasLastFrame)
                for (long p = 0; p < frame; p++) op[(long)m * perChannel + (long)lastFrame * frame + p] = 1f;
        }
        // Cond-latent channels: the FULL causal-VAE latent of the padded clip, all tLat frames.
        float* fp = (float*)condLatent.DataPointer;   // [1, latentCh, tLat, hLat, wLat]
        Buffer.MemoryCopy(fp, op + (long)maskCh * perChannel, (long)latentCh * perChannel * 4, (long)latentCh * perChannel * 4);
        return o;
    }

    /// <summary>Channel-concatenates two <c>[1, C, T, H, W]</c> tensors → <c>[1, Ca+Cb, T, H, W]</c>.</summary>
    private static Tensor ConcatChannels(Tensor a, Tensor b)
    {
        int ca = (int)a.Shape[1], cb = (int)b.Shape[1];
        int t = (int)a.Shape[2], h = (int)a.Shape[3], w = (int)a.Shape[4];
        long perChannel = (long)t * h * w;
        Tensor o = new Tensor(new TensorShape([1L, ca + cb, t, h, w]), DType.F32);
        float* op = (float*)o.DataPointer;
        Buffer.MemoryCopy((float*)a.DataPointer, op, (long)ca * perChannel * 4, (long)ca * perChannel * 4);
        Buffer.MemoryCopy((float*)b.DataPointer, op + (long)ca * perChannel, (long)cb * perChannel * 4, (long)cb * perChannel * 4);
        return o;
    }

    private static Tensor CloneLatents(Tensor latents)
    {
        Tensor o = new Tensor(latents.Shape, DType.F32);
        long bytes = latents.Shape.ElementCount * 4;
        Buffer.MemoryCopy((float*)latents.DataPointer, (float*)o.DataPointer, bytes, bytes);
        return o;
    }

    /// <summary>Overwrites latent frame 0 of <paramref name="latents"/> <c>[1,C,T,H,W]</c> with <paramref name="condition"/> <c>[1,C,1,H,W]</c>.</summary>
    private static void WriteFirstFrame(Tensor latents, Tensor condition)
    {
        int c = (int)latents.Shape[1], t = (int)latents.Shape[2];
        long frame = latents.Shape[3] * latents.Shape[4];
        float* lp = (float*)latents.DataPointer;
        float* cp = (float*)condition.DataPointer;
        for (int ci = 0; ci < c; ci++)
            Buffer.MemoryCopy(cp + ci * frame, lp + (long)ci * t * frame, frame * 4, frame * 4);
    }

    /// <summary>Overwrites latent frame <c>T-1</c> (the last frame) of <paramref name="latents"/> <c>[1,C,T,H,W]</c>
    /// with <paramref name="condition"/> <c>[1,C,1,H,W]</c> — the end-frame symmetric counterpart of
    /// <see cref="WriteFirstFrame"/>, same per-channel memcpy pattern with the frame offset shifted to <c>T-1</c>.</summary>
    private static void WriteLastFrame(Tensor latents, Tensor condition)
    {
        int c = (int)latents.Shape[1], t = (int)latents.Shape[2];
        long frame = latents.Shape[3] * latents.Shape[4];
        float* lp = (float*)latents.DataPointer;
        float* cp = (float*)condition.DataPointer;
        for (int ci = 0; ci < c; ci++)
            Buffer.MemoryCopy(cp + ci * frame, lp + ((long)ci * t + (t - 1)) * frame, frame * 4, frame * 4);
    }

    /// <summary>Evicts the (possibly KEEP_MODELS-resident) DiT before a whole-clip VAE encode when measured free VRAM is short of the conv-activation ceiling. Trims the pool first so slack from the previous generation doesn't make the reading pessimistic — that slack, not the DiT, is what the old unconditional path was really evicting for on warm gens. A VAE encoder on its OWN device (<see cref="DiffusionPipelineBase.VaeBackend"/>) never contends with the DiT's VRAM on <see cref="DiffusionPipelineBase.Backend"/>, so the whole fit-check/evict dance is skipped when split.</summary>
    private void EnsureVaeEncodeHeadroom(int numFrames, int width, int height)
    {
        if (!ReferenceEquals(VaeBackend, Backend))
        {
            Logs.Info("[wan-phase] cond encode: VaeBackend is a separate device — DiT eviction skipped (no contention).");
            return;
        }
        // ~160 F32 conv-activation copies/frame: measured 2026-07-11 — a 25f 512×320 encode consumed ≥7.5 GB
        // (~153 copies/frame) and OOM'd beside a resident DiT under the old 24-copies estimate.
        long encodeNeedBytes = (long)numFrames * height * width * 3L * 4 * 160;
        Backend.TrimMemoryPool();
        long free = Backend.FreeMemoryBytes();
        if (free < encodeNeedBytes + (2L << 30))
        {
            Logs.Info($"[wan-phase] cond encode: freeing transformer weights " +
                $"(need ~{encodeNeedBytes >> 20} MB + 2048 MB margin, free {free >> 20} MB)");
            Backend.FreeWeights(_transformer.EnumerateWeights());
            if (_transformer2 is not null) Backend.FreeWeights(_transformer2.EnumerateWeights());
            _loadedExpert = null;
            Backend.TrimMemoryPool();
        }
        else
        {
            Logs.Info($"[wan-phase] cond encode fits beside the resident DiT " +
                $"(need ~{encodeNeedBytes >> 20} MB + 2048 MB margin, free {free >> 20} MB) — evict skipped.");
        }
    }

    /// <summary>Post-denoise DiT residency (the HARTSY_KEEP_MODELS idiom): keeps the single-expert transformer device-resident across generations — the next gen's PreloadWeights becomes a cache-hit no-op — unless measured free VRAM can't cover the VAE decode (grid-scaled estimate; an OOM is worse than one re-upload). MoE experts always free: two 14B experts never co-reside. A VAE on its OWN device (<see cref="DiffusionPipelineBase.VaeBackend"/>) never contends with the DiT's VRAM on <see cref="DiffusionPipelineBase.Backend"/>, so the decode-headroom check is skipped when split — the DiT just stays resident.</summary>
    private void ReleaseOrKeepTransformer(int numFrames, int width, int height)
    {
        if (_transformer2 is not null)
        {
            if (_loadedExpert is not null) { Backend.FreeWeights(_loadedExpert.EnumerateWeights()); _loadedExpert = null; }
            return;
        }
        if (!KeepModelsResident)
        {
            Backend.FreeWeights(_transformer.EnumerateWeights());
            return;
        }
        if (!ReferenceEquals(VaeBackend, Backend))
        {
            Logs.Info("[wan-phase] DiT kept resident across generations (KEEP_MODELS; VaeBackend is a separate device — no decode contention).");
            return;
        }
        Backend.TrimMemoryPool();
        long decodeNeed = Math.Max(3L << 30, (long)numFrames * height * width * 160);
        long free = Backend.FreeMemoryBytes();
        if (free < decodeNeed)
        {
            Logs.Info($"[wan-phase] evicting resident DiT for VAE decode " +
                $"(free {free >> 20} MB < ~{decodeNeed >> 20} MB estimated decode peak).");
            Backend.FreeWeights(_transformer.EnumerateWeights());
            Backend.TrimMemoryPool();
        }
        else
        {
            Logs.Info($"[wan-phase] DiT kept resident across generations " +
                $"(KEEP_MODELS; free {free >> 20} MB ≥ ~{decodeNeed >> 20} MB decode estimate).");
        }
    }

    /// <summary>Frees the cross-generation resident DiT device weights on model switch — the backend weight cache would otherwise keep the device copies alive past the pipeline (the Chroma→Krea2 stranded-VRAM lesson) — plus the host conditioning cache.</summary>
    protected override void DisposeCore()
    {
        try
        {
            Backend.FreeWeights(_transformer.EnumerateWeights());
            if (_transformer2 is not null) Backend.FreeWeights(_transformer2.EnumerateWeights());
            _loadedExpert = null;
            Backend.TrimMemoryPool();
        }
        catch (Exception ex)
        {
            Logs.Error($"Wan pipeline dispose: failed to free resident DiT weights: {ex}");
        }
        _cachedCondition?.Dispose();
        _cachedCondition = null;
        _cachedConditionKey = null;
    }

    private static byte[] FrameToBytes(Tensor rgb, int frameIndex) => VideoRgbFrames.ExtractFrame(rgb, frameIndex);

    private static byte[] FrameToBytesGroup(Tensor group, int frameIndex) => VideoRgbFrames.ExtractFrame(group, frameIndex);
}
