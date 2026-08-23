using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Video.Pipelines;

/// <summary>LTX-Video (Lightricks, Apache-2.0) text-to-video pipeline. Reuses the foundation: the <see cref="LtxVideoTransformer"/> DiT, the <see cref="LtxVideoVaeDecoder"/> (base 0.9 config), flow-match Euler + 2-way CFG (<see cref="LancePipelineCommon"/>), and frame streaming (<see cref="VideoFrame"/> / <c>IVideoEncoder</c>).
///
/// <para>Takes pre-computed T5-XXL features (the transformer cross-attends to T5 hidden states; encode upstream with the shared T5 encoder). Latent grid is <c>(T_lat, H/32, W/32)</c> with <c>T_lat = (num_frames−1)/8 + 1</c>; tokens are the 128-channel latent flattened in <c>(f,h,w)</c> order (transformer patch 1). 2-way text CFG (guidance 3), 50 steps, rectified-flow Euler. <b>Status: built, first-run validation pending</b> — the flow-match timestep shift (dynamic μ), the timestep scaling into the DiT, and RoPE interpolation are validation-gated.</para></summary>
public sealed unsafe class LtxVideoPipeline : DiffusionPipelineBase
{
    private readonly LtxVideoTransformer _transformer;
    private readonly LtxVideoVaeDecoder _vae;
    private readonly LtxVideoConfig _config;
    private readonly int _latentChannels;

    public LtxVideoPipeline(IBackend backend, LtxVideoTransformer transformer, LtxVideoVaeDecoder vae, LtxVideoConfig config)
        : base(backend)
    {
        _transformer = transformer;
        _vae = vae;
        _config = config;
        _latentChannels = config.InChannels;
    }

    /// <summary>Generates frames from pre-computed T5 features. <paramref name="promptEmbeds"/>/<paramref name="negativeEmbeds"/> are <c>[L, captionChannels]</c>; returns one interleaved-RGB <c>byte[]</c> per frame.
    /// <para><paramref name="firstFrameLatent"/> (Tier 3.4, I2V) is an optional VAE-encoded single-frame latent
    /// <c>[1, inChannels, 1, H/32, W/32]</c> (produced by <see cref="Models.Vae.LtxVideoVaeEncoder"/> upstream —
    /// matches Wan's <c>firstFrameLatent</c> convention: the caller encodes, the pipeline conditions).</para></summary>
    public (byte[][] frames, int width, int height, int seed) GenerateFromEmbeddings(
        Tensor promptEmbeds, Tensor negativeEmbeds, TextToImageRequest request, int numFrames, int frameRate = 25,
        Action<GenerationProgress>? onProgress = null, Tensor? firstFrameLatent = null)
    {
        Tensor latent = RunDenoise(promptEmbeds, negativeEmbeds, request, numFrames, frameRate, onProgress, out int seed, firstFrameLatent);
        Tensor rgb;
        try { rgb = DecodeLatent(latent, seed); }
        finally { latent.Dispose(); }
        if (!ReferenceEquals(VaeBackend, Backend)) VaeBackend.Sync();

        int f = (int)rgb.Shape[2];
        byte[][] frames = new byte[f][];
        for (int i = 0; i < f; i++) frames[i] = FrameToBytes(rgb, i);
        rgb.Dispose();
        Logs.Info($"LTX-Video {(firstFrameLatent is null ? "T2V" : "I2V")} complete ({frames.Length} frames, seed={seed})");
        return (frames, request.Width ?? 768, request.Height ?? 512, seed);
    }

    /// <summary>Streams decoded frames (pull-based → memory bounded; pair with an <c>IVideoEncoder</c>).</summary>
    public async IAsyncEnumerable<VideoFrame> GenerateFramesAsync(
        Tensor promptEmbeds, Tensor negativeEmbeds, TextToImageRequest request, int numFrames, int frameRate = 25,
        Action<GenerationProgress>? onProgress = null, Tensor? firstFrameLatent = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        Tensor latent = RunDenoise(promptEmbeds, negativeEmbeds, request, numFrames, frameRate, onProgress, out int seed, firstFrameLatent);
        Tensor rgb;
        try { rgb = DecodeLatent(latent, seed); }
        finally { latent.Dispose(); }
        if (!ReferenceEquals(VaeBackend, Backend)) VaeBackend.Sync();

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

    /// <summary>Decodes the VAE-ready latent to RGB. For the base 0.9.0 VAE (<c>VaeTimestepConditioned=false</c>)
    /// this is a plain decode. For the timestep-conditioned VAE (0.9.1+ / 13B) it first blends Gaussian noise into the
    /// latent at <c>DecodeNoiseScale</c> and decodes at <c>DecodeTimestep</c>.</summary>
    private Tensor DecodeLatent(Tensor latent, int seed)
    {
        // LOAD-BEARING for VaeDevice: `latent` reaches here via UnpackVideoLatents (a host loop off the stepped token
        // buffer), so it's already host-current — a VaeBackend on another device just uploads from there.
        VaeBackend.PreloadWeights(_vae.EnumerateWeights());
        if (!_config.VaeTimestepConditioned)
            return _vae.Decode(VaeBackend, latent);

        // VALIDATION-PENDING: verify vs diffusers LTXPipeline 0.9.7 — diffusers blends pre-decode:
        //   noise = randn_like(latents)
        //   latents = (1 - decode_noise_scale) * latents + decode_noise_scale * noise
        // then vae.decode(latents, timestep=decode_timestep). The 1000× timestep_scale_multiplier is applied
        // *inside* the decoder, so we pass the raw decode_timestep (0.05) here.
        float s = _config.DecodeNoiseScale;
        // Distinct, reproducible noise stream from the sampling latent noise (seed offset).
        Tensor noise = SeedGenerator.CreateNoise(latent.Shape, unchecked(seed ^ 0x5EED_DEC0));
        float* lp = (float*)latent.DataPointer;
        float* np = (float*)noise.DataPointer;
        long n = latent.Shape.ElementCount;
        for (long i = 0; i < n; i++) lp[i] = (1f - s) * lp[i] + s * np[i];
        noise.Dispose();

        Logs.Info($"LTX-Video timestep-conditioned VAE decode (t={_config.DecodeTimestep:F3}, noise_scale={s:F3})");
        return _vae.Decode(VaeBackend, latent, _config.DecodeTimestep);
    }

    /// <summary>Runs the flow-match denoise loop and returns the VAE-ready latent <c>[1,128,T_lat,H_lat,W_lat]</c>.</summary>
    private Tensor RunDenoise(Tensor promptEmbeds, Tensor negativeEmbeds, TextToImageRequest request, int numFrames, int frameRate,
        Action<GenerationProgress>? onProgress, out int seed, Tensor? firstFrameLatent = null)
    {
        // Sampler selection is NOT wired on this family (2026-08-20 audit): its denoise step is a host-side
        // Euler in a different algebraic form than IBackend.CfgEulerStep, its schedule is a raw float[] rather
        // than a FlowMatchEulerDiscreteScheduler, and its I2V frame-0 re-pin is a post-step fixup a multi-evaluation sampler would skip. Converting it would change
        // default-Euler output, so it was deliberately skipped. Refuse a named sampler rather than accepting the
        // request and silently sampling with something else.
        if (!string.IsNullOrWhiteSpace(request.Scheduler))
        {
            throw new NotSupportedException(
                $"Sampler/schedule '{request.Scheduler}' is not available on LTX-Video. This family has not been "
                + "converted to the sampler seam yet, so it samples with its own Euler step only. Leave the "
                + "sampler unset.");
        }


        ThrowIfDisposed();
        seed = request.Seed ?? SeedGenerator.RandomSeed();
        int width = request.Width ?? 768, height = request.Height ?? 512;
        int sp = _config.VaeSpatialCompression, tp = _config.VaeTemporalCompression;
        if (width % sp != 0 || height % sp != 0)
            throw new ArgumentException($"Width/height must be divisible by {sp} for LTX-Video.");
        if (numFrames < 1 || (numFrames - 1) % tp != 0)
            throw new ArgumentException($"num_frames must satisfy (num_frames-1) % {tp} == 0; got {numFrames}.");

        int tLat = (numFrames - 1) / tp + 1;
        int hLat = height / sp, wLat = width / sp;
        int s = tLat * hLat * wLat;
        int steps = request.Steps ?? _config.NumInferenceSteps;
        float guidance = request.CfgScale ?? _config.GuidanceScale;

        if (firstFrameLatent is not null &&
            (firstFrameLatent.Shape.Rank != 5 || firstFrameLatent.Shape[0] != 1 || firstFrameLatent.Shape[1] != _latentChannels
             || firstFrameLatent.Shape[2] != 1 || firstFrameLatent.Shape[3] != hLat || firstFrameLatent.Shape[4] != wLat))
            throw new ArgumentException($"firstFrameLatent must be [1,{_latentChannels},1,{hLat},{wLat}]; got {firstFrameLatent.Shape}.", nameof(firstFrameLatent));

        // RoPE interpolation scale = (vae_temporal/frame_rate, vae_spatial, vae_spatial).
        (double T, double H, double W) interp = ((double)tp / frameRate, sp, sp);
        // Dynamic flow-match shift: mu = seq·m + b (base_seq 256 → 0.5, max_seq 4096 → 1.16), shift = exp(mu).
        double m = (1.16 - 0.5) / (4096 - 256), bShift = 0.5 - m * 256;
        float shift = (float)Math.Exp(s * m + bShift);

        Logs.Info($"LTX-Video {(firstFrameLatent is null ? "T2V" : "I2V")}: {numFrames}f {width}x{height}, {steps} steps, cfg={guidance}, seed={seed} (grid {tLat}x{hLat}x{wLat}, {s} tokens, shift={shift:F3})");
        Logs.Warning("LTX-Video pipeline is first-run-validation pending — numerics unverified vs the reference checkpoint.");

        Backend.PreloadWeights(_transformer.EnumerateWeights());
        Tensor latents = SeedGenerator.CreateNoise(new TensorShape(s, _latentChannels), seed);

        // I2V conditioning (Tier 3.4): diffusers' prepare_latents does
        //   latents = init_latents * conditioning_mask + noise * (1 - conditioning_mask)
        // with conditioning_mask = 1.0 at frame 0 only — since the mask is exactly 0/1 (no partial blend), this
        // reduces to a direct overwrite of the frame-0 tokens with the encoded image latent, no noise mixed in.
        Tensor? conditioningMask = null;
        Tensor? packedFirstFrame = null;   // kept alive to re-pin frame-0 tokens every step (see the loop below)
        int frame0Tokens = hLat * wLat;
        long frame0Bytes = (long)frame0Tokens * _latentChannels * 4;
        if (firstFrameLatent is not null)
        {
            packedFirstFrame = LtxVideo2Pipeline.PackVideoLatents(firstFrameLatent, _latentChannels);   // [hLat*wLat, C]
            Buffer.MemoryCopy((float*)packedFirstFrame.DataPointer, (float*)latents.DataPointer, frame0Bytes, frame0Bytes);

            conditioningMask = new Tensor(new TensorShape(s), DType.F32);
            float* mp = (float*)conditioningMask.DataPointer;
            for (int i = 0; i < s; i++) mp[i] = i < frame0Tokens ? 1f : 0f;
        }

        // Default-off perf knob (INFERENCE_ACCEL_GRIND §H1.5; Wan is the video reference wiring +
        // measurement — NOTE its verdict: on 50-step video the plain gate trades per-seed reproducibility
        // for speed; treat as a "fast non-reproducible sampling" opt-in). One cache per CFG stream. An
        // armed cache forces the eager path — variable per-step topology can't replay the CFG-pair graph,
        // so the latent must NOT be routed into the graph's fixed buffer either.
        float stepCacheThreshold = StepCacheEnv.ReadThreshold();
        if (conditioningMask is not null && stepCacheThreshold > 0f)
        {
            Logs.Warning("LTX I2V conditioning is not compatible with step-cache in this build — running this generation without step-cache.");
            stepCacheThreshold = 0f;
        }
        DeviceFeatureCache? condCache = null;
        DeviceFeatureCache? uncondCache = null;
        if (stepCacheThreshold > 0f)
        {
            if (Backend.SupportsDeviceStepCacheGate)
            {
                int stepCacheCap = StepCacheEnv.ReadCap();
                condCache = new DeviceFeatureCache(stepCacheThreshold, stepCacheCap, StepCacheEnv.ReadPoly(), StepCacheEnv.ReadCalibFile());
                uncondCache = new DeviceFeatureCache(stepCacheThreshold, stepCacheCap, StepCacheEnv.ReadPoly(), StepCacheEnv.ReadCalibFile());
                Logs.Info($"Step cache ON: threshold={stepCacheThreshold}, maxConsecutiveReuse={stepCacheCap} (step graph bypassed)");
            }
            else
            {
                Logs.Warning("HARTSY_STEP_CACHE set but the backend lacks a device-side gate " +
                    "(stepcache.ptx not compiled?) — running uncached.");
            }
        }

        // With HARTSY_DIT_GRAPH the transformer captures the CFG-pair step and replays it; the working latent must
        // live in its fixed buffer (denoised in place by device CfgEulerStep). Off (default), this returns `latents`.
        Tensor stepLatent = condCache is null ? _transformer.PrepareGraphLatent(Backend, latents) : latents;
        float[] tsteps = LancePipelineCommon.BuildShiftedTimesteps(steps, shift);

        string? diagFile = Environment.GetEnvironmentVariable("LTX_DIAG_FILE");
        bool dbg = diagFile is not null || Environment.GetEnvironmentVariable("LTX_DIAG") == "1";
        void Diag(string m) { Logs.Info(m); if (diagFile is not null) File.AppendAllText(diagFile, m + "\n"); }
        if (dbg) Diag($"[LTX-DIAG] promptEmbeds {Stat(promptEmbeds)} | negEmbeds {Stat(negativeEmbeds)} | initLatent {Stat(latents)}");

        for (int k = 0; k < steps; k++)
        {
            Stopwatch sw = Stopwatch.StartNew();
            float t = tsteps[k], dt = t - tsteps[k + 1];
            float tEmb = t * 1000f;   // DiT timestep scaling (validation-gated)
            (Tensor vCond, Tensor? vUncond, bool callerOwns) = _transformer.ForwardPaired(
                Backend, stepLatent, promptEmbeds, negativeEmbeds, tEmb, (tLat, hLat, wLat), interp, null,
                condCache, uncondCache, conditioningMask);
            if (dbg && (k == 0 || k == steps - 1))
                Diag($"[LTX-DIAG] step {k}: t={t:F4} dt={dt:F4} tEmb={tEmb:F1} | vCond {Stat(vCond)} | vUncond {Stat(vUncond!)} | diffL2={DiffL2(vCond, vUncond!):F4} | latent {Stat(stepLatent)}");
            if (callerOwns)
            {
                // Eager path — host CFG Euler, byte-identical to the validated pipeline.
                LancePipelineCommon.EulerCfgStep(stepLatent, vCond, vUncond!, guidance, dt);
                vCond.Dispose();
                vUncond!.Dispose();
                // I2V conditioning: diffusers re-pins frame 0 to the ORIGINAL image latent after every scheduler
                // step (pipeline_ltx_image2video.py — latents = cat([latents[:,:,:1], pred_latents], dim=2), i.e.
                // frame 0 never actually receives the Euler update at all, not just "starts there and drifts
                // slowly because its timestep is near 0"). Mirror that exactly, not just the initial overwrite.
                if (packedFirstFrame is not null)
                    Buffer.MemoryCopy((float*)packedFirstFrame.DataPointer, (float*)stepLatent.DataPointer, frame0Bytes, frame0Bytes);
            }
            else if (packedFirstFrame is not null)
            {
                // ForwardPaired forces graphMode off whenever conditioningMask is set (this branch should be
                // unreachable for I2V today), but that's an invariant held two call frames away — assert it
                // locally instead of silently skipping the per-step re-pin above, which would reproduce the
                // exact flat-red bug this file's own comment history already caught once.
                throw new InvalidOperationException("LTX I2V conditioning reached the step-graph path, which has no per-step frame-0 re-pin wired — this should not be reachable (ForwardPaired forces eager when conditioningMask is set).");
            }
            else
            {
                // Graph path — in-place device Euler on the fixed latent; velocities are transformer-owned fixed buffers.
                // CfgEulerStep does z += v·delta (v = guidance·pos+(1−guidance)·neg), so pass −dt to match the host
                // EulerCfgStep's z −= v·dt with LTX's positive dt = tsteps[k] − tsteps[k+1].
                Backend.CfgEulerStep(stepLatent, vCond, vUncond!, guidance, -dt);
            }
            sw.Stop();
            if (onProgress is not null)
            {
                // The working latent is token-packed [S, C]; hand preview encoders an NCHW
                // middle-frame slice instead (small copy: h·w·128 floats). Owned here, disposed
                // after the callback returns — encoders must not retain it.
                Tensor previewFrame = LtxVideo2Pipeline.ExtractMiddleFrame(stepLatent, tLat, hLat, wLat, _latentChannels);
                onProgress.Invoke(new GenerationProgress(k + 1, steps, sw.Elapsed.TotalMilliseconds)
                {
                    Latent = previewFrame,
                    LatentArch = LatentArchitecture.Ltx,
                });
                previewFrame.Dispose();
            }
        }

        if (condCache is not null)
        {
            Logs.Info($"Step cache: cond {condCache.Computes} computes / {condCache.Reuses} reuses; " +
                $"uncond {uncondCache!.Computes} computes / {uncondCache.Reuses} reuses");
            condCache.Dispose();
            uncondCache.Dispose();
        }

        Backend.Sync();
        Backend.FreeWeights(_transformer.EnumerateWeights());

        // Unpack from the working latent (the transformer-owned fixed buffer on the graph path); dispose only the
        // original noise tensor — the fixed buffer persists for the next generation (freed in the transformer).
        Tensor vaeLatent = LtxVideo2Pipeline.UnpackVideoLatents(stepLatent, tLat, hLat, wLat, _latentChannels);   // [1,C,T_lat,H_lat,W_lat]
        latents.Dispose();
        packedFirstFrame?.Dispose();
        conditioningMask?.Dispose();
        return vaeLatent;
    }

    private static byte[] FrameToBytes(Tensor rgb, int frameIndex) => VideoRgbFrames.ExtractFrame(rgb, frameIndex);

    /// <summary>Diagnostic: mean/std/min/max of a tensor (forces a host read — debug only).</summary>
    private static string Stat(Tensor t)
    {
        long n = t.Shape.ElementCount;
        float* p = (float*)t.DataPointer;
        double sum = 0, sum2 = 0; float mn = float.MaxValue, mx = float.MinValue;
        for (long i = 0; i < n; i++) { float v = p[i]; sum += v; sum2 += (double)v * v; if (v < mn) mn = v; if (v > mx) mx = v; }
        double mean = sum / n, var = sum2 / n - mean * mean;
        return $"[{t.Shape}] mean={mean:F4} std={Math.Sqrt(Math.Max(0, var)):F4} min={mn:F3} max={mx:F3}";
    }

    private static double DiffL2(Tensor a, Tensor b)
    {
        long n = a.Shape.ElementCount;
        float* pa = (float*)a.DataPointer; float* pb = (float*)b.DataPointer;
        double s = 0;
        for (long i = 0; i < n; i++) { double d = pa[i] - pb[i]; s += d * d; }
        return Math.Sqrt(s);
    }
}
