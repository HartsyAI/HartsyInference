using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>OmniGen 2 text-to-image and reference-image edit pipeline. Caller supplies pre-computed Qwen2.5-VL
/// caption embeddings — OmniGen 2's text encoder is wired through the existing
/// <see cref="Models.TextEncoders.LlamaStyleEncoder"/> at the Qwen2.5-VL preset. Note the upstream
/// <c>pipeline_omnigen2.py</c> conditions on TEXT ONLY even for editing (the mllm never sees the image — vision
/// tokens are exclusive to the separate chat pipeline/model); reference images enter solely as VAE latents in the
/// transformer's token stream, so <see cref="EditFromEmbeddings"/> takes the same text-only embeddings as t2i.</summary>
public sealed class OmniGen2Pipeline : DiffusionPipelineBase
{
    private readonly OmniGen2Transformer _transformer;
    private readonly VaeDecoder _vaeDecoder;
    private readonly VaeEncoder? _vaeEncoder;
    private readonly OmniGen2Config _config;

    // Edit-reference latent cache (single slot, keyed on ref pixel signatures): repeat edit generations
    // (same references, new seed/prompt) skip the VAE encoder pass entirely. Latents are host-materialized
    // at store time so FreeActivations between generations can't revert them.
    private Tensor[] _cachedRefLatents = [];
    private long _cachedRefSig;

    /// <summary>Creates an OmniGen 2 t2i pipeline. Caller owns each component.</summary>
    public OmniGen2Pipeline(IBackend backend, OmniGen2Transformer transformer, VaeDecoder vaeDecoder, OmniGen2Config config)
        : this(backend, transformer, vaeDecoder, vaeEncoder: null, config)
    {
    }

    /// <summary>Creates an OmniGen 2 pipeline with the VAE encoder loaded — required for the reference-image
    /// edit path (<see cref="EditFromEmbeddings"/>). Caller owns each component.</summary>
    public OmniGen2Pipeline(IBackend backend, OmniGen2Transformer transformer, VaeDecoder vaeDecoder,
        VaeEncoder? vaeEncoder, OmniGen2Config config)
        : base(backend)
    {
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _vaeEncoder = vaeEncoder;
        _config = config;
    }

    /// <summary>Generates an image from pre-computed Qwen2.5-VL caption embeddings <c>[1, T, 2048]</c>.
    /// CFG dual-pass when guidance is active with a negative-prompt embedding.
    /// <para>OmniGen 2's defining feature is DUAL guidance: <paramref name="textGuidanceScale"/> (default 4.0,
    /// diffusers <c>text_guidance_scale</c>) and <paramref name="imageGuidanceScale"/> (default 1.0,
    /// diffusers <c>image_guidance_scale</c>). With an input image the reference runs three forwards — see
    /// <see cref="EditFromEmbeddings"/> for the exact formula. This method is text-to-image (no input image),
    /// so it reduces to standard CFG with <paramref name="textGuidanceScale"/>:
    /// <c>pred = uncond + text_guidance_scale*(cond - uncond)</c>.</para>
    /// <para><paramref name="cfgRange"/> = <c>(start, end)</c> as fractions of the schedule: CFG is only applied
    /// when <c>start &lt;= i/num_steps &lt;= end</c>; outside that window the bare conditional prediction is used.</para>
    /// <para><paramref name="cfgScale"/> is accepted for back-compat with the generic
    /// <see cref="TextToImageRequest"/> flow, but <paramref name="textGuidanceScale"/> is the effective guidance:
    /// when the caller leaves <paramref name="textGuidanceScale"/> at its default and passes a non-default
    /// <paramref name="cfgScale"/>, the <paramref name="cfgScale"/> value is used as the text guidance.</para></summary>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromEmbeddings(
        Tensor captionEmbeddings,
        TextToImageRequest request,
        float cfgScale = 1.0f,
        Tensor? negativeCaptionEmbeddings = null,
        float textGuidanceScale = 4.0f,
        float imageGuidanceScale = 1.0f,
        (float start, float end)? cfgRange = null,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        // Wrap-pad every conv backend for this call so the output tiles seamlessly; restores on dispose.
        using IDisposable seamlessScope = BeginSeamlessTiling(request.SeamlessTiling);

        // VALIDATION-PENDING: verify vs diffusers OmniGen2Pipeline.
        // Resolve the effective text guidance. textGuidanceScale (default 4.0) is the OmniGen 2 default and
        // takes priority. If the caller left it at the default but supplied a non-default generic cfgScale,
        // honour the generic value so the legacy request.CfgScale path keeps working.
        float effectiveTextGuidance =
            Math.Abs(textGuidanceScale - 4.0f) < 1e-6f
                ? cfgScale
                : textGuidanceScale;

        // cfg_range gates which steps apply CFG. Default = full schedule (0..1).
        (float rangeStart, float rangeEnd) = cfgRange ?? (0f, 1f);

        bool guidanceActive = CfgHelper.IsGuidanceActive(effectiveTextGuidance);

        if (guidanceActive && negativeCaptionEmbeddings is null)
            throw new ArgumentException(
                "negativeCaptionEmbeddings is required when text guidance > 1.0.",
                nameof(negativeCaptionEmbeddings));

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        (int steps, _, int width, int height) = GenerationDefaults.OmniGen2.Resolve(request);
        int latentH = height / 8;
        int latentW = width / 8;

        Logs.Info($"OmniGen 2 t2i: {width}x{height}, {steps} steps, textGuidance={effectiveTextGuidance}, " +
            $"imageGuidance={imageGuidanceScale}, cfgRange=[{rangeStart},{rangeEnd}], seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        TensorShape latentShape = new(1, _config.InChannels, latentH, latentW);
        int hPacked = latentH / _config.PatchSize;
        int wPacked = latentW / _config.PatchSize;

        // Drain-free denoise loop (the Boogu/Z-Image pattern): patchify the seeded noise to token space ONCE and
        // keep the latent GPU-resident in [1, imgSeqLen, p²·C] across the whole loop. ForwardPacked skips the
        // per-forward host patchify/unpatchify D2H drains (also the cpu-glue-async-race crash source) and returns
        // the negated packed velocity; CfgEulerStep does the CFG combine + flow-match Euler step in one in-place
        // device op (x += (g·cond + (1-g)·uncond)·dt, dt = σ_next − σ). One device unpatchify feeds the VAE.
        Tensor noiseNchw = TakeOrCreateNoise(request, latentShape, seed);
        Tensor latent = DiTUtils.PatchifyNCHW(noiseNchw, _config.PatchSize);
        noiseNchw.Dispose();

        FlowMatchEulerDiscreteScheduler scheduler = new(3.0f);
        scheduler.SetTimesteps(steps);
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;
        int textSeqLen = (int)captionEmbeddings.Shape[1];
        int negSeqLen = negativeCaptionEmbeddings is null ? 0 : (int)negativeCaptionEmbeddings.Shape[1];

        Backend.PreloadWeights(_transformer.EnumerateWeights());

        for (int i = 0; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            // scheduler.Timesteps are sigma·1000; the transformer wants the raw flow-match sigma in [0,1].
            float t = timesteps[i] / 1000.0f;
            float dt = scheduler.Dt(i);

            // cfg_range gating: CFG only when start <= i/steps <= end, otherwise the bare conditional (upstream).
            float schedFraction = steps > 1 ? (float)i / steps : 0f;
            bool inCfgRange = schedFraction >= rangeStart && schedFraction <= rangeEnd;

            Tensor cond = _transformer.ForwardPacked(Backend, latent, t, captionEmbeddings, textSeqLen, hPacked, wPacked);
            if (guidanceActive && inCfgRange)
            {
                Tensor uncond = _transformer.ForwardPacked(Backend, latent, t, negativeCaptionEmbeddings!, negSeqLen, hPacked, wPacked);
                Backend.CfgEulerStep(latent, cond, uncond, effectiveTextGuidance, dt);
                uncond.Dispose();
                // Triple-pass image guidance lives in EditFromEmbeddings; t2i uses only the text-guidance term.
                _ = imageGuidanceScale;
            }
            else
            {
                Backend.CfgEulerStep(latent, cond, cond, 1.0f, dt);   // bare conditional (pos=neg → v=cond)
            }
            cond.Dispose();

            stepSw.Stop();
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds));
        }

        Backend.Sync();
        Backend.FreeWeights(_transformer.EnumerateWeights());

        // One device unpatchify back to NCHW for the VAE (the latent never left the GPU during the loop).
        Tensor latentNchw = new(latentShape, DType.F32);
        Backend.UnpatchifyTokens(latentNchw, latent, _config.InChannels, hPacked, wPacked, _config.PatchSize,
            innerChannelFastest: true);
        latent.Dispose();

        Tensor decoded = _vaeDecoder.Decode(Backend, latentNchw);
        latentNchw.Dispose();
        byte[] rgb = ImagePostProcessor.TensorToRgbBytes(decoded);
        decoded.Dispose();

        sw.Stop();
        Logs.Info($"OmniGen 2 t2i complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        return (rgb, width, height, seed);
    }

    /// <summary>Reference-image edit (text + image → image), mirroring diffusers/upstream
    /// <c>OmniGen2Pipeline.processing</c>. Reference images are RGB tensors <c>[1, 3, Hr, Wr]</c> in
    /// <c>[-1, 1]</c> with dims divisible by 16 (the caller applies the reference's
    /// <c>max_pixels</c>/<c>max_input_image_side_length</c> resize); they are VAE-encoded once (cached across
    /// generations by content signature) and fed into the transformer's token stream at every forward.
    /// <para>Dual guidance per the reference: when both scales are &gt; 1, three forwards per step —
    /// <c>cond</c> (positive text + refs), <c>refPred</c> (negative text + refs), <c>uncond</c> (negative text,
    /// NO refs) — combined as <c>pred = uncond + ig·(refPred − uncond) + tg·(cond − refPred)</c>. When only
    /// <paramref name="textGuidanceScale"/> &gt; 1, standard CFG <c>uncond + tg·(cond − uncond)</c> with the
    /// uncond pass dropping the refs (upstream passes <c>ref_image_hidden_states=None</c> there too). When
    /// neither exceeds 1 the bare conditional is used — upstream applies <paramref name="imageGuidanceScale"/>
    /// only in combination with active text guidance. <paramref name="cfgRange"/> gates both scales to 1 outside
    /// <c>start &lt;= i/steps &lt;= end</c>.</para>
    /// <para>The schedule uses the reference's <c>dynamic_time_shift</c>: <c>m = sqrt(latent_h·latent_w)/40</c>,
    /// which reduces algebraically to the standard flow-match sigma shift <c>σ' = mσ/(1+(m−1)σ)</c> — exact
    /// parity with <c>scheduling_flow_match_euler_discrete.py</c> (<c>t' = t/(m − m·t + t)</c> under
    /// <c>σ = 1 − t</c>).</para>
    /// <para>VRAM staging: the reference VAE encode runs BEFORE the transformer preload (encoder weights are
    /// staged in and freed around the encode), so the encode never executes beside the resident DiT.</para></summary>
    public (byte[] rgbData, int width, int height, int seed) EditFromEmbeddings(
        Tensor captionEmbeddings,
        Tensor? negativeCaptionEmbeddings,
        IReadOnlyList<Tensor> referenceImages,
        TextToImageRequest request,
        float textGuidanceScale = 5.0f,
        float imageGuidanceScale = 2.0f,
        (float start, float end)? cfgRange = null,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        if (_vaeEncoder is null)
            throw new InvalidOperationException("Edit requires a VAE encoder; construct the pipeline with vaeEncoder != null.");
        if (referenceImages.Count == 0)
            throw new ArgumentException("Edit requires at least one reference image.", nameof(referenceImages));
        if (textGuidanceScale > 1.0f && negativeCaptionEmbeddings is null)
            throw new ArgumentException("negativeCaptionEmbeddings is required when textGuidanceScale > 1.0.",
                nameof(negativeCaptionEmbeddings));

        (float rangeStart, float rangeEnd) = cfgRange ?? (0f, 1f);
        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        (int steps, _, int width, int height) = GenerationDefaults.OmniGen2.Resolve(request);
        int latentH = height / 8;
        int latentW = width / 8;

        Logs.Info($"OmniGen 2 edit: {width}x{height}, {steps} steps, textGuidance={textGuidanceScale}, " +
            $"imageGuidance={imageGuidanceScale}, cfgRange=[{rangeStart},{rangeEnd}], refs={referenceImages.Count}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        // Reference VAE encode, staged BEFORE the transformer preload (never beside the resident DiT), with a
        // single-slot content-signature cache so repeat edits of the same reference skip the encoder entirely.
        Tensor[] refLatents = GetOrEncodeRefLatents(referenceImages);

        TensorShape latentShape = new(1, _config.InChannels, latentH, latentW);
        Tensor latent = TakeOrCreateNoise(request, latentShape, seed);

        // Reference dynamic_time_shift (m = sqrt(latent_h·latent_w)/40) ≡ sigma shift m — see method docs.
        float dynamicShift = MathF.Sqrt((float)latentH * latentW) / 40.0f;
        FlowMatchEulerDiscreteScheduler scheduler = new(dynamicShift);
        scheduler.SetTimesteps(steps);
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;
        int textSeqLen = (int)captionEmbeddings.Shape[1];
        int negSeqLen = negativeCaptionEmbeddings is null ? 0 : (int)negativeCaptionEmbeddings.Shape[1];

        Backend.PreloadWeights(_transformer.EnumerateWeights());

        for (int i = 0; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float t = timesteps[i] / 1000.0f;

            // cfg_range gating (upstream: scales fall back to 1.0 outside start <= i/steps <= end).
            float schedFraction = steps > 1 ? (float)i / steps : 0f;
            bool inCfgRange = schedFraction >= rangeStart && schedFraction <= rangeEnd;
            float tg = inCfgRange ? textGuidanceScale : 1.0f;
            float ig = inCfgRange ? imageGuidanceScale : 1.0f;

            Tensor cond = _transformer.Forward(Backend, latent, t, captionEmbeddings, textSeqLen, refLatents);
            Tensor velocity;
            if (tg > 1.0f && ig > 1.0f)
            {
                Tensor refPred = _transformer.Forward(Backend, latent, t, negativeCaptionEmbeddings!, negSeqLen, refLatents);
                Tensor uncond = _transformer.Forward(Backend, latent, t, negativeCaptionEmbeddings!, negSeqLen, refLatents: null);
                velocity = CfgHelper.ApplyDualCfg(cond, refPred, uncond, tg, ig);
                refPred.Dispose();
                uncond.Dispose();
                cond.Dispose();
            }
            else if (tg > 1.0f)
            {
                Tensor uncond = _transformer.Forward(Backend, latent, t, negativeCaptionEmbeddings!, negSeqLen, refLatents: null);
                velocity = CfgHelper.ApplyCfg(uncond, cond, tg);
                uncond.Dispose();
                cond.Dispose();
            }
            else
            {
                velocity = cond;
            }

            Tensor newLatent = new(latentShape, DType.F32);
            scheduler.Step(newLatent, velocity, latent, i);
            velocity.Dispose();
            latent.Dispose();
            latent = newLatent;

            stepSw.Stop();
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds));
        }

        Backend.Sync();
        Backend.FreeWeights(_transformer.EnumerateWeights());

        Tensor decoded = _vaeDecoder.Decode(Backend, latent);
        latent.Dispose();
        byte[] rgb = ImagePostProcessor.TensorToRgbBytes(decoded);
        decoded.Dispose();

        sw.Stop();
        Logs.Info($"OmniGen 2 edit complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        return (rgb, width, height, seed);
    }

    /// <summary>VAE-encodes the reference images (or reuses the single-slot cache when their content signatures
    /// match the previous generation). Encoder weights are staged in for the encode and freed right after; the
    /// latents are host-materialized so later activation reclaims can't lose them.</summary>
    private unsafe Tensor[] GetOrEncodeRefLatents(IReadOnlyList<Tensor> referenceImages)
    {
        long refSig = ImageSignature.CombineList(referenceImages);
        if (_cachedRefLatents.Length == referenceImages.Count && _cachedRefSig == refSig)
        {
            Logs.Info($"OmniGen 2 edit: reference-latent cache hit — VAE encode skipped ({_cachedRefLatents.Length} ref(s)).");
            return _cachedRefLatents;
        }

        Stopwatch refSw = Stopwatch.StartNew();
        Backend.PreloadWeights(_vaeEncoder!.EnumerateWeights());
        Tensor[] refLatents = new Tensor[referenceImages.Count];
        for (int j = 0; j < referenceImages.Count; j++)
        {
            refLatents[j] = _vaeEncoder.Encode(Backend, referenceImages[j]);
            _ = refLatents[j].DataPointer;   // host-materialize: survives activation reclaims
        }
        Backend.Sync();
        Backend.FreeWeights(_vaeEncoder.EnumerateWeights());
        refSw.Stop();
        Logs.Info($"OmniGen 2 edit: {referenceImages.Count} reference(s) VAE-encoded in {refSw.ElapsedMilliseconds}ms.");

        // Replace the single-slot cache. Evict the old device pins BEFORE disposing: the weight cache is keyed
        // on the tensor object, and disposing a still-registered pin would leak its device allocation.
        if (_cachedRefLatents.Length > 0)
        {
            Backend.FreeWeights(_cachedRefLatents);
            foreach (Tensor old in _cachedRefLatents) old.Dispose();
        }
        _cachedRefLatents = refLatents;
        _cachedRefSig = refSig;
        return refLatents;
    }

    /// <summary>Releases the pipeline-internal reference-latent cache (components are caller-owned).</summary>
    protected override void DisposeCore()
    {
        if (_cachedRefLatents.Length > 0)
        {
            Backend.FreeWeights(_cachedRefLatents);
            foreach (Tensor old in _cachedRefLatents) old.Dispose();
            _cachedRefLatents = [];
        }
    }
}
