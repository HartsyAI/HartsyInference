using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Runtime;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>Flux.2 text-to-image pipeline (Klein 4B / Klein 9B / Dev). Orchestrates Qwen3-4B (Klein) or Mistral-Small-3 (Dev) text encoding → <see cref="Flux2Transformer"/> denoising with flow matching → BN-style latent un-normalization → 2×2 unpatchify → VAE decode. Differs from Flux.1: no CLIP-L pooled / no T5, multi-layer text-encoder hidden state concat, 32-channel VAE latent, BN normalization applied at the pipeline boundary.</summary>
public sealed unsafe class Flux2Pipeline : DiffusionPipelineBase
{
    private readonly LlamaStyleEncoder _textEncoder;
    private readonly Flux2Transformer _transformer;
    private readonly VaeDecoder _vaeDecoder;
    private readonly VaeEncoder? _vaeEncoder;
    private readonly Flux2Config _config;
    private readonly Tensor _bnMean;     // [128] — running_mean of the patchified-latent BatchNorm
    private readonly Tensor _bnVar;      // [128] — running_var
    private readonly float _bnEps;
    private readonly int[] _hiddenLayers;

    /// <summary>Keeps the DiT weights GPU-resident across generations (skips the post-loop free + next-gen
    /// re-upload). A prompt-cache MISS frees the DiT before the text-encoder forward — Dev's Mistral-Small
    /// (~12 GB fp4) and the 32B Q4 DiT (~18 GB) cannot coexist on a 24 GB card, which is also why this
    /// pipeline stages weights at all. Standard-profile default ON (HARTSY_KEEP_MODELS=0 disables).</summary>
    private static readonly bool KeepModelsResident =
        EnvSwitch.IsEnabled("HARTSY_KEEP_MODELS", defaultOn: true);
    private bool _ditResident;

    // Prompt-embedding cache (last-used), keyed on the tokenizer ids — the Krea2/Chroma pattern. A hit
    // skips the whole text-encoder phase (weights never upload), which on Dev also avoids the 12 GB
    // TE↔DiT swap entirely.
    private int[]? _cachedPromptKey;
    private Tensor? _cachedTextEmbeddings;

    /// <summary>Creates a Flux.2 pipeline. Img2img is unavailable; use the overload accepting a <see cref="VaeEncoder"/> to enable it.</summary>
    public Flux2Pipeline(IBackend backend, LlamaStyleEncoder textEncoder,
        Flux2Transformer transformer, VaeDecoder vaeDecoder,
        Tensor bnMean, Tensor bnVar, Flux2Config config,
        int[]? hiddenLayers = null, float bnEps = 1e-5f)
        : this(backend, textEncoder, transformer, vaeDecoder, vaeEncoder: null, bnMean, bnVar, config, hiddenLayers, bnEps)
    {
    }

    /// <summary>Creates a Flux.2 pipeline with both VAE halves loaded. Required for img2img.</summary>
    public Flux2Pipeline(IBackend backend, LlamaStyleEncoder textEncoder,
        Flux2Transformer transformer, VaeDecoder vaeDecoder, VaeEncoder? vaeEncoder,
        Tensor bnMean, Tensor bnVar, Flux2Config config,
        int[]? hiddenLayers = null, float bnEps = 1e-5f)
        : base(backend)
    {
        _textEncoder = textEncoder;
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _vaeEncoder = vaeEncoder;
        _bnMean = bnMean;
        _bnVar = bnVar;
        _bnEps = bnEps;
        _config = config;
        // Text-encoder hidden-state taps differ by encoder (BFL flux2 text_encoder.py):
        // Mistral-Small (Dev) taps [10,20,30]; Qwen3 (Klein) taps [9,18,27].
        _hiddenLayers = hiddenLayers ??
            (config.TextEncoderType == Flux2TextEncoderType.Mistral ? [10, 20, 30] : [9, 18, 27]);
    }

    /// <summary>Generates an image from pre-tokenized prompt input. Handles both text-to-image and image-to-image via the runtime type of <paramref name="request"/>:
    /// <list type="bullet">
    /// <item>Plain <see cref="TextToImageRequest"/> → text-to-image.</item>
    /// <item><see cref="ImageToImageRequest"/> → img2img. Source goes VAE-encode (32ch latent) → 2×2 patchify → BN-normalize → pack → AddNoise at sigma[startStep]. Requires a <see cref="VaeEncoder"/>.</item>
    /// <item><see cref="ImageToImageRequest"/> with a <c>Mask</c> → blend-on-vanilla inpaint: per-step packed-latent blend keeps the unmasked region on the source's noise trajectory, plus a final pixel-space recomposite (same pattern as <see cref="FluxPipeline"/>).</item>
    /// </list>
    /// </summary>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromTokens(
        int[] promptTokenIds,
        TextToImageRequest request,
        float guidanceScale = 3.5f,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        bool isImg2Img = request is ImageToImageRequest;
        if (isImg2Img && _vaeEncoder is null)
            throw new InvalidOperationException("ImageToImageRequest requires a VaeEncoder. Construct the pipeline with the overload that accepts one.");

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        (int steps, _, int reqWidth, int reqHeight) = GenerationDefaults.Flux2.Resolve(request);

        // Round image dims down to multiple of 16 (VAE 8× × 2 patch). Latent dims are then
        // image_h/8 (latent space, 32 channels) and image_h/16 (after 2×2 patchify, 128 channels).
        int imgH = (reqHeight / _config.VaeDownscaleFactor) * _config.VaeDownscaleFactor;
        int imgW = (reqWidth / _config.VaeDownscaleFactor) * _config.VaeDownscaleFactor;
        int latH = imgH / 8;            // VAE-latent spatial (32 ch)
        int latW = imgW / 8;
        int patH = imgH / 16;           // Patchified-latent spatial (128 ch) — what the transformer sees
        int patW = imgW / 16;
        int imgSeqLen = patH * patW;

        // Img2img: validate source shape + handle strength=0 short-circuit BEFORE any model work.
        // NOTE: cannot use the shared Img2ImgSetup helper here — Flux.2 rounds image dimensions to
        // a multiple of 16, so the source-shape validation must compare against the *rounded*
        // imgH/imgW (not request.Height/Width). Img2ImgSetup hard-checks against the request, which
        // would reject any input that wasn't already 16-aligned.
        int startStep = 0;
        Tensor? maskPixel = null;
        if (request is ImageToImageRequest img2img)
        {
            Tensor src = img2img.SourceImage;
            if (src.Shape.Rank != 4 || src.Shape[0] != 1 || src.Shape[1] != 3 ||
                src.Shape[2] != imgH || src.Shape[3] != imgW)
            {
                throw new ArgumentException(
                    $"SourceImage shape must be [1, 3, {imgH}, {imgW}] (matching the 16-rounded request resolution); got {src.Shape}.",
                    nameof(request));
            }

            if (img2img.Mask is not null)
            {
                Tensor mk = img2img.Mask;
                if (mk.Shape.Rank != 4 || mk.Shape[0] != 1 || mk.Shape[1] != 1 ||
                    mk.Shape[2] != imgH || mk.Shape[3] != imgW)
                {
                    throw new ArgumentException(
                        $"Mask shape must be [1, 1, {imgH}, {imgW}] (matching the 16-rounded source); got {mk.Shape}.",
                        nameof(request));
                }
                if (mk.DType != DType.F32)
                {
                    throw new ArgumentException($"Mask must be F32 in [0, 1]; got {mk.DType}.", nameof(request));
                }
                maskPixel = mk;
            }

            float strength = Math.Clamp(img2img.Strength, 0f, 1f);
            int initTimesteps = (int)MathF.Round(steps * strength);
            startStep = Math.Max(steps - initTimesteps, 0);

            if (initTimesteps == 0)
            {
                Logs.Info("Strength=0; passing source through unchanged");
                return (ImagePostProcessor.TensorToRgbBytes(src), imgW, imgH, seed);
            }
        }
        bool isMaskedInpaint = maskPixel is not null;

        string variant = _config.TextEncoderType == Flux2TextEncoderType.Mistral ? "Dev" : "Klein";
        string opMode = isMaskedInpaint ? $"inpaint (start={startStep}/{steps})"
                      : isImg2Img ? $"img2img (start={startStep}/{steps})"
                      : "txt2img";
        Logs.Info($"Flux.2 ({variant}) {opMode}: {imgW}x{imgH}, {steps} steps, guidance={guidanceScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        // ── 1. Text encoder forward (with cross-generation prompt-embedding cache) ──
        Tensor textEmbeddings;
        if (_cachedPromptKey is not null && promptTokenIds.AsSpan().SequenceEqual(_cachedPromptKey))
        {
            Logs.Info("Flux.2 prompt-embedding cache HIT — skipping the text-encoder phase.");
            textEmbeddings = _cachedTextEmbeddings!;
        }
        else
        {
            Logs.Info($"Encoding text with {(_config.TextEncoderType == Flux2TextEncoderType.Mistral ? "Mistral-Small" : "Qwen3")} (multi-layer hidden states, layers [{string.Join(",", _hiddenLayers)}])...");

            // The TE cannot coexist with a resident DiT (Dev: ~12 GB fp4 TE + ~18 GB Q4 DiT on 24 GB);
            // evict for this encode, the denoise section re-preloads. Cache hits never pay this.
            if (_ditResident)
            {
_transformer.InvalidateStepGraph(Backend);
                                Backend.FreeWeights(_transformer.EnumerateWeights());
                _ditResident = false;
            }

            int[][] batchedTokenIds = [promptTokenIds];
            textEmbeddings = _textEncoder.EncodeMultiLayer(Backend, batchedTokenIds, _hiddenLayers);
            Logs.Info($"Text encoding done in {sw.ElapsedMilliseconds}ms (seqLen={textEmbeddings.Shape[1]}, hidden={textEmbeddings.Shape[2]})");
            LogTensorStats("text embeddings", textEmbeddings);

            // Free the TE weights (auto-promoted into the weight cache during the forward), host-materialize
            // the embeddings so they survive activation sweeps across generations, and reclaim the encoder's
            // device intermediates before the DiT phase.
            Backend.FreeWeights(_textEncoder.EnumerateWeights());
            _ = textEmbeddings.DataPointer;
            Backend.FreeActivations();

            if (_cachedTextEmbeddings != textEmbeddings) _cachedTextEmbeddings?.Dispose();
            _cachedPromptKey = (int[])promptTokenIds.Clone();
            _cachedTextEmbeddings = textEmbeddings;
        }
        int txtSeqLen = (int)textEmbeddings.Shape[1];

        // ── 2. Set up dynamic-shift flow-match scheduler ──────────────
        TensorShape noiseShape = new TensorShape(1, _config.InChannels, patH, patW);
        TensorShape packedShape = new TensorShape(1, imgSeqLen, _config.InChannels);
        FlowMatchEulerDiscreteScheduler scheduler =
            FlowMatchEulerDiscreteScheduler.CreateWithDynamicShift(imgSeqLen);
        scheduler.SetTimesteps(steps);

        // ── 3. Build initial packed latent (t2i: noise * initSigma; img2img: encode + patchify + BN-norm + pack + AddNoise) ──
        //    For masked inpaint the packed source latent + packed mask stay alive for per-step blending.
        (Tensor packedLatent, Tensor? packedSourceLatent) =
            BuildInitialPackedLatent(request, scheduler, noiseShape, packedShape, latH, latW, patH, patW, seed, startStep, keepSourceLatent: isMaskedInpaint);
        Tensor? packedMask = null;
        if (isMaskedInpaint)
        {
            // One packed-mask value per 2×2 latent sub-patch. Flux.2's patchified feature layout is
            // c·4 + (py·2 + px) — the same (tl, tr, bl, br) order PackLatentMask2x2 emits, so the Flux.1
            // packed blend applies unchanged with featDim=128 (32 latent channels × 4 sub-patches).
            Tensor latentMask = MaskBlendUtilities.DownsampleMaskAreaAverage(maskPixel!, latH, latW);
            packedMask = MaskBlendUtilities.PackLatentMask2x2(latentMask, latH, latW);
            latentMask.Dispose();
        }

        // ── 4. Denoising loop (from startStep onward) ──
        // Preload the DiT as one bulk upload (a KEEP_MODELS-resident DiT makes this a cache no-op). Without
        // it every Linear pays its own cache-miss H2D; with the Q4 GGUF source the quant blocks then stay
        // resident and dequantize transiently per use (a full F16 cast cache would not fit 24 GB for Dev).
        Backend.PreloadWeights(_transformer.EnumerateWeights());

        // Drain-free fast path (plain t2i / img2img, the Chroma/Krea2 pattern): the latent stays
        // device-resident across the loop and the Euler step runs as ONE in-place device op (Flux.2 has no
        // true-CFG — guidance is embedded — so CfgEulerStep with g=1 is the plain z += v·dt). Masked inpaint
        // keeps the host branch (per-step host blend must read the latent anyway).
        bool drainFree = !isMaskedInpaint;
        Logs.Info(drainFree ? "Flux.2 loop: drain-free device-resident path." : "Flux.2 loop: host-step path.");

        // Persistent step graph (the Chroma/Flux.1 recipe): route the latent through the transformer-owned
        // FIXED buffer so the whole forward is captured once per (prompt ref, grid) signature and replayed
        // with one cuGraphLaunch per step; it survives across generations (this pipeline never sweeps
        // activations on the drain-free route, and the post-decode reclaim below trims instead).
        // StepGraphEnabled is per-checkpoint: default-ON for Klein (BF16), opt-in HARTSY_DIT_GRAPH=1 for the
        // Q4-GGUF Dev whose transient per-GEMM dequants OOM the capture on 24 GB (see Flux2Transformer).
        bool graphRoute = drainFree && packedSourceLatent is null
            && _transformer.StepGraphEnabled && Backend.StepGraphSupported;
        if (graphRoute)
        {
            Tensor fixedLatent = _transformer.PrepareGraphLatent(Backend, packedLatent);
            packedLatent.Dispose();
            packedLatent = fixedLatent;
        }

        ReadOnlySpan<float> timesteps = scheduler.Timesteps;
        for (int i = startStep; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float sigma = timesteps[i] / 1000.0f;

            if (graphRoute)
            {
                (Tensor velocity, bool callerOwns) = _transformer.ForwardGraphable(
                    Backend, packedLatent, textEmbeddings, sigma, guidanceScale, patH, patW);
                // Euler stays OUTSIDE the capture: in-place on the fixed latent the graph reads next step.
                Backend.CfgEulerStep(packedLatent, velocity, velocity, 1.0f, scheduler.Dt(i));
                if (callerOwns) velocity.Dispose();

                stepSw.Stop();
                Logs.Debug($"Step {i + 1}/{steps} (sigma={sigma:F4}) done in {stepSw.ElapsedMilliseconds}ms");
                onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds)
                {
                    LatentArch = LatentArchitecture.Flux2,
                });
                continue;
            }

            Tensor velocityPred = _transformer.Forward(
                Backend, packedLatent, textEmbeddings, sigma, guidanceScale, patH, patW);

            if (drainFree)
            {
                Backend.CfgEulerStep(packedLatent, velocityPred, velocityPred, 1.0f, scheduler.Dt(i));
                velocityPred.Dispose();
            }
            else
            {
            TensorShape packedStepShape = new TensorShape(1, imgSeqLen, _config.InChannels);
            Tensor newLatent = new Tensor(packedStepShape, DType.F32);
            scheduler.Step(newLatent, velocityPred, packedLatent, i);
            velocityPred.Dispose();
            packedLatent.Dispose();
            packedLatent = newLatent;
            }

            // Masked-inpaint blend in packed form: keep unmasked region on the source's
            // flow-matching trajectory by re-noising the source at the next step's sigma.
            // Final step blends with the clean source — no further denoising follows.
            if (packedMask is not null && packedSourceLatent is not null)
            {
                int nextStep = i + 1;
                Tensor noisedSource;
                if (nextStep < steps)
                {
                    Tensor freshNoise = SeedGenerator.CreateNoise(noiseShape, seed + nextStep);
                    Tensor freshPackedNoise = PackLatent(freshNoise);
                    freshNoise.Dispose();
                    noisedSource = new Tensor(packedShape, DType.F32);
                    scheduler.AddNoise(noisedSource, packedSourceLatent, freshPackedNoise, nextStep);
                    freshPackedNoise.Dispose();
                }
                else
                {
                    noisedSource = packedSourceLatent;
                }
                MaskBlendUtilities.BlendPackedInPlace(packedLatent, noisedSource, packedMask);
                if (noisedSource != packedSourceLatent) noisedSource.Dispose();
            }

            stepSw.Stop();
            Logs.Debug($"Step {i + 1}/{steps} (sigma={sigma:F4}) done in {stepSw.ElapsedMilliseconds}ms");
            // No live preview for Flux.2 yet. Three things would need to line up:
            //   1. Unpack packedLatent [B, S, 128] → [B, 128, patH, patW] (mechanical).
            //   2. Reverse the pipeline's BN normalization using _bnMean/_bnVar (also mechanical).
            //   3. Unpatchify [B, 128, patH, patW] → [B, 32, latH, latW] (already in this file).
            // But (4) we'd then need a Flux.2-specific 32×3 latent2rgb factor matrix that
            // doesn't exist publicly, AND there's no published TAESD checkpoint for the
            // 32-channel Flux.2 VAE. So Flux.2 emits progress percentages only — no preview
            // frames. PreviewEncoder skips when Latent is null.
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds)
            {
                LatentArch = LatentArchitecture.Flux2,
            });
        }

        // On the graph route packedLatent IS the transformer-owned fixed buffer (alive across generations
        // for graph replay) — hand a snapshot to the unpack/dispose tail instead.
        if (graphRoute)
            packedLatent = _transformer.SnapshotGraphLatent(Backend);

        // textEmbeddings is a cross-generation cache entry — not disposed here.
        packedSourceLatent?.Dispose();
        packedMask?.Dispose();

        // Free the DiT before VAE decode unless KEEP_MODELS keeps it resident (the 32-channel decode's
        // tiled fallback covers the tight case; a future prompt-cache miss evicts before the TE).
        if (!KeepModelsResident)
        {
            _transformer.InvalidateStepGraph(Backend);
            Backend.FreeWeights(_transformer.EnumerateWeights());
            _ditResident = false;
        }
        else
        {
            _ditResident = true;
        }

        // ── 5. Unpack [B, S, 128] → [B, 128, patH, patW] ──────────────
        Tensor unpackedPatched = UnpackLatent(packedLatent, patH, patW);
        packedLatent.Dispose();

        // ── 6. BN un-normalize on the 128-channel patchified latent ──
        // latent = latent * sqrt(running_var + eps) + running_mean
        Tensor unBn = ApplyBnUnNormalize(unpackedPatched, _bnMean, _bnVar, _bnEps);
        unpackedPatched.Dispose();

        // ── 7. 2×2 unpatchify: [B, 128, patH, patW] → [B, 32, latH, latW] ──
        Tensor latent32 = UnpatchifyLatent(unBn, _config.VaeLatentChannels, _config.PatchSize);
        unBn.Dispose();

        // ── 8. VAE decode: [B, 32, latH, latW] → [B, 3, imgH, imgW] ──
        // Tiled decode: caps im2col workspace at ~2.4 GB per tile.
        Logs.Verbose("Decoding latents (tiled F32 path)...");
        Stopwatch vaeSw = Stopwatch.StartNew();
        Tensor image = _vaeDecoder.DecodeTiled(Backend, latent32);
        latent32.Dispose();
        vaeSw.Stop();
        Logs.Verbose($"VAE decode done in {vaeSw.ElapsedMilliseconds}ms");

        // ── 9. Pixel-space recomposite for masked inpaint: paste decoded over source where mask=1.
        //    Suppresses VAE encode/decode drift in unmasked regions (same as SDXL / Flux.1).
        if (isMaskedInpaint && ((ImageToImageRequest)request).RecompositeAtEnd)
        {
            MaskBlendUtilities.BlendChannelsInPlace(image, ((ImageToImageRequest)request).SourceImage, maskPixel!);
        }

        // ── 10. RGB conversion ────────────────────────────────────────
        byte[] rgbData = ImagePostProcessor.TensorToRgbBytes(image);
        image.Dispose();

        // Final reclaim (the Flux.1 post-decode idiom): in a long-lived host the VAE-decode intermediates
        // otherwise hold pool reservation against the next generation / model switch. While a captured step
        // graph is alive, FreeActivations would destroy it (and free its fixed latent/velocity buffers) —
        // trim the pool instead; per-op disposal already freed the intermediates, so the trim returns the
        // unused reservation without touching live buffers. Everything that must survive across generations
        // (cached prompt embeddings) is host-materialized, and the rope tables live in the weight cache.
        if (Backend.StepGraphReady)
            Backend.TrimMemoryPool();
        else
            Backend.FreeActivations();

        sw.Stop();
        Logs.Info($"Flux.2 image generation complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        return (rgbData, imgW, imgH, seed);
    }

    /// <summary>Packs [B, C, H, W] to [B, H*W, C] (no 2×2 spatial reshuffle — Flux.2 noise is already in patchified form). Equivalent to <c>view+permute+reshape</c> per diffusers <c>_pack_latents</c>.</summary>
    private static Tensor PackLatent(Tensor latent)
    {
        int batch = (int)latent.Shape[0];
        int channels = (int)latent.Shape[1];
        int h = (int)latent.Shape[2];
        int w = (int)latent.Shape[3];
        int seqLen = h * w;
        TensorShape outShape = new TensorShape(batch, seqLen, channels);
        Tensor packed = new Tensor(outShape, DType.F32);
        float* inPtr = (float*)latent.DataPointer;
        float* outPtr = (float*)packed.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    int seqIdx = y * w + x;
                    int outBase = (b * seqLen + seqIdx) * channels;
                    for (int c = 0; c < channels; c++)
                    {
                        int inIdx = ((b * channels + c) * h + y) * w + x;
                        outPtr[outBase + c] = inPtr[inIdx];
                    }
                }
            }
        }
        return packed;
    }

    /// <summary>Unpacks [B, H*W, C] back to [B, C, H, W].</summary>
    private static Tensor UnpackLatent(Tensor packed, int h, int w)
    {
        int batch = (int)packed.Shape[0];
        int channels = (int)packed.Shape[2];
        int seqLen = h * w;
        TensorShape outShape = new TensorShape(batch, channels, h, w);
        Tensor unpacked = new Tensor(outShape, DType.F32);
        float* inPtr = (float*)packed.DataPointer;
        float* outPtr = (float*)unpacked.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    int seqIdx = y * w + x;
                    int inBase = (b * seqLen + seqIdx) * channels;
                    for (int c = 0; c < channels; c++)
                    {
                        int outIdx = ((b * channels + c) * h + y) * w + x;
                        outPtr[outIdx] = inPtr[inBase + c];
                    }
                }
            }
        }
        return unpacked;
    }

    /// <summary>Applies BatchNorm un-normalization on the patchified latent: <c>z = z * sqrt(var + eps) + mean</c>, per-channel (mean/var have shape <c>[128]</c>).</summary>
    private static Tensor ApplyBnUnNormalize(Tensor latent, Tensor mean, Tensor var, float eps)
    {
        int batch = (int)latent.Shape[0];
        int channels = (int)latent.Shape[1];
        int h = (int)latent.Shape[2];
        int w = (int)latent.Shape[3];
        int spatial = h * w;
        Tensor output = new Tensor(latent.Shape, DType.F32);
        float* inPtr = (float*)latent.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        float* meanPtr = (float*)mean.DataPointer;
        float* varPtr = (float*)var.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                float std = MathF.Sqrt(varPtr[c] + eps);
                float m = meanPtr[c];
                int chanBase = (b * channels + c) * spatial;
                for (int s = 0; s < spatial; s++)
                    outPtr[chanBase + s] = inPtr[chanBase + s] * std + m;
            }
        }
        return output;
    }

    /// <summary>2×2 spatial unpatchify: <c>[B, C*4, H, W] → [B, C, H*2, W*2]</c>. Equivalent to nn.PixelShuffle(2) applied per-channel-group.</summary>
    private static Tensor UnpatchifyLatent(Tensor input, int outChannels, int patchSize)
    {
        int batch = (int)input.Shape[0];
        int inChannels = (int)input.Shape[1];
        int h = (int)input.Shape[2];
        int w = (int)input.Shape[3];
        int outH = h * patchSize;
        int outW = w * patchSize;
        if (inChannels != outChannels * patchSize * patchSize)
            throw new InvalidOperationException(
                $"UnpatchifyLatent: in_channels ({inChannels}) must equal out_channels * patch² ({outChannels} * {patchSize}² = {outChannels * patchSize * patchSize})");

        TensorShape outShape = new TensorShape(batch, outChannels, outH, outW);
        Tensor output = new Tensor(outShape, DType.F32);
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        // Per the diffusers permute (0, 1, 4, 2, 5, 3): viewed shape is
        //   [B, C, P, P, H, W]
        // permuted to
        //   [B, C, H, P, W, P]
        // i.e. for output (b, c, oy, ox) where oy = y*P + py and ox = x*P + px,
        //   value comes from input view at (b, c, py, px, y, x), which in the original
        //   [B, C*P*P, H, W] tensor is at channel index `c * (P*P) + py * P + px`.
        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < outChannels; c++)
            {
                for (int y = 0; y < h; y++)
                {
                    for (int x = 0; x < w; x++)
                    {
                        for (int py = 0; py < patchSize; py++)
                        {
                            for (int px = 0; px < patchSize; px++)
                            {
                                int inChannel = c * patchSize * patchSize + py * patchSize + px;
                                int inIdx = ((b * inChannels + inChannel) * h + y) * w + x;
                                int oy = y * patchSize + py;
                                int ox = x * patchSize + px;
                                int outIdx = ((b * outChannels + c) * outH + oy) * outW + ox;
                                outPtr[outIdx] = inPtr[inIdx];
                            }
                        }
                    }
                }
            }
        }
        return output;
    }

    /// <summary>2×2 spatial patchify (inverse of <see cref="UnpatchifyLatent"/>): <c>[B, C, H*P, W*P] → [B, C*P², H, W]</c>. Used by img2img to convert the VAE-encoded source latent into the patchified form the transformer operates on.</summary>
    private static Tensor PatchifyLatent(Tensor input, int outChannels, int patchSize)
    {
        int batch = (int)input.Shape[0];
        int inChannels = (int)input.Shape[1];
        int inH = (int)input.Shape[2];
        int inW = (int)input.Shape[3];
        if (outChannels != inChannels * patchSize * patchSize)
            throw new InvalidOperationException(
                $"PatchifyLatent: out_channels ({outChannels}) must equal in_channels * patch² ({inChannels} * {patchSize}² = {inChannels * patchSize * patchSize})");

        int outH = inH / patchSize;
        int outW = inW / patchSize;
        TensorShape outShape = new TensorShape(batch, outChannels, outH, outW);
        Tensor output = new Tensor(outShape, DType.F32);
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < inChannels; c++)
            {
                for (int y = 0; y < outH; y++)
                {
                    for (int x = 0; x < outW; x++)
                    {
                        for (int py = 0; py < patchSize; py++)
                        {
                            for (int px = 0; px < patchSize; px++)
                            {
                                int outChannel = c * patchSize * patchSize + py * patchSize + px;
                                int outIdx = ((b * outChannels + outChannel) * outH + y) * outW + x;
                                int iy = y * patchSize + py;
                                int ix = x * patchSize + px;
                                int inIdx = ((b * inChannels + c) * inH + iy) * inW + ix;
                                outPtr[outIdx] = inPtr[inIdx];
                            }
                        }
                    }
                }
            }
        }
        return output;
    }

    /// <summary>Inverse of <see cref="ApplyBnUnNormalize"/>: <c>(latent - mean) / sqrt(var + eps)</c>. Used by img2img to renormalize the VAE-encoded source into the BN-normalized space the transformer expects.</summary>
    private static Tensor ApplyBnNormalize(Tensor latent, Tensor mean, Tensor var, float eps)
    {
        int batch = (int)latent.Shape[0];
        int channels = (int)latent.Shape[1];
        int h = (int)latent.Shape[2];
        int w = (int)latent.Shape[3];
        int spatial = h * w;
        Tensor output = new Tensor(latent.Shape, DType.F32);
        float* inPtr = (float*)latent.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        float* meanPtr = (float*)mean.DataPointer;
        float* varPtr = (float*)var.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                float invStd = 1.0f / MathF.Sqrt(varPtr[c] + eps);
                float m = meanPtr[c];
                int chanBase = (b * channels + c) * spatial;
                for (int s = 0; s < spatial; s++)
                    outPtr[chanBase + s] = (inPtr[chanBase + s] - m) * invStd;
            }
        }
        return output;
    }

    /// <summary>Builds the initial packed latent for Flux.2 denoising. T2I: fresh noise scaled by initSigma. Img2img: source goes VAE-encode → 2×2 patchify → BN-normalize → pack → AddNoise at sigma[startStep].
    /// <para>When <paramref name="keepSourceLatent"/> is true (masked inpaint), the clean BN-normalized packed source is returned alongside the noised latent for per-step blending. Caller disposes both. Source is null for txt2img and plain img2img.</para></summary>
    private (Tensor packedLatent, Tensor? packedSourceLatent) BuildInitialPackedLatent(
        TextToImageRequest request,
        FlowMatchEulerDiscreteScheduler scheduler,
        TensorShape noiseShape,
        TensorShape packedShape,
        int latH, int latW, int patH, int patW,
        int seed, int startStep,
        bool keepSourceLatent)
    {
        Tensor unpackedNoise = SeedGenerator.CreateNoise(noiseShape, seed);
        Tensor packedNoise = PackLatent(unpackedNoise);
        unpackedNoise.Dispose();

        if (request is ImageToImageRequest img2img)
        {
            Stopwatch vaeEncSw = Stopwatch.StartNew();
            // VaeConfig.Flux2 has ScalingFactor=1.0 + ShiftFactor=null, so encoder output is the raw mean (no scaling).
            Tensor sourceLatent32 = _vaeEncoder!.Encode(Backend, img2img.SourceImage);  // [1, 32, latH, latW]
            vaeEncSw.Stop();
            Logs.Info($"VAE encode done in {vaeEncSw.ElapsedMilliseconds}ms");

            // 2×2 patchify: [1, 32, latH, latW] → [1, 128, patH, patW]
            Tensor sourcePatchified = PatchifyLatent(sourceLatent32, _config.InChannels, _config.PatchSize);
            sourceLatent32.Dispose();

            // BN normalize: (latent - running_mean) / sqrt(running_var + eps)
            Tensor sourceBn = ApplyBnNormalize(sourcePatchified, _bnMean, _bnVar, _bnEps);
            sourcePatchified.Dispose();

            // Pack: [1, 128, patH, patW] → [1, S, 128]
            Tensor sourcePacked = PackLatent(sourceBn);
            sourceBn.Dispose();

            // AddNoise at sigma[startStep]: noisy = (1 - sigma) * source + sigma * noise
            Tensor result = new Tensor(packedShape, DType.F32);
            scheduler.AddNoise(result, sourcePacked, packedNoise, startStep);
            packedNoise.Dispose();
            if (keepSourceLatent)
            {
                return (result, sourcePacked);
            }
            sourcePacked.Dispose();
            return (result, null);
        }

        // T2I path: scale packed noise by initSigma.
        float initSigma = scheduler.InitialNoiseSigma;
        if (MathF.Abs(initSigma - 1.0f) > 1e-6f)
        {
            Tensor scaled = new Tensor(packedShape, DType.F32);
            Backend.Scale(scaled, packedNoise, initSigma);
            packedNoise.Dispose();
            return (scaled, null);
        }
        return (packedNoise, null);
    }

    private static void LogTensorStats(string name, Tensor tensor)
    {
        ReadOnlySpan<float> data = tensor.AsReadOnlySpan<float>();
        float min = float.MaxValue, max = float.MinValue;
        double sum = 0;
        int nan = 0, inf = 0;
        for (int i = 0; i < data.Length; i++)
        {
            float v = data[i];
            if (float.IsNaN(v)) { nan++; continue; }
            if (float.IsInfinity(v)) { inf++; continue; }
            if (v < min) min = v;
            if (v > max) max = v;
            sum += v;
        }
        Logs.Debug($"  [{name}] shape={tensor.Shape} min={min:E3} max={max:E3} mean={sum / data.Length:E3} nan={nan} inf={inf}");
    }
}
