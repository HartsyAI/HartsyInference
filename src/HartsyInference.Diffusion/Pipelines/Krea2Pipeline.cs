using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae.QwenImage;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>Krea 2 text-to-image pipeline. Encodes the prompt through Qwen3-VL-4B (12-layer hidden-state tap) via
/// <see cref="LlamaStyleEncoder.EncodeMultiLayer"/>, runs the <see cref="Krea2Transformer"/> (which fuses the tapped
/// layers, patchifies the latent, and predicts the flow-match velocity) under a resolution-aware flow-match Euler
/// schedule, and decodes through the 16-channel Qwen-Image VAE. CFG is a dual pass when <c>request.CfgScale &gt; 1</c>
/// (Base, ~4.5); Turbo/TDM runs a single conditional pass (guidance off). See <c>docs/Research/KREA2.md</c>.</summary>
public sealed class Krea2Pipeline : DiffusionPipelineBase
{
    private readonly LlamaStyleEncoder _textEncoder;
    private readonly Krea2Transformer _transformer;
    private readonly QwenImageVaeDecoder _vaeDecoder;
    private readonly QwenImageVaeEncoder? _vaeEncoder;
    private readonly Krea2Config _config;

    /// <summary>Creates a Krea 2 pipeline. The caller owns each component's lifetime (they may be shared/reused). Img2img is unavailable; use the overload accepting a <see cref="QwenImageVaeEncoder"/> to enable it.</summary>
    public Krea2Pipeline(IBackend backend, LlamaStyleEncoder textEncoder, Krea2Transformer transformer,
        QwenImageVaeDecoder vaeDecoder, Krea2Config config)
        : this(backend, textEncoder, transformer, vaeDecoder, vaeEncoder: null, config)
    {
    }

    /// <summary>Creates a Krea 2 pipeline with both VAE halves loaded — required for img2img / inpaint (pass an <see cref="ImageToImageRequest"/> to <see cref="GenerateFromTokens"/>).</summary>
    public Krea2Pipeline(IBackend backend, LlamaStyleEncoder textEncoder, Krea2Transformer transformer,
        QwenImageVaeDecoder vaeDecoder, QwenImageVaeEncoder? vaeEncoder, Krea2Config config)
        : base(backend)
    {
        _textEncoder = textEncoder;
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _vaeEncoder = vaeEncoder;
        _config = config;
    }

    /// <summary>Generates an image. <paramref name="promptTokenIds"/> / <paramref name="negativeTokenIds"/> are the
    /// chat-templated Qwen token sequences; the leading <paramref name="promptDropIndex"/> (system-prefix) hidden
    /// states are dropped (Krea 2's <c>prompt_template_encode_start_idx = 34</c>). The negative stream is used only
    /// when <c>request.CfgScale &gt; 1</c> (Base); for Turbo pass <c>CfgScale ≤ 1</c>.
    /// <para>An <see cref="ImageToImageRequest"/> selects img2img: the source is encoded through the Qwen-Image
    /// 3D-causal VAE encoder and noised via flow-matching <c>AddNoise</c> at <c>sigma[startStep]</c> — requires a
    /// <see cref="QwenImageVaeEncoder"/> on construction. A <c>Mask</c> additionally enables blend-on-vanilla inpaint
    /// (per-step latent blend + final pixel recomposite, same as Z-Image). Strength=0 short-circuits to byte-identical
    /// pass-through.</para></summary>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromTokens(
        int[] promptTokenIds,
        int[]? negativeTokenIds,
        TextToImageRequest request,
        Action<GenerationProgress>? onProgress = null,
        int promptDropIndex = 34,
        int negativeDropIndex = 34)
    {
        ThrowIfDisposed();
        bool isImg2Img = request is ImageToImageRequest;
        if (isImg2Img && _vaeEncoder is null)
            throw new InvalidOperationException("ImageToImageRequest requires a QwenImageVaeEncoder. Construct the pipeline with the overload that accepts one.");

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        (int steps, float cfgScale, int width, int height) = GenerationDefaults.Generic.Resolve(request);
        int latentH = height / 8;
        int latentW = width / 8;
        int hPacked = latentH / _config.PatchSize;
        int wPacked = latentW / _config.PatchSize;
        int imageSeqLen = hPacked * wPacked;
        bool useCfg = cfgScale > 1.0f && negativeTokenIds is not null;

        Img2ImgSetup.Plan plan = Img2ImgSetup.Prepare(request, height, width, steps);
        if (plan.PassThrough)
        {
            Logs.Info("Strength=0; passing source through unchanged");
            return (ImagePostProcessor.TensorToRgbBytes(((ImageToImageRequest)request).SourceImage), width, height, seed);
        }
        int startStep = plan.StartStep;
        Tensor? maskPixel = plan.MaskPixel;
        bool isMaskedInpaint = maskPixel is not null;

        string opMode = isMaskedInpaint ? $"inpaint (start={startStep}/{steps})"
                      : isImg2Img ? $"img2img (start={startStep}/{steps})"
                      : "t2i";
        Logs.Info($"Krea 2 {opMode}: {width}x{height}, {steps} steps, cfg={cfgScale}, seed={seed}, distilled={_config.IsDistilled}");
        Stopwatch sw = Stopwatch.StartNew();

        // ── encode prompt: 12-layer tap → [1, S, 12·2560], drop the system prefix ──
        Backend.PreloadWeights(_textEncoder.EnumerateWeights());
        Tensor condHidden = EncodeTapped(promptTokenIds, promptDropIndex);
        Tensor? uncondHidden = useCfg ? EncodeTapped(negativeTokenIds!, negativeDropIndex) : null;
        Backend.FreeWeights(_textEncoder.EnumerateWeights());

        // ── scheduler: resolution-aware exp shift (Turbo pins mu=1.15) ──
        FlowMatchEulerDiscreteScheduler scheduler = _config.IsDistilled
            ? new FlowMatchEulerDiscreteScheduler(MathF.Exp(1.15f))
            : FlowMatchEulerDiscreteScheduler.CreateWithDynamicShift(imageSeqLen, baseSeqLen: 256, maxSeqLen: 6400,
                baseShift: 0.5f, maxShift: 1.15f);
        scheduler.SetTimesteps(steps);
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;

        TensorShape latentShape = new(1, _config.VaeChannels, latentH, latentW);
        (Tensor latent, Tensor? sourceLatent) =
            BuildInitialLatent(request, scheduler, latentShape, seed, startStep, keepSourceLatent: isMaskedInpaint);
        Tensor? latentMask = null;
        if (isMaskedInpaint)
        {
            latentMask = MaskBlendUtilities.DownsampleMaskAreaAverage(maskPixel!, latentH, latentW);
        }

        Backend.PreloadWeights(_transformer.EnumerateWeights());

        // Fast path (plain t2i, no img2img / masked-inpaint): keep the latent in the transformer's patchified token
        // space across the WHOLE sampling loop — patchify once here, run each step's flow-match Euler update
        // (x += v·dt) on-device (Scale+Add), unpatchify once after. A denoise step then never reads a tensor's
        // DataPointer, so the host queues all steps without the per-step D2H pipeline drain that serialized host
        // dispatch against GPU execution (the dominant host-bound cost). Img2img / inpaint keep the pixel-space path.
        bool fastPath = !isImg2Img && !isMaskedInpaint;
        Tensor? patchLatent = fastPath ? _transformer.PatchifyLatent(latent) : null;

        for (int i = startStep; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float t = timesteps[i] / 1000.0f; // scheduler stores sigma·1000; transformer takes t∈[0,1]

            if (fastPath)
            {
                Tensor v;
                if (useCfg)
                {
                    Tensor condV = _transformer.ForwardPatched(Backend, patchLatent!, t, condHidden, hPacked, wPacked);
                    Tensor uncondV = _transformer.ForwardPatched(Backend, patchLatent!, t, uncondHidden!, hPacked, wPacked);
                    v = CfgHelper.ApplyCfgCondAnchored(condV, uncondV, cfgScale);
                    uncondV.Dispose();
                    condV.Dispose();
                }
                else
                {
                    v = _transformer.ForwardPatched(Backend, patchLatent!, t, condHidden, hPacked, wPacked);
                }

                // On-device Euler step: patchLatent += v · dt  (Scale then Add; both stay GPU-resident).
                float dt = scheduler.Dt(i);
                Tensor vScaled = new Tensor(v.Shape, DType.F32);
                Backend.Scale(vScaled, v, dt);
                Tensor newPatch = new Tensor(patchLatent!.Shape, DType.F32);
                Backend.Add(newPatch, patchLatent!, vScaled);
                v.Dispose();
                vScaled.Dispose();
                patchLatent!.Dispose();
                patchLatent = newPatch;

                stepSw.Stop();
                onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds));
                continue;
            }

            Tensor noisePred;
            if (useCfg)
            {
                Tensor cond = _transformer.Forward(Backend, latent, t, condHidden);
                Tensor uncond = _transformer.Forward(Backend, latent, t, uncondHidden!);
                // VALIDATION-PENDING: Krea 2 uses cond-anchored CFG (cond + scale*(cond - uncond)); guidance_scale=4.5 here ≈ 5.5 under the standard uncond-anchored convention. Verify vs reference.
                noisePred = CfgHelper.ApplyCfgCondAnchored(cond, uncond, cfgScale);
                uncond.Dispose();
                cond.Dispose();
            }
            else
            {
                noisePred = _transformer.Forward(Backend, latent, t, condHidden);
            }

            Tensor newLatent = new Tensor(latentShape, DType.F32);
            scheduler.Step(newLatent, noisePred, latent, i);
            noisePred.Dispose();
            latent.Dispose();
            latent = newLatent;

            // Masked-inpaint blend: keep unmasked region on the source's flow-matching trajectory
            // by re-noising the source latent at the next step's sigma. Final step blends with the
            // clean source — no further denoising follows.
            if (latentMask is not null && sourceLatent is not null)
            {
                int nextStep = i + 1;
                Tensor noisedSource;
                if (nextStep < steps)
                {
                    Tensor freshNoise = SeedGenerator.CreateNoise(latentShape, seed + nextStep);
                    noisedSource = new Tensor(latentShape, DType.F32);
                    scheduler.AddNoise(noisedSource, sourceLatent, freshNoise, nextStep);
                    freshNoise.Dispose();
                }
                else
                {
                    noisedSource = sourceLatent;
                }
                MaskBlendUtilities.BlendChannelsInPlace(latent, noisedSource, latentMask);
                if (noisedSource != sourceLatent) noisedSource.Dispose();
            }

            stepSw.Stop();
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds));
        }

        // Fast path: bring the final patchified latent back to pixel space once for the VAE.
        if (fastPath)
        {
            latent.Dispose();
            latent = _transformer.UnpatchifyLatent(patchLatent!, hPacked, wPacked);
            patchLatent!.Dispose();
        }

        condHidden.Dispose();
        uncondHidden?.Dispose();
        sourceLatent?.Dispose();
        latentMask?.Dispose();

        Backend.Sync();
        Backend.FreeWeights(_transformer.EnumerateWeights());

        Backend.PreloadWeights(_vaeDecoder.EnumerateWeights());
        Tensor image = _vaeDecoder.Decode(Backend, latent);
        latent.Dispose();

        // Pixel-space recomposite for masked inpaint: paste decoded over source where mask=1.
        // Suppresses VAE encode/decode drift in unmasked regions (same as SDXL / Flux).
        if (isMaskedInpaint && ((ImageToImageRequest)request).RecompositeAtEnd)
        {
            MaskBlendUtilities.BlendChannelsInPlace(image, ((ImageToImageRequest)request).SourceImage, maskPixel!);
        }

        byte[] rgbData = ImagePostProcessor.TensorToRgbBytes(image);
        image.Dispose();

        sw.Stop();
        Logs.Info($"Krea 2 {opMode} complete in {sw.ElapsedMilliseconds}ms (seed={seed})");
        return (rgbData, width, height, seed);
    }

    /// <summary>Builds the initial latent. T2I: noise * initSigma. Img2img: the source is encoded through the
    /// Qwen-Image 3D-causal VAE encoder (already per-channel normalized to the transformer's latent space) and
    /// combined with fresh noise via flow-matching <c>AddNoise</c> at <c>sigma[startStep]</c>.
    /// <para>When <paramref name="keepSourceLatent"/> is true (masked inpaint), the clean source latent is returned
    /// alongside the noised latent for per-step blending. Caller disposes both. Source is null for txt2img and plain
    /// img2img.</para></summary>
    private (Tensor latent, Tensor? sourceLatent) BuildInitialLatent(TextToImageRequest request,
        FlowMatchEulerDiscreteScheduler scheduler, TensorShape latentShape, int seed, int startStep,
        bool keepSourceLatent)
    {
        if (request is ImageToImageRequest img2img)
        {
            Stopwatch vaeEncSw = Stopwatch.StartNew();
            Tensor sourceLatent = _vaeEncoder!.Encode(Backend, img2img.SourceImage);
            vaeEncSw.Stop();
            Logs.Info($"VAE encode done in {vaeEncSw.ElapsedMilliseconds}ms");

            Tensor noise = SeedGenerator.CreateNoise(latentShape, seed);
            Tensor latent = new Tensor(latentShape, DType.F32);
            scheduler.AddNoise(latent, sourceLatent, noise, startStep);
            noise.Dispose();
            if (keepSourceLatent)
            {
                return (latent, sourceLatent);
            }
            sourceLatent.Dispose();
            return (latent, null);
        }

        Tensor t2iNoise = SeedGenerator.CreateNoise(latentShape, seed);
        float initSigma = scheduler.InitialNoiseSigma;
        if (MathF.Abs(initSigma - 1.0f) > 1e-6f)
        {
            Tensor scaled = new Tensor(latentShape, DType.F32);
            Backend.Scale(scaled, t2iNoise, initSigma);
            t2iNoise.Dispose();
            return (scaled, null);
        }
        return (t2iNoise, null);
    }

    /// <summary>Encodes a token sequence, stacks the 12 selected layers (tap-major <c>[1, S, 12·2560]</c>) and drops
    /// the first <paramref name="dropIndex"/> token positions (the chat-template system prefix).</summary>
    private unsafe Tensor EncodeTapped(int[] tokenIds, int dropIndex)
    {
        Tensor full = _textEncoder.EncodeMultiLayer(Backend, [tokenIds], Krea2Config.TextEncoderSelectLayers,
            interleavedLayout: false);
        if (dropIndex <= 0) return full;

        int s = (int)full.Shape[1];
        int feat = (int)full.Shape[2];
        int keep = Math.Max(1, s - dropIndex);
        Tensor sliced = new Tensor(new TensorShape(1, keep, feat), DType.F32);
        long rowBytes = (long)feat * sizeof(float);
        float* src = (float*)full.DataPointer + (long)dropIndex * feat;
        Buffer.MemoryCopy(src, (void*)sliced.DataPointer, (long)keep * rowBytes, (long)keep * rowBytes);
        full.Dispose();
        return sliced;
    }
}
