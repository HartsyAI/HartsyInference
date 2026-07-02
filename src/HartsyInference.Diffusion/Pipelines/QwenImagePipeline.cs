using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Models.Vae.QwenImage;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>Qwen-Image text-to-image pipeline. Encodes the prompt through Qwen2.5-VL via <see cref="LlamaStyleEncoder"/>, packs the noisy latent into 2×2 patch tokens, runs <see cref="QwenImageTransformer"/> with flow-match Euler scheduling (dynamic shift), unpacks the predicted velocity back to <c>[B, 16, H, W]</c>, and decodes through the 16-channel Qwen-Image VAE. CFG is applied as a dual-pass when <c>request.CfgScale &gt; 1</c>; otherwise a single conditional forward is used. Transformer + text encoder weights are evicted from VRAM before VAE decode (Phase 3 deviations #18 / #33) so a 30 GB Qwen-Image FP8 stack still leaves room for tiled decode workspace on a 40 GB card.</summary>
public sealed unsafe class QwenImagePipeline : DiffusionPipelineBase
{
    private readonly LlamaStyleEncoder _textEncoder;
    private readonly QwenImageTransformer _transformer;
    private readonly QwenImageVaeDecoder _vaeDecoder;
    private readonly QwenImageVaeEncoder? _vaeEncoder;
    private readonly QwenImageConfig _config;

    /// <summary>Creates a Qwen-Image pipeline. Caller is responsible for the lifetime of the components — they may be reused across pipelines. Img2img is unavailable; use the overload accepting a <see cref="QwenImageVaeEncoder"/> to enable it.</summary>
    public QwenImagePipeline(IBackend backend, LlamaStyleEncoder textEncoder,
        QwenImageTransformer transformer, QwenImageVaeDecoder vaeDecoder, QwenImageConfig config)
        : this(backend, textEncoder, transformer, vaeDecoder, vaeEncoder: null, config)
    {
    }

    /// <summary>Creates a Qwen-Image pipeline with both VAE halves loaded — required for img2img / inpaint (pass an <see cref="ImageToImageRequest"/> to <see cref="GenerateFromTokens"/>).</summary>
    public QwenImagePipeline(IBackend backend, LlamaStyleEncoder textEncoder,
        QwenImageTransformer transformer, QwenImageVaeDecoder vaeDecoder, QwenImageVaeEncoder? vaeEncoder, QwenImageConfig config)
        : base(backend)
    {
        _textEncoder = textEncoder;
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _vaeEncoder = vaeEncoder;
        _config = config;
    }

    /// <summary>Generates an image from pre-tokenized prompt input. The negative prompt tokens are ignored when <see cref="TextToImageRequest.CfgScale"/> ≤ 1. An <see cref="ImageToImageRequest"/> selects img2img: the source is encoded via the Qwen-Image 3D-causal VAE encoder, packed (2×2 patchify), and combined with fresh packed noise via flow-matching <c>AddNoise</c> at <c>sigma[startStep]</c> — requires a <see cref="QwenImageVaeEncoder"/> on construction. A <c>Mask</c> additionally enables blend-on-vanilla inpaint (per-step packed blend + final pixel recomposite, same as Flux). Strength=0 short-circuits to byte-identical pass-through.</summary>
    /// <param name="promptTokenIds">Conditional prompt token IDs (Qwen2.5-VL vocab).</param>
    /// <param name="negativeTokenIds">Negative-prompt token IDs (same length as <paramref name="promptTokenIds"/> recommended). Used only when <c>CfgScale &gt; 1</c>.</param>
    /// <param name="request">Generation parameters. Pass an <see cref="ImageToImageRequest"/> for img2img / inpaint.</param>
    /// <param name="onProgress">Optional per-step progress callback.</param>
    /// <param name="promptDropIndex">Number of leading hidden-state positions to drop from the conditional
    /// stream after encoding. Qwen-Image wraps the prompt in a system+user-header template and discards
    /// those prefix tokens' hidden states (diffusers' <c>prompt_template_encode_start_idx</c>). 0 = keep all
    /// (raw-prompt path).</param>
    /// <param name="negativeDropIndex">Same as <paramref name="promptDropIndex"/> for the negative stream.</param>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromTokens(
        int[] promptTokenIds,
        int[] negativeTokenIds,
        TextToImageRequest request,
        Action<GenerationProgress>? onProgress = null,
        int promptDropIndex = 0,
        int negativeDropIndex = 0)
    {
        ThrowIfDisposed();
        bool isImg2Img = request is ImageToImageRequest;
        if (isImg2Img && _vaeEncoder is null)
            throw new InvalidOperationException("ImageToImageRequest requires a QwenImageVaeEncoder. Construct the pipeline with the overload that accepts one.");

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int width = request.Width ?? GenerationDefaults.QwenImage.Width;
        int height = request.Height ?? GenerationDefaults.QwenImage.Height;
        int latentH = height / 8;
        int latentW = width / 8;
        int hPacked = latentH / _config.PatchSize;
        int wPacked = latentW / _config.PatchSize;
        int imgSeqLen = hPacked * wPacked;
        int patchDim = _config.PatchSize * _config.PatchSize * _config.InChannels;
        int steps = request.Steps ?? GenerationDefaults.QwenImage.Steps;
        float cfgScale = request.CfgScale ?? GenerationDefaults.QwenImage.CfgScale;
        bool useCfg = cfgScale > 1.0f;

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
                      : "txt2img";
        Logs.Info($"Qwen-Image {opMode}: {width}x{height}, {steps} steps, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        Backend.PreloadWeights(_textEncoder.EnumerateWeights());

        int[][] batchPrompt = [promptTokenIds];
        Tensor condHidden = _textEncoder.Encode(Backend, batchPrompt);
        if (promptDropIndex > 0)
        {
            Tensor trimmed = DropPrefixHiddenStates(condHidden, promptDropIndex);
            condHidden.Dispose();
            condHidden = trimmed;
        }

        Tensor? uncondHidden = null;
        if (useCfg)
        {
            int[][] batchNeg = [negativeTokenIds];
            uncondHidden = _textEncoder.Encode(Backend, batchNeg);
            if (negativeDropIndex > 0)
            {
                Tensor trimmed = DropPrefixHiddenStates(uncondHidden, negativeDropIndex);
                uncondHidden.Dispose();
                uncondHidden = trimmed;
            }
        }

        Logs.Info($"Text encoding done in {sw.ElapsedMilliseconds}ms (txt seqLen={condHidden.Shape[1]})");

        Backend.FreeWeights(_textEncoder.EnumerateWeights());

        // Materialize the conditioning on the host, then reclaim every encoder intermediate. The Qwen2.5-VL
        // encoder leaves hundreds of device-cached activations that otherwise linger until GC finalization —
        // they'd hold multi-GB into the DiT phase. condHidden/uncondHidden may still be live GPU activations
        // (the drop-index path already copied them on the host, the raw path did not); touching DataPointer
        // forces the D2H sync + cache eviction, making them safe across the FreeActivations calls below.
        _ = condHidden.DataPointer;
        if (uncondHidden is not null) _ = uncondHidden.DataPointer;
        Backend.FreeActivations();

        TensorShape latentShape = new TensorShape(1, _config.InChannels, latentH, latentW);
        TensorShape packedShape = new TensorShape(1, imgSeqLen, patchDim);

        // Qwen-Image scheduler constants (Qwen/Qwen-Image scheduler_config.json) — NOT Flux's defaults:
        // max_image_seq_len 8192 (Flux 4096), max_shift 0.9 (Flux 1.15); base 256 / 0.5 match.
        // shift_terminal 0.02 is applied inside SetTimesteps below.
        FlowMatchEulerDiscreteScheduler scheduler =
            FlowMatchEulerDiscreteScheduler.CreateWithDynamicShift(
                imgSeqLen, baseSeqLen: 256, maxSeqLen: 8192, baseShift: 0.5f, maxShift: 0.9f, shiftTerminal: 0.02f);
        scheduler.SetTimesteps(steps);

        // Build initial packed latent — t2i: packed noise * initSigma; img2img: encode + pack + AddNoise
        // at sigma[startStep]. Masked inpaint keeps the packed source + packed mask for per-step blend.
        (Tensor packedLatent, Tensor? packedSourceLatent) =
            BuildInitialPackedLatent(request, scheduler, latentShape, packedShape, latentH, latentW, seed, startStep, keepSourceLatent: isMaskedInpaint);
        Tensor? packedMask = null;
        if (isMaskedInpaint)
        {
            if (_config.PatchSize != 2)
                throw new InvalidOperationException(
                    $"Masked inpaint requires patch size 2 (PackLatentMask2x2); config has {_config.PatchSize}.");
            Tensor latentMask = MaskBlendUtilities.DownsampleMaskAreaAverage(maskPixel!, latentH, latentW);
            packedMask = MaskBlendUtilities.PackLatentMask2x2(latentMask, latentH, latentW);
            latentMask.Dispose();
        }

        Backend.PreloadWeights(_transformer.EnumerateWeights());

        // Materialize every tensor that must survive across steps on the host, then reclaim the VAE-encode /
        // packing intermediates. The per-step FreeActivations below frees device buffers WITHOUT a D2H
        // sync-back, so anything still device-only here would be silently lost — the t2i initSigma path leaves
        // packedLatent as a live Backend.Scale output on the GPU.
        _ = packedLatent.DataPointer;
        if (packedSourceLatent is not null) _ = packedSourceLatent.DataPointer;
        if (packedMask is not null) _ = packedMask.DataPointer;
        Backend.FreeActivations();

        Logs.Info("Starting Qwen-Image denoising loop...");
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;

        for (int i = startStep; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float t = timesteps[i];
            float normalizedT = t / 1000.0f;

            Tensor noisePred;
            if (useCfg)
            {
                Tensor condPred = _transformer.Forward(Backend, packedLatent, condHidden, normalizedT, hPacked, wPacked);
                Tensor uncondPred = _transformer.Forward(Backend, packedLatent, uncondHidden!, normalizedT, hPacked, wPacked);
                noisePred = CfgHelper.ApplyCfg(uncondPred, condPred, cfgScale);
                uncondPred.Dispose();
                condPred.Dispose();
            }
            else
            {
                noisePred = _transformer.Forward(Backend, packedLatent, condHidden, normalizedT, hPacked, wPacked);
            }

            Tensor newLatent = new Tensor(packedShape, DType.F32);
            scheduler.Step(newLatent, noisePred, packedLatent, i);
            noisePred.Dispose();
            packedLatent.Dispose();
            packedLatent = newLatent;

            // Masked-inpaint blend in packed form: keep unmasked region on the source's
            // flow-matching trajectory by re-noising the source latent at the next step's
            // sigma. Final step blends with the clean source — no further denoising follows.
            if (packedMask is not null && packedSourceLatent is not null)
            {
                int nextStep = i + 1;
                Tensor noisedSource;
                if (nextStep < steps)
                {
                    Tensor freshUnpackedNoise = SeedGenerator.CreateNoise(latentShape, seed + nextStep);
                    Tensor freshPackedNoise = PackLatent(freshUnpackedNoise, latentH, latentW, _config.InChannels, _config.PatchSize);
                    freshUnpackedNoise.Dispose();
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
            Logs.Debug($"Step {i + 1}/{steps} (t={t:F1}) done in {stepSw.ElapsedMilliseconds}ms");
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds));

            // Reclaim GPU-resident activation buffers between steps: the DiT keeps intermediates on-device and
            // any not read-back/disposed linger until GC, accumulating to OOM over the schedule (same fix as
            // Flux / the video pipelines). Safe: scheduler.Step runs on the host, so packedLatent — the only
            // tensor the next step needs — is host-resident, and everything else persistent was materialized
            // above the loop.
            Backend.FreeActivations();
        }

        condHidden.Dispose();
        uncondHidden?.Dispose();
        packedSourceLatent?.Dispose();
        packedMask?.Dispose();

        QwenImageTransformer.DumpFinalLatent(packedLatent);

        Backend.Sync();
        Backend.FreeWeights(_transformer.EnumerateWeights());

        Tensor unpacked = UnpackLatent(packedLatent, latentH, latentW, _config.InChannels, _config.PatchSize);
        packedLatent.Dispose();

        // The QwenImage VAE is the WAN-2.1-family 3D causal autoencoder (decoder.conv1/upsamples/…), NOT a
        // diffusers AutoencoderKL — it must decode through QwenImageVaeDecoder (same as AnimaPipeline). The
        // per-channel latent un-normalization (latents_mean/std from VaeConfig.QwenImage) happens INSIDE
        // QwenImageVaeDecoder.Decode → UndoScaling, so we pass the unpacked latent directly (no scalar
        // shift/scale here — VaeConfig.QwenImage's scale=1/shift=0 made the old ApplyVaeShiftScale a no-op
        // anyway, and it skipped the real per-channel step).
        Backend.PreloadWeights(_vaeDecoder.EnumerateWeights());
        Logs.Info("Decoding latents to image (QwenImage 3D-causal VAE)...");
        Stopwatch vaeSw = Stopwatch.StartNew();
        Tensor image = _vaeDecoder.Decode(Backend, unpacked);
        unpacked.Dispose();
        vaeSw.Stop();
        Logs.Info($"VAE decode done in {vaeSw.ElapsedMilliseconds}ms");

        // Pixel-space recomposite for masked inpaint: paste decoded over source where mask=1.
        // Suppresses VAE encode/decode drift in unmasked regions (same as SDXL / Flux).
        if (isMaskedInpaint && ((ImageToImageRequest)request).RecompositeAtEnd)
        {
            MaskBlendUtilities.BlendChannelsInPlace(image, ((ImageToImageRequest)request).SourceImage, maskPixel!);
        }

        byte[] rgbData = ImagePostProcessor.TensorToRgbBytes(image);
        image.Dispose();

        // Final reclaim: in a long-lived host (SwarmUI), VAE-decode intermediates otherwise sit in device
        // memory until GC finalization and shrink the budget of whatever generation runs next.
        Backend.FreeActivations();

        sw.Stop();
        Logs.Info($"Qwen-Image {opMode} complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        return (rgbData, width, height, seed);
    }

    /// <summary>Builds the initial packed latent for Qwen-Image denoising. T2I: fresh Gaussian noise packed and scaled by the scheduler's initial sigma. Img2img: source encoded via the Qwen-Image 3D-causal VAE encoder (already per-channel normalized to the transformer's latent space), packed via <see cref="PackLatent"/>, and combined with fresh packed noise via flow-matching <c>AddNoise</c>: <c>noisy = (1-sigma) * source + sigma * noise</c>.
    /// <para>When <paramref name="keepSourceLatent"/> is true (masked inpaint), the packed source latent is returned as the second tuple element for per-step blending. Caller disposes both. Source is null for txt2img and plain img2img.</para></summary>
    private (Tensor packedLatent, Tensor? packedSourceLatent) BuildInitialPackedLatent(
        TextToImageRequest request,
        FlowMatchEulerDiscreteScheduler scheduler,
        TensorShape latentShape,
        TensorShape packedShape,
        int latentH, int latentW, int seed, int startStep,
        bool keepSourceLatent)
    {
        Tensor unpackedNoise = SeedGenerator.CreateNoise(latentShape, seed);
        Tensor packedNoise = PackLatent(unpackedNoise, latentH, latentW, _config.InChannels, _config.PatchSize);
        unpackedNoise.Dispose();

        if (request is ImageToImageRequest img2img)
        {
            Stopwatch vaeEncSw = Stopwatch.StartNew();
            Tensor sourceUnpacked = _vaeEncoder!.Encode(Backend, img2img.SourceImage);  // [1, 16, latentH, latentW]
            vaeEncSw.Stop();
            Logs.Info($"VAE encode done in {vaeEncSw.ElapsedMilliseconds}ms");

            Tensor sourcePacked = PackLatent(sourceUnpacked, latentH, latentW, _config.InChannels, _config.PatchSize);
            sourceUnpacked.Dispose();

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

    /// <summary>Drops the first <paramref name="drop"/> sequence positions from a <c>[1, seq, hidden]</c>
    /// F32 hidden-state tensor, returning <c>[1, seq-drop, hidden]</c>. Used to discard the system+user-header
    /// template prefix from Qwen-Image text conditioning (the kept tail = prompt content + assistant suffix,
    /// matching diffusers' <c>split_hidden_states[drop_idx:]</c>). No-op clone guard if drop is out of range.</summary>
    private static Tensor DropPrefixHiddenStates(Tensor hidden, int drop)
    {
        long batch = hidden.Shape[0];
        long seq = hidden.Shape[1];
        long hiddenDim = hidden.Shape[2];
        long elem = hidden.DType.SizeInBytes;
        long rowBytes = hiddenDim * elem;
        // Guard: if the drop would empty the sequence (or batch isn't 1), copy the whole tensor unchanged
        // rather than drop — the caller always disposes the input and adopts our return value.
        long effectiveDrop = (drop <= 0 || drop >= seq || batch != 1) ? 0 : drop;
        long newSeq = seq - effectiveDrop;
        Tensor result = new Tensor(new TensorShape(batch, newSeq, hiddenDim), hidden.DType);
        byte* src = (byte*)hidden.DataPointer;
        byte* dst = (byte*)result.DataPointer;
        Buffer.MemoryCopy(src + effectiveDrop * rowBytes, dst, newSeq * rowBytes, newSeq * rowBytes);
        return result;
    }

    /// <summary>Packs a latent tensor from <c>[B, C, H, W]</c> to <c>[B, (H/p)*(W/p), p² * C]</c> by interleaving p×p spatial blocks into the feature dim. Diffusers uses <c>view → permute(0,2,4,1,3,5) → reshape</c>; we replicate the resulting layout where each p×p patch contributes channels C × p² to the token, ordered <c>(c, py, px)</c> so the inverse <see cref="UnpackLatent"/> is exact.</summary>
    private static Tensor PackLatent(Tensor latent, int h, int w, int channels, int patchSize)
    {
        int batch = (int)latent.Shape[0];
        int hPacked = h / patchSize;
        int wPacked = w / patchSize;
        int patchDim = channels * patchSize * patchSize;
        int seqLen = hPacked * wPacked;

        TensorShape packedShape = new TensorShape(batch, seqLen, patchDim);
        Tensor packed = new Tensor(packedShape, DType.F32);

        float* inPtr = (float*)latent.DataPointer;
        float* outPtr = (float*)packed.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int ph = 0; ph < hPacked; ph++)
            {
                for (int pw = 0; pw < wPacked; pw++)
                {
                    int seqIdx = ph * wPacked + pw;
                    long outBase = ((long)b * seqLen + seqIdx) * patchDim;

                    for (int c = 0; c < channels; c++)
                    {
                        long inChannelBase = ((long)b * channels + c) * h * w;
                        long patchBase = outBase + (long)c * patchSize * patchSize;

                        for (int py = 0; py < patchSize; py++)
                        {
                            for (int px = 0; px < patchSize; px++)
                            {
                                int row = ph * patchSize + py;
                                int col = pw * patchSize + px;
                                outPtr[patchBase + py * patchSize + px] = inPtr[inChannelBase + (long)row * w + col];
                            }
                        }
                    }
                }
            }
        }

        return packed;
    }

    /// <summary>Inverse of <see cref="PackLatent"/>.</summary>
    private static Tensor UnpackLatent(Tensor packed, int h, int w, int channels, int patchSize)
    {
        int batch = (int)packed.Shape[0];
        int hPacked = h / patchSize;
        int wPacked = w / patchSize;
        int patchDim = channels * patchSize * patchSize;
        int seqLen = hPacked * wPacked;

        TensorShape unpackedShape = new TensorShape(batch, channels, h, w);
        Tensor unpacked = new Tensor(unpackedShape, DType.F32);

        float* inPtr = (float*)packed.DataPointer;
        float* outPtr = (float*)unpacked.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int ph = 0; ph < hPacked; ph++)
            {
                for (int pw = 0; pw < wPacked; pw++)
                {
                    int seqIdx = ph * wPacked + pw;
                    long inBase = ((long)b * seqLen + seqIdx) * patchDim;

                    for (int c = 0; c < channels; c++)
                    {
                        long outChannelBase = ((long)b * channels + c) * h * w;
                        long patchBase = inBase + (long)c * patchSize * patchSize;

                        for (int py = 0; py < patchSize; py++)
                        {
                            for (int px = 0; px < patchSize; px++)
                            {
                                int row = ph * patchSize + py;
                                int col = pw * patchSize + px;
                                outPtr[outChannelBase + (long)row * w + col] = inPtr[patchBase + py * patchSize + px];
                            }
                        }
                    }
                }
            }
        }

        return unpacked;
    }

    /// <summary>Applies the diffusers-style VAE input transform: <c>(latent / scaling_factor) + shift_factor</c>. The decoder expects this de-normalized input; the scaling and shift constants come from the model card.</summary>
    private static void ApplyVaeShiftScale(Tensor output, Tensor input, float shift, float scale)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        long count = output.ElementCount;
        float invScale = 1.0f / scale;
        for (long i = 0; i < count; i++)
            outPtr[i] = inPtr[i] * invScale + shift;
    }
}
