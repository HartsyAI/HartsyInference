using System.Diagnostics;
using SharpInference.Core.Backends;
using SharpInference.Core.Logging;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.TextEncoders;
using SharpInference.Diffusion.Models.Vae;
using SharpInference.Diffusion.Requests;
using SharpInference.Diffusion.Schedulers;
using SharpInference.Diffusion.Utilities;

namespace SharpInference.Diffusion.Pipelines;

/// <summary>
/// Flux.2 text-to-image pipeline (Klein 4B / Klein 9B / Dev). Orchestrates Qwen3-4B (Klein) or
/// Mistral-Small-3 (Dev) text encoding → <see cref="Flux2Transformer"/> denoising with flow
/// matching → BN-style latent un-normalization → 2×2 unpatchify → VAE decode → RGB image.
/// <para>Differences from Flux.1: no CLIP-L pooled embedding, no T5; multi-layer text-encoder
/// hidden state concat; 32-channel VAE latent (16× effective downscale once 2×2 patchify is
/// applied); BatchNorm-style latent normalization (<c>bn.running_mean/var</c>) applied at the
/// pipeline boundary, not inside the VAE module.</para>
/// </summary>
public sealed unsafe class Flux2Pipeline : IDisposable
{
    private readonly IBackend _backend;
    private readonly LlamaStyleEncoder _textEncoder;
    private readonly Flux2Transformer _transformer;
    private readonly VaeDecoder _vaeDecoder;
    private readonly VaeEncoder? _vaeEncoder;
    private readonly Flux2Config _config;
    private readonly Tensor _bnMean;     // [128] — running_mean of the patchified-latent BatchNorm
    private readonly Tensor _bnVar;      // [128] — running_var
    private readonly float _bnEps;
    private readonly int[] _hiddenLayers;
    private int _disposed;

    /// <summary>Creates a Flux.2 pipeline. Img2img is unavailable; use the overload accepting a <see cref="VaeEncoder"/> to enable it.</summary>
    public Flux2Pipeline(IBackend backend, LlamaStyleEncoder textEncoder,
        Flux2Transformer transformer, VaeDecoder vaeDecoder,
        Tensor bnMean, Tensor bnVar, Flux2Config config,
        int[]? hiddenLayers = null, float bnEps = 1e-5f)
        : this(backend, textEncoder, transformer, vaeDecoder, vaeEncoder: null, bnMean, bnVar, config, hiddenLayers, bnEps)
    {
    }

    /// <summary>Creates a Flux.2 pipeline with both VAE halves loaded. Required for img2img.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="textEncoder">Llama-style encoder configured for the variant (Qwen3-4B for Klein).</param>
    /// <param name="transformer">Pre-loaded <see cref="Flux2Transformer"/>.</param>
    /// <param name="vaeDecoder">VAE decoder loaded with <see cref="VaeConfig.Flux2"/>.</param>
    /// <param name="vaeEncoder">Optional VAE encoder loaded with <see cref="VaeConfig.Flux2"/>. Required for img2img.</param>
    /// <param name="bnMean">BN <c>running_mean</c> tensor of shape <c>[128]</c> (= <c>32 latent channels × 4 patch</c>).</param>
    /// <param name="bnVar">BN <c>running_var</c> tensor of shape <c>[128]</c>.</param>
    /// <param name="config">Flux.2 variant config.</param>
    /// <param name="hiddenLayers">Text-encoder hidden-state layer indices to concatenate (Klein default: <c>[9, 18, 27]</c>). The encoder's per-layer outputs are concatenated along the hidden dim and fed into <c>context_embedder</c>.</param>
    /// <param name="bnEps">BatchNorm epsilon (matches <c>vae.config.batch_norm_eps</c>; default 1e-5).</param>
    public Flux2Pipeline(IBackend backend, LlamaStyleEncoder textEncoder,
        Flux2Transformer transformer, VaeDecoder vaeDecoder, VaeEncoder? vaeEncoder,
        Tensor bnMean, Tensor bnVar, Flux2Config config,
        int[]? hiddenLayers = null, float bnEps = 1e-5f)
    {
        _backend = backend;
        _textEncoder = textEncoder;
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _vaeEncoder = vaeEncoder;
        _bnMean = bnMean;
        _bnVar = bnVar;
        _bnEps = bnEps;
        _config = config;
        _hiddenLayers = hiddenLayers ?? [9, 18, 27];
    }

    /// <summary>Generates an image from pre-tokenized prompt input. Handles both text-to-image and image-to-image via the runtime type of <paramref name="request"/>:
    /// <list type="bullet">
    /// <item>Plain <see cref="TextToImageRequest"/> → text-to-image.</item>
    /// <item><see cref="ImageToImageRequest"/> → img2img. Source goes VAE-encode (32ch latent) → 2×2 patchify → BN-normalize → pack → AddNoise at sigma[startStep]. Requires a <see cref="VaeEncoder"/>.</item>
    /// </list>
    /// </summary>
    /// <param name="promptTokenIds">Padded token IDs <c>[seqLen]</c>.</param>
    /// <param name="request">Generation parameters. Pass an <see cref="ImageToImageRequest"/> for img2img.</param>
    /// <param name="guidanceScale">Guidance scale for Dev (embedded via MLP). Ignored when <see cref="Flux2Config.GuidanceEmbed"/> is false (Klein). Default 3.5.</param>
    /// <param name="onProgress">Optional progress callback.</param>
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
        int steps = request.Steps;

        // Round image dims down to multiple of 16 (VAE 8× × 2 patch). Latent dims are then
        // image_h/8 (latent space, 32 channels) and image_h/16 (after 2×2 patchify, 128 channels).
        int imgH = (request.Height / _config.VaeDownscaleFactor) * _config.VaeDownscaleFactor;
        int imgW = (request.Width / _config.VaeDownscaleFactor) * _config.VaeDownscaleFactor;
        int latH = imgH / 8;            // VAE-latent spatial (32 ch)
        int latW = imgW / 8;
        int patH = imgH / 16;           // Patchified-latent spatial (128 ch) — what the transformer sees
        int patW = imgW / 16;
        int imgSeqLen = patH * patW;

        // Img2img: validate source shape + handle strength=0 short-circuit BEFORE any model work.
        // Note source must match the rounded imgH/imgW (not request.Height/Width) since Flux.2 rounds.
        int startStep = 0;
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

            float strength = Math.Clamp(img2img.Strength, 0f, 1f);
            int initTimesteps = (int)MathF.Round(steps * strength);
            startStep = Math.Max(steps - initTimesteps, 0);

            if (initTimesteps == 0)
            {
                Logs.Info("Strength=0; passing source through unchanged");
                return (ImagePostProcessor.TensorToRgbBytes(src), imgW, imgH, seed);
            }
        }

        string variant = _config.TextEncoderType == Flux2TextEncoderType.Mistral ? "Dev" : "Klein";
        string opMode = isImg2Img ? $"img2img (start={startStep}/{steps})" : "txt2img";
        Logs.Info($"Flux.2 ({variant}) {opMode}: {imgW}x{imgH}, {steps} steps, guidance={guidanceScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        // ── 1. Text encoder forward ───────────────────────────────────
        Logs.Info("Encoding text with Qwen3 (multi-layer hidden states)...");
        int[][] batchedTokenIds = [promptTokenIds];
        Tensor textEmbeddings = _textEncoder.EncodeMultiLayer(_backend, batchedTokenIds, _hiddenLayers);
        int txtSeqLen = (int)textEmbeddings.Shape[1];
        Logs.Info($"Text encoding done in {sw.ElapsedMilliseconds}ms (seqLen={txtSeqLen}, hidden={textEmbeddings.Shape[2]})");
        LogTensorStats("text embeddings", textEmbeddings);

        // ── 2. Set up dynamic-shift flow-match scheduler ──────────────
        TensorShape noiseShape = new TensorShape(1, _config.InChannels, patH, patW);
        TensorShape packedShape = new TensorShape(1, imgSeqLen, _config.InChannels);
        FlowMatchEulerDiscreteScheduler scheduler =
            FlowMatchEulerDiscreteScheduler.CreateWithDynamicShift(imgSeqLen);
        scheduler.SetTimesteps(steps);

        // ── 3. Build initial packed latent (t2i: noise * initSigma; img2img: encode + patchify + BN-norm + pack + AddNoise) ──
        Tensor packedLatent = BuildInitialPackedLatent(request, scheduler, noiseShape, packedShape, latH, latW, patH, patW, seed, startStep);

        // ── 4. Denoising loop (from startStep onward) ──
        Logs.Info("Starting Flux.2 denoising loop...");
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;
        for (int i = startStep; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float sigma = timesteps[i] / 1000.0f;

            Tensor velocityPred = _transformer.Forward(
                _backend, packedLatent, textEmbeddings, sigma, guidanceScale, patH, patW);

            TensorShape packedStepShape = new TensorShape(1, imgSeqLen, _config.InChannels);
            Tensor newLatent = new Tensor(packedStepShape, DType.F32);
            scheduler.Step(newLatent, velocityPred, packedLatent, i);
            velocityPred.Dispose();
            packedLatent.Dispose();
            packedLatent = newLatent;

            stepSw.Stop();
            Logs.Debug($"Step {i + 1}/{steps} (sigma={sigma:F4}) done in {stepSw.ElapsedMilliseconds}ms");
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds));
        }

        textEmbeddings.Dispose();

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
        Tensor image = _vaeDecoder.DecodeTiled(_backend, latent32);
        latent32.Dispose();
        vaeSw.Stop();
        Logs.Verbose($"VAE decode done in {vaeSw.ElapsedMilliseconds}ms");

        // ── 9. RGB conversion ─────────────────────────────────────────
        byte[] rgbData = ImagePostProcessor.TensorToRgbBytes(image);
        image.Dispose();

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

    /// <summary>
    /// Applies BatchNorm un-normalization on the patchified latent: <c>z = z * std + mean</c>.
    /// Operates per-channel (mean/var have shape <c>[128]</c>; broadcast across batch and spatial dims).
    /// Mirrors the diffusers reference: <c>latents = latents * sqrt(running_var + eps) + running_mean</c>.
    /// </summary>
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

    /// <summary>
    /// 2×2 spatial unpatchify: <c>[B, C*4, H, W] → [B, C, H*2, W*2]</c>. Implements the diffusers
    /// <c>_unpatchify_latents</c>: reshape to <c>[B, C, 2, 2, H, W]</c>, permute to
    /// <c>[B, C, H, 2, W, 2]</c>, reshape to <c>[B, C, H*2, W*2]</c>. Equivalent to nn.PixelShuffle(2)
    /// applied per-channel-group.
    /// </summary>
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

    /// <summary>Builds the initial packed latent for Flux.2 denoising. T2I: fresh noise scaled by initSigma. Img2img: source goes VAE-encode → 2×2 patchify → BN-normalize → pack → AddNoise at sigma[startStep].</summary>
    private Tensor BuildInitialPackedLatent(
        TextToImageRequest request,
        FlowMatchEulerDiscreteScheduler scheduler,
        TensorShape noiseShape,
        TensorShape packedShape,
        int latH, int latW, int patH, int patW,
        int seed, int startStep)
    {
        Tensor packedNoise = PackLatent(SeedGenerator.CreateNoise(noiseShape, seed));

        if (request is ImageToImageRequest img2img)
        {
            Stopwatch vaeEncSw = Stopwatch.StartNew();
            // VaeConfig.Flux2 has ScalingFactor=1.0 + ShiftFactor=null, so encoder output is the raw mean (no scaling).
            Tensor sourceLatent32 = _vaeEncoder!.Encode(_backend, img2img.SourceImage);  // [1, 32, latH, latW]
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
            sourcePacked.Dispose();
            packedNoise.Dispose();
            return result;
        }

        // T2I path: scale packed noise by initSigma.
        float initSigma = scheduler.InitialNoiseSigma;
        if (MathF.Abs(initSigma - 1.0f) > 1e-6f)
        {
            Tensor scaled = new Tensor(packedShape, DType.F32);
            _backend.Scale(scaled, packedNoise, initSigma);
            packedNoise.Dispose();
            return scaled;
        }
        return packedNoise;
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

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
    }

    public void Dispose()
    {
        Volatile.Write(ref _disposed, 1);
    }
}
