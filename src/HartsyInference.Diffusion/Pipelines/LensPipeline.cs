using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>Microsoft Lens text-to-image pipeline (`microsoft/Lens` RL-tuned, `microsoft/Lens-Turbo` 4-step distilled, `microsoft/Lens-Base` SFT-only). Like <see cref="Kandinsky5Pipeline"/> this pipeline only handles the diffusion side — the GPT-OSS MoE text encoder is not yet implemented as a HartsyInference text encoder. Callers must pre-compute the four multi-layer hidden-state captures (at layers 5/11/17/23 of GPT-OSS, each <c>[B, S_txt, 2880]</c>) plus their negative-prompt counterparts; this matches diffusers' <c>LensPipeline.__call__(prompt_embeds=, negative_prompt_embeds=, ...)</c> escape hatch. Once a GPT-OSS MoE encoder lands in HartsyInference the pipeline can grow a tokenizer-fed overload without touching the denoising path.
///
/// <para><b>Distinguishing details vs Flux/SD3/Qwen-Image pipelines:</b></para>
/// <list type="bullet">
/// <item><b>Empirical-mu shift schedule.</b> <c>mu = compute_empirical_mu(seq_len, num_steps)</c> per upstream's piecewise-linear calibration constants (a1/b1/a2/b2). Sigmas are <c>linspace(1.0, 1.0/N, N)</c> with a 0 appended at the end (matches diffusers' <c>FlowMatchEulerDiscreteScheduler.set_timesteps(sigmas=..., mu=...)</c>). The <c>time_shift(mu, 1, sigma) = exp(mu) / (exp(mu) + (1/sigma - 1))</c> form is algebraically identical to the existing C# scheduler's <c>shift * sigma / (1 + (shift - 1) * sigma)</c> when <c>shift = exp(mu)</c> — so we set the scheduler's shift parameter directly.</item>
/// <item><b>Norm-rescaled CFG.</b> After computing <c>comb = uncond + cfg * (cond - uncond)</c>, rescale each token so its L2 norm matches the conditional branch's: <c>noise_pred = comb * (||cond|| / max(||comb||, 1e-12))</c>. Important even at cfg=1.0 (no-op in expectation, but the multiplication still runs and we mirror upstream exactly).</item>
/// <item><b>Packed-token latent throughout.</b> Shape <c>[B, latent_h * latent_w, 128]</c> never leaves packed form until VAE decode. <c>latent_h = H/16</c>, <c>latent_w = W/16</c> — the pipeline's <c>vae_scale_factor=16</c> bakes in Flux.2 VAE's 8× downsample × the pipeline-side 2×2 patchify.</item>
/// <item><b>Reuses the Flux.2 VAE</b> verbatim (<see cref="VaeDecoder"/> + BN un-norm at the pipeline boundary, same pattern as <see cref="Flux2Pipeline"/>).</item>
/// </list></summary>
public sealed unsafe class LensPipeline : DiffusionPipelineBase
{
    private readonly LensTransformer _transformer;
    private readonly LensGptOssEncoder? _textEncoder;
    private readonly VaeDecoder _vaeDecoder;
    private readonly LensConfig _config;
    private readonly Tensor _bnMean;
    private readonly Tensor _bnVar;
    private readonly float _bnEps;

    /// <summary>Creates a Lens pipeline without an attached text encoder. Use this when callers will
    /// supply pre-computed multi-layer text embeddings via <see cref="GenerateFromEmbeddings"/>.</summary>
    public LensPipeline(IBackend backend, LensTransformer transformer, VaeDecoder vaeDecoder,
        Tensor bnMean, Tensor bnVar, LensConfig config, float bnEps = 1e-5f)
        : this(backend, transformer, textEncoder: null, vaeDecoder, bnMean, bnVar, config, bnEps)
    {
    }

    /// <summary>Creates a Lens pipeline with an attached <see cref="LensGptOssEncoder"/>. Required
    /// for the tokenizer-fed <see cref="GenerateFromTokens"/> overload.</summary>
    public LensPipeline(IBackend backend, LensTransformer transformer,
        LensGptOssEncoder? textEncoder, VaeDecoder vaeDecoder,
        Tensor bnMean, Tensor bnVar, LensConfig config, float bnEps = 1e-5f)
        : base(backend)
    {
        _transformer = transformer;
        _textEncoder = textEncoder;
        _vaeDecoder = vaeDecoder;
        _bnMean = bnMean;
        _bnVar = bnVar;
        _bnEps = bnEps;
        _config = config;
    }

    /// <summary>Generates an image from chat-templated token id arrays. Runs the
    /// <see cref="LensGptOssEncoder"/> on the positive prompt (and on the negative when
    /// <c>request.CfgScale &gt; 1</c>), then delegates to <see cref="GenerateFromEmbeddings"/>. Frees
    /// the encoder's GPU weights after the encode pass so the DiT has maximum VRAM available — same
    /// pattern as <see cref="Flux2Pipeline"/> / <see cref="Sd3Pipeline"/>.</summary>
    /// <param name="positiveTokenIds">Chat-templated tokens for the positive prompt (rendered by
    /// <see cref="HartsyInference.Tokenizers.GptOssTokenizer.BuildChatInputs"/>).</param>
    /// <param name="negativeTokenIds">Chat-templated tokens for the negative prompt. Required when
    /// <c>request.CfgScale &gt; 1.0</c>; pass <c>null</c> when CFG is disabled (Lens-Turbo).</param>
    /// <param name="request">Generation parameters.</param>
    /// <param name="onProgress">Optional per-step progress callback.</param>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromTokens(
        int[] positiveTokenIds,
        int[]? negativeTokenIds,
        TextToImageRequest request,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        if (_textEncoder is null)
            throw new InvalidOperationException(
                "GenerateFromTokens requires a LensGptOssEncoder; the pipeline was constructed without one. " +
                "Use the constructor that accepts a text encoder, or call GenerateFromEmbeddings with pre-computed captures.");

        float cfgScale = request.CfgScale ?? _config.DefaultCfgScale;
        bool useCfg = cfgScale > 1.0f;
        if (useCfg && negativeTokenIds is null)
            throw new ArgumentException("cfgScale > 1.0 requires negativeTokenIds.", nameof(negativeTokenIds));

        Stopwatch encSw = Stopwatch.StartNew();
        Logs.Info("Encoding text with GPT-OSS (multi-layer hidden states at [5,11,17,23])...");
        List<Tensor> positiveLayers = _textEncoder.EncodeForLens(Backend, positiveTokenIds);
        List<Tensor>? negativeLayers = null;
        if (useCfg) negativeLayers = _textEncoder.EncodeForLens(Backend, negativeTokenIds!);
        encSw.Stop();
        Logs.Info($"Text encoding done in {encSw.ElapsedMilliseconds}ms (S_txt+={positiveLayers[0].Shape[1]}, " +
                  $"S_txt-={negativeLayers?[0].Shape[1] ?? 0})");

        // Free encoder weights before the denoise loop so the DiT can claim the VRAM (no-op on
        // backends without a weight cache).
        Backend.Sync();
        Backend.FreeWeights(_textEncoder.EnumerateWeights());

        try
        {
            return GenerateFromEmbeddings(positiveLayers, negativeLayers, request, onProgress);
        }
        finally
        {
            for (int i = 0; i < positiveLayers.Count; i++) positiveLayers[i].Dispose();
            if (negativeLayers is not null)
                for (int i = 0; i < negativeLayers.Count; i++) negativeLayers[i].Dispose();
        }
    }

    /// <summary>Generates an image from pre-computed GPT-OSS multi-layer text embeddings. Four positive captures + four negative captures (same length per layer; positives and negatives may differ in S_txt — cond and uncond run as separate forwards at their natural lengths). Each capture is <c>[B, S_txt, 2880]</c> matching <see cref="LensConfig.EncoderHiddenDim"/>.</summary>
    /// <param name="positiveLayers">Positive-prompt hidden states captured at <see cref="LensConfig.SelectedEncoderLayers"/>. Must have exactly 4 entries.</param>
    /// <param name="negativeLayers">Negative-prompt hidden states (same layer order as <paramref name="positiveLayers"/>). Required when <c>request.CfgScale &gt; 1.0</c>; ignored otherwise.</param>
    /// <param name="request">Generation parameters. <c>request.CfgScale</c> falls back to <see cref="LensConfig.DefaultCfgScale"/> if unset. <c>request.Steps</c> falls back to <see cref="LensConfig.DefaultSteps"/>.</param>
    /// <param name="onProgress">Optional per-step progress callback.</param>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromEmbeddings(
        IReadOnlyList<Tensor> positiveLayers,
        IReadOnlyList<Tensor>? negativeLayers,
        TextToImageRequest request,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        ValidateEmbeddings(positiveLayers, negativeLayers);

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int steps = request.Steps ?? _config.DefaultSteps;
        float cfgScale = request.CfgScale ?? _config.DefaultCfgScale;
        bool useCfg = cfgScale > 1.0f;
        if (useCfg && negativeLayers is null)
            throw new ArgumentException("CfgScale > 1.0 requires negativeLayers.", nameof(negativeLayers));

        // Lens uses vae_scale_factor = 16 (Flux.2 VAE 8× + pipeline-side 2×2 patchify).
        // Round image dims to a multiple of 16 so latent grids tile cleanly.
        const int vaeScaleFactor = 16;
        int reqHeight = request.Height ?? GenerationDefaults.Generic.Height;
        int reqWidth = request.Width ?? GenerationDefaults.Generic.Width;
        int imgH = (reqHeight / vaeScaleFactor) * vaeScaleFactor;
        int imgW = (reqWidth / vaeScaleFactor) * vaeScaleFactor;
        int patH = imgH / vaeScaleFactor;
        int patW = imgW / vaeScaleFactor;
        int imgSeqLen = patH * patW;
        int latH = imgH / 8;
        int latW = imgW / 8;

        Logs.Info($"Lens: {imgW}x{imgH}, {steps} steps, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        // ── 1. Empirical-mu schedule ──────────────────────────────
        float mu = ComputeEmpiricalMu(imgSeqLen, steps);
        float shift = MathF.Exp(mu);
        Logs.Info($"Lens empirical_mu={mu:F4} → shift={shift:F4} (seq_len={imgSeqLen}, steps={steps})");
        FlowMatchEulerDiscreteScheduler scheduler = new FlowMatchEulerDiscreteScheduler(shift);
        scheduler.SetTimesteps(steps);

        // ── 2. Initial noise in packed token form ─────────────────
        // (No pos/neg length alignment: the reference right-pads + masks only because it batches
        // cond and uncond into one batch-of-2 forward. We run two separate forwards, each at its
        // prompt's natural S_txt — mathematically identical to the masked reference.)
        TensorShape packedShape = new TensorShape(1, imgSeqLen, _config.InChannels);
        Tensor packedLatent = SeedGenerator.CreateNoise(packedShape, seed);
        Logs.Verbose($"Initial latent: shape={packedLatent.Shape}, init_sigma={scheduler.InitialNoiseSigma:F4}");

        // ── 3. Denoising loop ─────────────────────────────────────
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;
        for (int i = 0; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float sigma = timesteps[i] / 1000.0f;

            Tensor noisePred;
            if (useCfg)
            {
                Tensor condPred = _transformer.Forward(Backend, packedLatent, positiveLayers, sigma, patH, patW);
                Tensor uncondPred = _transformer.Forward(Backend, packedLatent, negativeLayers!, sigma, patH, patW);
                noisePred = ApplyNormRescaledCfg(condPred, uncondPred, cfgScale);
                condPred.Dispose();
                uncondPred.Dispose();
            }
            else
            {
                noisePred = _transformer.Forward(Backend, packedLatent, positiveLayers, sigma, patH, patW);
            }

            TensorShape stepShape = new TensorShape(1, imgSeqLen, _config.InChannels);
            Tensor newLatent = new Tensor(stepShape, DType.F32);
            scheduler.Step(newLatent, noisePred, packedLatent, i);
            noisePred.Dispose();
            packedLatent.Dispose();
            packedLatent = newLatent;

            stepSw.Stop();
            Logs.Debug($"Step {i + 1}/{steps} (sigma={sigma:F4}) done in {stepSw.ElapsedMilliseconds}ms");
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds)
            {
                LatentArch = LatentArchitecture.Flux2,
            });
        }

        // Free transformer (matches PHASE_3_DEVIATIONS #18: eviction before VAE).
        Backend.Sync();
        Backend.FreeWeights(_transformer.EnumerateWeights());

        // ── 5. Unpack [B, S, 128] → [B, 128, patH, patW] ──────────
        Tensor unpackedPatched = UnpackLatent(packedLatent, patH, patW);
        packedLatent.Dispose();

        // ── 6. BN un-normalize on the 128-channel patchified latent (per-channel mean/var)──
        Tensor unBn = ApplyBnUnNormalize(unpackedPatched, _bnMean, _bnVar, _bnEps);
        unpackedPatched.Dispose();

        // ── 7. 2×2 unpatchify: [B, 128, patH, patW] → [B, 32, latH, latW] ──
        Tensor latent32 = UnpatchifyLatent(unBn, _config.OutChannels, _config.PatchSize);
        unBn.Dispose();

        // ── 8. VAE decode ─────────────────────────────────────────
        Logs.Verbose("Decoding latents (tiled F32 path)...");
        Stopwatch vaeSw = Stopwatch.StartNew();
        Tensor image = _vaeDecoder.DecodeTiled(Backend, latent32);
        latent32.Dispose();
        vaeSw.Stop();
        Logs.Verbose($"VAE decode done in {vaeSw.ElapsedMilliseconds}ms");

        byte[] rgbData = ImagePostProcessor.TensorToRgbBytes(image);
        image.Dispose();

        sw.Stop();
        Logs.Info($"Lens generation complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        return (rgbData, imgW, imgH, seed);
    }

    /// <summary>Empirical-mu schedule from <c>lens/pipeline.py:compute_empirical_mu</c>. Piecewise linear in image_seq_len for short sequences (≤4300), single linear regression for long ones.</summary>
    private static float ComputeEmpiricalMu(int imageSeqLen, int numSteps)
    {
        const double a1 = 8.73809524e-05, b1 = 1.89833333;
        const double a2 = 0.00016927, b2 = 0.45666666;
        if (imageSeqLen > 4300)
            return (float)(a2 * imageSeqLen + b2);

        double m200 = a2 * imageSeqLen + b2;
        double m10 = a1 * imageSeqLen + b1;
        double a = (m200 - m10) / 190.0;
        double b = m200 - 200.0 * a;
        return (float)(a * numSteps + b);
    }

    /// <summary>Applies upstream's norm-rescaled CFG: <c>comb = uncond + cfg * (cond - uncond)</c>, then per-token L2-norm rescale <c>noise = comb * (||cond|| / max(||comb||, 1e-12))</c>. Operates on packed-token tensors <c>[B, S, C]</c>; the norm is over the channel dim.</summary>
    private static Tensor ApplyNormRescaledCfg(Tensor cond, Tensor uncond, float cfgScale)
    {
        if (cond.Shape != uncond.Shape)
            throw new ArgumentException($"cond shape {cond.Shape} must match uncond shape {uncond.Shape}.");
        int batch = (int)cond.Shape[0];
        int seqLen = (int)cond.Shape[1];
        int channels = (int)cond.Shape[2];

        Tensor output = new Tensor(cond.Shape, DType.F32);
        float* condPtr = (float*)cond.DataPointer;
        float* uncondPtr = (float*)uncond.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int tokOffset = (b * seqLen + s) * channels;
                float condNormSq = 0f;
                float combNormSq = 0f;
                for (int c = 0; c < channels; c++)
                {
                    float condVal = condPtr[tokOffset + c];
                    float uncondVal = uncondPtr[tokOffset + c];
                    float combVal = uncondVal + cfgScale * (condVal - uncondVal);
                    outPtr[tokOffset + c] = combVal;
                    condNormSq += condVal * condVal;
                    combNormSq += combVal * combVal;
                }
                float condNorm = MathF.Sqrt(condNormSq);
                float combNorm = MathF.Sqrt(combNormSq);
                float scale = combNorm > 0f ? condNorm / MathF.Max(combNorm, 1e-12f) : 1.0f;
                for (int c = 0; c < channels; c++)
                    outPtr[tokOffset + c] *= scale;
            }
        }
        return output;
    }

    private void ValidateEmbeddings(IReadOnlyList<Tensor> positives, IReadOnlyList<Tensor>? negatives)
    {
        int expected = _config.SelectedEncoderLayers.Length;
        if (positives.Count != expected)
            throw new ArgumentException($"positiveLayers must have {expected} entries; got {positives.Count}.", nameof(positives));
        long s0 = positives[0].Shape[1];
        for (int i = 0; i < expected; i++)
        {
            if (positives[i].Shape.Rank != 3 || positives[i].Shape[2] != _config.EncoderHiddenDim)
                throw new ArgumentException($"positiveLayers[{i}] shape {positives[i].Shape} does not match expected [B, S, {_config.EncoderHiddenDim}].");
            if (positives[i].Shape[1] != s0)
                throw new ArgumentException($"positiveLayers[{i}] seq_len {positives[i].Shape[1]} disagrees with positiveLayers[0] seq_len {s0}.");
        }
        if (negatives is null) return;
        if (negatives.Count != expected)
            throw new ArgumentException($"negativeLayers must have {expected} entries; got {negatives.Count}.", nameof(negatives));
        long n0 = negatives[0].Shape[1];
        for (int i = 0; i < expected; i++)
        {
            if (negatives[i].Shape.Rank != 3 || negatives[i].Shape[2] != _config.EncoderHiddenDim)
                throw new ArgumentException($"negativeLayers[{i}] shape {negatives[i].Shape} does not match expected [B, S, {_config.EncoderHiddenDim}].");
            if (negatives[i].Shape[1] != n0)
                throw new ArgumentException($"negativeLayers[{i}] seq_len {negatives[i].Shape[1]} disagrees with negativeLayers[0] seq_len {n0}.");
        }
    }

    /// <summary>Unpacks [B, S, C] to [B, C, H, W] (no spatial reshuffle — packed dim is already row-major H*W).</summary>
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

    /// <summary>BN un-normalize on the 128-channel patchified latent: <c>z = z * sqrt(var + eps) + mean</c> per channel. Mean/var must be shape <c>[128]</c>. Matches Flux.2 pipeline's <c>ApplyBnUnNormalize</c> exactly.</summary>
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

    /// <summary>2×2 spatial unpatchify: <c>[B, C*4, H, W] → [B, C, H*2, W*2]</c>. Matches Flux.2 pipeline's <c>UnpatchifyLatent</c>.</summary>
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
                                int srcChan = c * (patchSize * patchSize) + py * patchSize + px;
                                int srcIdx = ((b * inChannels + srcChan) * h + y) * w + x;
                                int oy = y * patchSize + py;
                                int ox = x * patchSize + px;
                                int dstIdx = ((b * outChannels + c) * outH + oy) * outW + ox;
                                outPtr[dstIdx] = inPtr[srcIdx];
                            }
                        }
                    }
                }
            }
        }
        return output;
    }
}
