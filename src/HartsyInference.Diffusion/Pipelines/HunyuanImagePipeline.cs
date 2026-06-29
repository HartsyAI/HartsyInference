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

/// <summary>Hunyuan Image 2.1 text-to-image pipeline. The transformer takes per-token text embeddings
/// at 3584-dim (Qwen2.5-VL-7B hidden) plus an optional 1472-dim secondary stream (ByT5 in the upstream
/// model). Conditioning is driven by the timestep + optional distilled-guidance embedding — there is no
/// pooled-conditioning path, so the CLIP encoder constructor parameter is accepted for API compatibility
/// but unused. The pipeline patchifies the latent before the transformer call (<c>patch_size</c> from
/// config) and unpatchifies after, then VAE-decodes through the 32-channel Hunyuan VAE.
///
/// <para>Two text-encode paths: the primary <see cref="HunyuanImageQwenTextEncoder"/> path matching the
/// upstream model (template → pad 1034 → <c>hidden_states[-3]</c> → drop the 34-token prefix), and a
/// legacy CLIP-L + T5-XXL constructor kept for repackaged checkpoints that bundle those encoders.
/// The ByT5 glyph branch (1472-dim secondary stream for quoted text) is not implemented yet — the
/// transformer receives <c>encoderHidden2 = null</c>, equivalent to diffusers' zero-tensor fallback for
/// prompts without quoted text. TODO (validation-gated): add a ByT5-small encoder for glyph prompts.</para></summary>
public sealed unsafe class HunyuanImagePipeline : DiffusionPipelineBase
{
    private readonly ClipTextEncoder? _clipEncoder;
    private readonly T5TextEncoder? _t5Encoder;
    private readonly HunyuanImageQwenTextEncoder? _qwenEncoder;
    private readonly HunyuanImageTransformer _transformer;
    private readonly VaeDecoder _vaeDecoder;
    private readonly HunyuanImageConfig _config;

    /// <summary>Creates a Hunyuan Image pipeline with the primary Qwen2.5-VL-7B text encoder (matches <c>tencent/HunyuanImage-2.1</c>).</summary>
    public HunyuanImagePipeline(IBackend backend, HunyuanImageQwenTextEncoder qwenEncoder,
        HunyuanImageTransformer transformer, VaeDecoder vaeDecoder, HunyuanImageConfig config)
        : base(backend)
    {
        _qwenEncoder = qwenEncoder ?? throw new ArgumentNullException(nameof(qwenEncoder));
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _config = config;
    }

    /// <summary>Legacy/fallback constructor using CLIP-L + T5-XXL (repackaged checkpoints).
    /// <paramref name="clipEncoder"/> is accepted for API compatibility but not consumed — Hunyuan's
    /// pooled-conditioning path is replaced by the timestep + distilled-guidance embedding inside the
    /// transformer.</summary>
    public HunyuanImagePipeline(IBackend backend, ClipTextEncoder? clipEncoder, T5TextEncoder t5Encoder,
        HunyuanImageTransformer transformer, VaeDecoder vaeDecoder, HunyuanImageConfig config)
        : base(backend)
    {
        _clipEncoder = clipEncoder;
        _t5Encoder = t5Encoder ?? throw new ArgumentNullException(nameof(t5Encoder));
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _config = config;
    }

    /// <summary>Generates an image via the primary Qwen2.5-VL path. Token ids come from
    /// <c>Qwen2Tokenizer.EncodeChat(prompt, systemPrompt: HunyuanImageQwenTextEncoder.SystemPrompt,
    /// addGenerationPrompt: false)</c> padded to <see cref="HunyuanImageQwenTextEncoder.PaddedLength"/>
    /// with matching attention masks. Negative inputs may be null when CFG is off (cfg ≤ 1).</summary>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromTokens(
        int[] promptTokenIdsQwen,
        int[] promptAttentionMaskQwen,
        int[]? negativePromptTokenIdsQwen,
        int[]? negativeAttentionMaskQwen,
        TextToImageRequest request,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        if (_qwenEncoder is null)
            throw new InvalidOperationException(
                "This pipeline was constructed with the legacy CLIP+T5 encoders — use the CLIP/T5 GenerateFromTokens overload.");

        bool useCfg = request.CfgScale > 1.0f;
        if (useCfg && (negativePromptTokenIdsQwen is null || negativeAttentionMaskQwen is null))
            throw new ArgumentException("Negative prompt tokens + mask are required when CfgScale > 1.", nameof(negativePromptTokenIdsQwen));

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        Logs.Info("Encoding text with Qwen2.5-VL-7B (primary 3584-dim per-token)...");
        Backend.PreloadWeights(_qwenEncoder.EnumerateWeights());
        Tensor cond = _qwenEncoder.Encode(Backend, promptTokenIdsQwen, promptAttentionMaskQwen);
        Tensor? uncond = useCfg
            ? _qwenEncoder.Encode(Backend, negativePromptTokenIdsQwen!, negativeAttentionMaskQwen!)
            : null;
        Backend.Sync();
        Backend.FreeWeights(_qwenEncoder.EnumerateWeights());

        return Denoise(cond, uncond, request, seed, onProgress);
    }

    /// <summary>Generates an image from pre-tokenized input via the legacy CLIP+T5 path. CLIP token args
    /// are unused (kept for signature stability); the T5 encoder produces per-token context fed to the
    /// transformer.</summary>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromTokens(
        int[] promptTokenIdsClip,
        int[] negativePromptTokenIdsClip,
        int promptEosPositionClip,
        int negativeEosPositionClip,
        int[] promptTokenIdsT5,
        int[] negativePromptTokenIdsT5,
        int[]? promptAttentionMaskT5,
        int[]? negativeAttentionMaskT5,
        TextToImageRequest request,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        if (_t5Encoder is null)
            throw new InvalidOperationException(
                "This pipeline was constructed with the Qwen2.5-VL encoder — use the Qwen GenerateFromTokens overload.");

        bool useCfg = request.CfgScale > 1.0f;
        int seed = request.Seed ?? SeedGenerator.RandomSeed();

        Logs.Info("Encoding text with T5-XXL (legacy per-token context path)...");
        int[][] condTokenBatch = [promptTokenIdsT5];
        int[][]? condMaskBatch = promptAttentionMaskT5 is null ? null : [promptAttentionMaskT5];
        Tensor cond = _t5Encoder.Encode(Backend, condTokenBatch, condMaskBatch);

        Tensor? uncond = null;
        if (useCfg)
        {
            int[][] uncondTokenBatch = [negativePromptTokenIdsT5];
            int[][]? uncondMaskBatch = negativeAttentionMaskT5 is null ? null : [negativeAttentionMaskT5];
            uncond = _t5Encoder.Encode(Backend, uncondTokenBatch, uncondMaskBatch);
        }

        return Denoise(cond, uncond, request, seed, onProgress);
    }

    /// <summary>Shared denoise + VAE-decode body. Takes ownership of <paramref name="condText"/> / <paramref name="uncondText"/> and disposes them.</summary>
    private (byte[] rgbData, int width, int height, int seed) Denoise(Tensor condText, Tensor? uncondText,
        TextToImageRequest request, int seed, Action<GenerationProgress>? onProgress)
    {
        int width = request.Width;
        int height = request.Height;
        const int VaeDownscale = 32;
        if (width % VaeDownscale != 0 || height % VaeDownscale != 0)
            throw new ArgumentException($"Hunyuan Image: width and height must be divisible by {VaeDownscale}.");
        int latentH = height / VaeDownscale;
        int latentW = width / VaeDownscale;
        int steps = request.Steps;
        float cfgScale = request.CfgScale;
        bool useCfg = cfgScale > 1.0f;
        int patch = _config.PatchSize;
        int inChannels = _config.InChannels;
        int hPacked = latentH / patch;
        int wPacked = latentW / patch;

        Logs.Info($"Hunyuan Image 2.1: {width}x{height}, {steps} steps, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        float distilledGuidance = _config.GuidanceEmbed ? cfgScale : 1.0f;

        TensorShape latentShape = new(1, inChannels, latentH, latentW);
        Tensor latent = SeedGenerator.CreateNoise(latentShape, seed);

        FlowMatchEulerDiscreteScheduler scheduler = new(_config.SamplingShift);
        scheduler.SetTimesteps(steps);
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;

        // Bulk-upload transformer weights before the denoise loop. Paired with FreeWeights at
        // the VAE handoff. No-op on backends without a weight cache.
        Backend.PreloadWeights(_transformer.EnumerateWeights());

        Logs.Info($"Starting denoise loop ({steps} steps, distilled guidance={distilledGuidance})...");
        for (int i = 0; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float t = timesteps[i];

            Tensor patched = PatchifyLatent(latent, inChannels, latentH, latentW, patch);

            Tensor velocity = _transformer.Forward(Backend, patched, condText, encoderHidden2: null,
                t, distilledGuidance, hPacked, wPacked);
            Tensor velocityImage = UnpatchifyTokens(velocity, inChannels, hPacked, wPacked, patch);
            velocity.Dispose();

            if (useCfg && _config.GuidanceEmbed == false)
            {
                // Standard CFG path (un-distilled variant).
                Tensor uncondPatched = PatchifyLatent(latent, inChannels, latentH, latentW, patch);
                Tensor uncondVel = _transformer.Forward(Backend, uncondPatched, uncondText!, encoderHidden2: null,
                    t, 1.0f, hPacked, wPacked);
                Tensor uncondImage = UnpatchifyTokens(uncondVel, inChannels, hPacked, wPacked, patch);
                uncondVel.Dispose();
                uncondPatched.Dispose();

                Tensor combined = CfgHelper.ApplyCfg(uncondImage, velocityImage, cfgScale);
                velocityImage.Dispose();
                uncondImage.Dispose();
                velocityImage = combined;
            }

            patched.Dispose();

            Tensor newLatent = new(latentShape, DType.F32);
            scheduler.Step(newLatent, velocityImage, latent, i);
            velocityImage.Dispose();
            latent.Dispose();
            latent = newLatent;

            stepSw.Stop();
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds));
        }

        condText.Dispose();
        uncondText?.Dispose();

        Backend.Sync();
        Backend.FreeWeights(_transformer.EnumerateWeights());

        Logs.Info("Decoding latents to image (32-channel VAE)...");
        Stopwatch vaeSw = Stopwatch.StartNew();
        Tensor decoded = _vaeDecoder.Decode(Backend, latent);
        latent.Dispose();
        vaeSw.Stop();

        byte[] rgb = ImagePostProcessor.TensorToRgbBytes(decoded);
        decoded.Dispose();

        sw.Stop();
        Logs.Info($"Hunyuan Image generation complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        return (rgb, width, height, seed);
    }

    /// <summary>Patchifies <c>[B, C, H, W]</c> -&gt; <c>[B, S, p²·C]</c> with channel-outer ordering inside each patch
    /// (matches the diffusers <c>einops.rearrange("B C (H p) (W q) -&gt; B (H W) (p q C)")</c>).</summary>
    private static Tensor PatchifyLatent(Tensor latent, int channels, int height, int width, int patch)
    {
        int batch = (int)latent.Shape[0];
        int hPacked = height / patch;
        int wPacked = width / patch;
        int seqLen = hPacked * wPacked;
        int patchVolume = patch * patch * channels;

        TensorShape outShape = new(batch, seqLen, patchVolume);
        Tensor result = new(outShape, DType.F32);

        float* src = (float*)latent.DataPointer;
        float* dst = (float*)result.DataPointer;
        long chwStride = (long)channels * height * width;
        long hwStride = (long)height * width;

        for (int b = 0; b < batch; b++)
        {
            float* batchSrc = src + b * chwStride;
            float* batchDst = dst + (long)b * seqLen * patchVolume;
            for (int hp = 0; hp < hPacked; hp++)
            {
                for (int wp = 0; wp < wPacked; wp++)
                {
                    long tokenIdx = (long)hp * wPacked + wp;
                    float* tokenDst = batchDst + tokenIdx * patchVolume;
                    int outIdx = 0;
                    for (int py = 0; py < patch; py++)
                    {
                        int srcRow = hp * patch + py;
                        for (int px = 0; px < patch; px++)
                        {
                            int srcCol = wp * patch + px;
                            for (int c = 0; c < channels; c++)
                                tokenDst[outIdx++] = batchSrc[c * hwStride + srcRow * width + srcCol];
                        }
                    }
                }
            }
        }
        return result;
    }

    /// <summary>Inverse of <see cref="PatchifyLatent"/>.</summary>
    private static Tensor UnpatchifyTokens(Tensor tokens, int channels, int hPacked, int wPacked, int patch)
    {
        int batch = (int)tokens.Shape[0];
        int height = hPacked * patch;
        int width = wPacked * patch;
        int seqLen = hPacked * wPacked;
        int patchVolume = patch * patch * channels;

        TensorShape outShape = new(batch, channels, height, width);
        Tensor result = new(outShape, DType.F32);

        float* src = (float*)tokens.DataPointer;
        float* dst = (float*)result.DataPointer;
        long chwStride = (long)channels * height * width;
        long hwStride = (long)height * width;

        for (int b = 0; b < batch; b++)
        {
            float* batchSrc = src + (long)b * seqLen * patchVolume;
            float* batchDst = dst + b * chwStride;
            for (int hp = 0; hp < hPacked; hp++)
            {
                for (int wp = 0; wp < wPacked; wp++)
                {
                    long tokenIdx = (long)hp * wPacked + wp;
                    float* tokenSrc = batchSrc + tokenIdx * patchVolume;
                    int srcIdx = 0;
                    for (int py = 0; py < patch; py++)
                    {
                        int dstRow = hp * patch + py;
                        for (int px = 0; px < patch; px++)
                        {
                            int dstCol = wp * patch + px;
                            for (int c = 0; c < channels; c++)
                                batchDst[c * hwStride + dstRow * width + dstCol] = tokenSrc[srcIdx++];
                        }
                    }
                }
            }
        }
        return result;
    }
}
