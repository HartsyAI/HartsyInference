using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Runtime;
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
    private readonly HunyuanImageVaeDecoder? _hyVaeDecoder;
    private readonly HunyuanImageConfig _config;

    /// <summary>Keeps the DiT weights GPU-resident across generations (skips the post-loop FreeWeights + next-gen
    /// re-upload). The 7B Qwen2.5-VL TE cannot coexist with the resident DiT on 24 GB, so a prompt-cache MISS under
    /// this flag frees the DiT first, encodes, then re-preloads — repeat prompts skip both (the QwenImagePipeline
    /// staging pattern). Standard-profile default ON (HARTSY_KEEP_MODELS=0 disables).</summary>
    private static readonly bool KeepModelsResident =
        EnvSwitch.IsEnabled("HARTSY_KEEP_MODELS", defaultOn: true);
    private bool _ditResident;

    // Prompt-embedding cache (one cond + one uncond, last-used), keyed on the Qwen token ids — the
    // FLite/QwenImage pattern. A hit skips the whole TE phase (DiT evict + TE preload + encode + free).
    // NOTE: when the ByT5 glyph stream gets wired (encoderHidden2), its embedding must join this cache
    // under the same key — both streams derive from the same prompt.
    private int[]? _cachedCondKey;
    private Tensor? _cachedCond;
    private int[]? _cachedUncondKey;
    private Tensor? _cachedUncond;

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

    /// <summary>Creates a pipeline with the faithful HunyuanImage pixel-shuffle VAE decoder (the generic <see cref="VaeDecoder"/> cannot express its conv→depth-to-space upsamplers).</summary>
    public HunyuanImagePipeline(IBackend backend, HunyuanImageQwenTextEncoder qwenEncoder,
        HunyuanImageTransformer transformer, HunyuanImageVaeDecoder vaeDecoder, HunyuanImageConfig config)
        : base(backend)
    {
        _qwenEncoder = qwenEncoder ?? throw new ArgumentNullException(nameof(qwenEncoder));
        _transformer = transformer;
        _hyVaeDecoder = vaeDecoder;
        _vaeDecoder = null!;
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

        bool useCfg = (request.CfgScale ?? GenerationDefaults.HunyuanImage.CfgScale) > 1.0f;
        if (useCfg && (negativePromptTokenIdsQwen is null || negativeAttentionMaskQwen is null))
            throw new ArgumentException("Negative prompt tokens + mask are required when CfgScale > 1.", nameof(negativePromptTokenIdsQwen));

        int seed = request.Seed ?? SeedGenerator.RandomSeed();

        // Prompt-embedding cache: a hit skips the whole 7B TE phase, which also avoids the TE⇄DiT VRAM swap.
        // The conditioning is per-token only (no resolution-dependent component), so token ids fully key it.
        bool condHit = _cachedCond is not null && _cachedCondKey is not null
            && _cachedCondKey.AsSpan().SequenceEqual(promptTokenIdsQwen);
        bool uncondHit = !useCfg || (_cachedUncond is not null && _cachedUncondKey is not null
            && _cachedUncondKey.AsSpan().SequenceEqual(negativePromptTokenIdsQwen!));
        Tensor cond;
        Tensor? uncond = null;
        if (condHit && uncondHit)
        {
            cond = _cachedCond!;
            if (useCfg) uncond = _cachedUncond;
            Logs.Info("[HunyuanImage] prompt-embedding cache HIT — TE phase skipped.");
        }
        else
        {
            // Cache miss: the DiT cannot share the card with the 7B TE's preload — evict it first.
            if (_ditResident)
            {
                Backend.Sync();
                Backend.FreeWeights(_transformer.EnumerateWeights());
                _ditResident = false;
            }
            Backend.TrimMemoryPool();

            Logs.Info("Encoding text with Qwen2.5-VL-7B (primary 3584-dim per-token)...");
            Backend.PreloadWeights(_qwenEncoder.EnumerateWeights());
            cond = _qwenEncoder.Encode(Backend, promptTokenIdsQwen, promptAttentionMaskQwen);
            uncond = useCfg
                ? _qwenEncoder.Encode(Backend, negativePromptTokenIdsQwen!, negativeAttentionMaskQwen!)
                : null;
            Backend.Sync();
            Backend.FreeWeights(_qwenEncoder.EnumerateWeights());

            // Host-materialize so the cached embeddings survive the activation sweep, then reclaim the
            // encoder's device intermediates before the DiT preload.
            _ = cond.DataPointer;
            if (uncond is not null) _ = uncond.DataPointer;
            Backend.FreeActivations();
            Backend.TrimMemoryPool();

            _cachedCond?.Dispose();
            _cachedCond = cond;
            _cachedCondKey = (int[])promptTokenIdsQwen.Clone();
            if (useCfg)
            {
                _cachedUncond?.Dispose();
                _cachedUncond = uncond;
                _cachedUncondKey = (int[])negativePromptTokenIdsQwen!.Clone();
            }
        }

        return Denoise(cond, uncond, request, seed, textOwned: false, onProgress);
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

        bool useCfg = (request.CfgScale ?? GenerationDefaults.HunyuanImage.CfgScale) > 1.0f;
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

        return Denoise(cond, uncond, request, seed, textOwned: true, onProgress);
    }

    /// <summary>Shared denoise + VAE-decode body. The latent stays in patch-token space for the entire loop
    /// (patchify once, in-place device <c>CfgEulerStep</c> per step, one unpatchify at the end via the device
    /// <see cref="IBackend.UnpatchifyTokens"/> op) so the pipeline glue never forces a per-step D2H round trip —
    /// token layout is identical on both sides of the transformer ((py, px, c) inner order), making token-space
    /// Euler a pure permutation of the old image-space step. When <paramref name="textOwned"/> is true the
    /// conditioning tensors are disposed at the end; the Qwen path passes false (they live in the prompt cache).</summary>
    private (byte[] rgbData, int width, int height, int seed) Denoise(Tensor condText, Tensor? uncondText,
        TextToImageRequest request, int seed, bool textOwned, Action<GenerationProgress>? onProgress)
    {
        (int steps, float cfgScale, int width, int height) = GenerationDefaults.HunyuanImage.Resolve(request);
        const int VaeDownscale = 32;
        if (width % VaeDownscale != 0 || height % VaeDownscale != 0)
            throw new ArgumentException($"Hunyuan Image: width and height must be divisible by {VaeDownscale}.");
        int latentH = height / VaeDownscale;
        int latentW = width / VaeDownscale;
        bool useCfg = cfgScale > 1.0f;
        int patch = _config.PatchSize;
        int inChannels = _config.InChannels;
        int hPacked = latentH / patch;
        int wPacked = latentW / patch;

        Logs.Info($"Hunyuan Image 2.1: {width}x{height}, {steps} steps, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        float distilledGuidance = _config.GuidanceEmbed ? cfgScale : 1.0f;

        TensorShape latentShape = new(1, inChannels, latentH, latentW);

        // Initial noise is created in image space (seed parity with the verified e2e recipe) and patchified
        // ONCE — the loop then integrates in token space, so the old per-step patchify/unpatchify host loops
        // (a full D2H drain of the velocity every step) are gone.
        Tensor noise = SeedGenerator.CreateNoise(latentShape, seed);
        Tensor latentTokens = PatchifyLatent(noise, inChannels, latentH, latentW, patch);
        noise.Dispose();

        FlowMatchEulerDiscreteScheduler scheduler = new(_config.SamplingShift);
        scheduler.SetTimesteps(steps);
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;

        // Bulk-upload transformer weights before the denoise loop. Kept resident across generations under
        // HARTSY_KEEP_MODELS (the prompt-cache miss path evicts them before the TE phase).
        if (!_ditResident)
        {
            Backend.PreloadWeights(_transformer.EnumerateWeights());
            _ditResident = true;
        }

        Logs.Info($"Starting denoise loop ({steps} steps, distilled guidance={distilledGuidance})...");
        for (int i = 0; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float t = timesteps[i];

            // CFG combine + Euler run as ONE in-place device op (CfgEulerStep: z += (neg + gw·(pos−neg))·dt —
            // for gw=cfgScale this IS uncond + cfg·(cond−uncond); gw=1 degenerates to the plain Euler step),
            // keeping the latent tokens device-resident across the whole loop.
            Tensor condVel = _transformer.Forward(Backend, latentTokens, condText, encoderHidden2: null,
                t, distilledGuidance, hPacked, wPacked);
            if (useCfg && _config.GuidanceEmbed == false)
            {
                // Standard CFG path (un-distilled variant).
                Tensor uncondVel = _transformer.Forward(Backend, latentTokens, uncondText!, encoderHidden2: null,
                    t, 1.0f, hPacked, wPacked);
                Backend.CfgEulerStep(latentTokens, condVel, uncondVel, cfgScale, scheduler.Dt(i));
                uncondVel.Dispose();
            }
            else
            {
                Backend.CfgEulerStep(latentTokens, condVel, condVel, 1.0f, scheduler.Dt(i));
            }
            condVel.Dispose();

            stepSw.Stop();
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds));
        }

        if (textOwned)
        {
            condText.Dispose();
            uncondText?.Dispose();
        }

        Backend.Sync();
        if (!KeepModelsResident)
        {
            Backend.FreeWeights(_transformer.EnumerateWeights());
            _ditResident = false;
        }

        // Device unpatchify (innerChannelFastest: (py, px, c) inner order — matches PatchifyLatent) keeps the
        // latent → VAE chain device-resident; the CPU backend falls back to the IBackend host loop.
        Tensor latent = new(latentShape, DType.F32);
        Backend.UnpatchifyTokens(latent, latentTokens, inChannels, hPacked, wPacked, patch, innerChannelFastest: true);
        latentTokens.Dispose();

        Logs.Info("Decoding latents to image (32-channel VAE)...");
        Stopwatch vaeSw = Stopwatch.StartNew();
        Tensor decoded = _hyVaeDecoder is not null
            ? _hyVaeDecoder.Decode(Backend, latent)
            : _vaeDecoder.Decode(Backend, latent);
        latent.Dispose();
        vaeSw.Stop();

        byte[] rgb = ImagePostProcessor.TensorToRgbBytes(decoded);
        decoded.Dispose();

        // Final reclaim: in a long-lived host (SwarmUI) the decode intermediates otherwise sit in device
        // memory until GC finalization and shrink the next generation's budget — and under KEEP_MODELS the
        // trimmed pool is what lets a cross-model switch reclaim VRAM beside the resident DiT.
        Backend.FreeActivations();

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

}
