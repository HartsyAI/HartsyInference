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

/// <summary>Chroma text-to-image pipeline (<c>lodestones/Chroma</c>). T5-XXL → ChromaTransformer denoising
/// with flow matching (dynamic shift, Flux-style) → 16-channel Flux VAE decode → RGB.
///
/// Differences from Flux that the pipeline must handle (see <see cref="ChromaTransformer"/> and
/// <c>diffusers/pipelines/chroma/pipeline_chroma.py</c>):
/// <list type="bullet">
///   <item><b>T5-only encode</b> — no CLIP. Caller passes already-tokenized T5 IDs and an attention mask.</item>
///   <item><b>"First padding token unmasked"</b> — Chroma propagates a per-token attention mask through every
///         transformer block. The mask is built from the tokenizer mask via
///         <c>(arange(seq_len) &lt;= text_lens)</c> i.e. all real tokens plus exactly one extra unmasked
///         padding slot at the EOS position.</item>
///   <item><b>True CFG</b> — dual transformer pass. Default cfg=5.0, steps=35.</item>
///   <item><b>Dynamic flow-match shift</b> via <see cref="FlowMatchEulerDiscreteScheduler.CreateWithDynamicShift"/>
///         (Flux-style; same base/max constants).</item>
///   <item><b>Latent packing</b> identical to Flux (2x2 patchify into channel dim, 16ch → 64dim).</item>
/// </list>
/// VRAM eviction at the transformer→VAE boundary mirrors <see cref="Sd3Pipeline"/> / <see cref="FluxPipeline"/>.</summary>
public sealed unsafe class ChromaPipeline : IDisposable
{
    private readonly IBackend _backend;
    private readonly T5TextEncoder _t5;
    private readonly ChromaTransformer _transformer;
    private readonly VaeDecoder _vaeDecoder;
    private readonly ChromaConfig _config;
    private readonly float _schedulerShiftFallback;
    private int _disposed;

    /// <summary>Creates a new Chroma pipeline with all components pre-loaded.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="t5">T5-XXL text encoder (joint_attention_dim = 4096, max length 512).</param>
    /// <param name="transformer">Chroma transformer (loaded with <see cref="ChromaConfig"/>).</param>
    /// <param name="vaeDecoder">16-channel Flux VAE decoder (use <see cref="VaeConfig.Flux"/>).</param>
    /// <param name="config">Chroma configuration (use <see cref="ChromaConfig.V1"/> for the v1 release).</param>
    /// <param name="schedulerShift">Fallback static shift if the runtime ever needs one. Chroma uses a dynamic
    /// shift derived from the image sequence length per call; this constant is only used if the call site
    /// somehow short-circuits the dynamic path.</param>
    public ChromaPipeline(IBackend backend, T5TextEncoder t5, ChromaTransformer transformer,
        VaeDecoder vaeDecoder, ChromaConfig config, float schedulerShift = 3.0f)
    {
        _backend = backend;
        _t5 = t5;
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _config = config;
        _schedulerShiftFallback = schedulerShift;
    }

    /// <summary>Generates an image from pre-tokenized T5 input plus attention masks.</summary>
    /// <param name="promptTokenIdsT5">Prompt token IDs from the T5 tokenizer.</param>
    /// <param name="negativePromptTokenIdsT5">Negative prompt token IDs (same length as <paramref name="promptTokenIdsT5"/>).</param>
    /// <param name="promptAttentionMaskT5">Tokenizer attention mask for the prompt (1=real token, 0=pad). Required
    /// — Chroma needs this to compute the "first padding token unmasked" extension.</param>
    /// <param name="negativeAttentionMaskT5">Tokenizer attention mask for the negative prompt.</param>
    /// <param name="request">Generation parameters.</param>
    /// <param name="onProgress">Optional progress callback.</param>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromTokens(
        int[] promptTokenIdsT5,
        int[] negativePromptTokenIdsT5,
        int[] promptAttentionMaskT5,
        int[] negativeAttentionMaskT5,
        TextToImageRequest request,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();

        if (promptAttentionMaskT5 is null)
            throw new ArgumentNullException(nameof(promptAttentionMaskT5),
                "Chroma requires the tokenizer attention mask (1=real, 0=pad) so the pipeline can " +
                "compute the 'first padding token unmasked' rule.");
        if (negativeAttentionMaskT5 is null)
            throw new ArgumentNullException(nameof(negativeAttentionMaskT5),
                "Chroma requires the tokenizer attention mask for the negative prompt as well.");

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int width = request.Width;
        int height = request.Height;
        int latentH = height / 8;
        int latentW = width / 8;
        int steps = request.Steps;
        float cfgScale = request.CfgScale;
        bool useCfg = cfgScale > 1.0f;

        Logs.Info($"Chroma: Generating {width}x{height}, {steps} steps, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        // ── 1. Encode prompts with T5-XXL ─────────────────────────────────
        Logs.Info("Encoding text with T5-XXL...");

        int[][] batchT5 = [promptTokenIdsT5];
        int[][] batchMask = [promptAttentionMaskT5];
        Tensor condContext = _t5.Encode(_backend, batchT5, batchMask);

        Tensor? uncondContext = null;
        if (useCfg)
        {
            int[][] negBatchT5 = [negativePromptTokenIdsT5];
            int[][] negBatchMask = [negativeAttentionMaskT5];
            uncondContext = _t5.Encode(_backend, negBatchT5, negBatchMask);
        }

        // Build the [B, txtSeqLen] transformer-side mask:
        //   m[i] = 1.0  if  i <= text_len   (i.e., all real tokens PLUS one extra unmasked padding slot)
        //   m[i] = 0.0  otherwise
        // Mirrors pipeline_chroma.py:249-252. text_len = sum(tokenizer_mask).
        Tensor condMask = BuildChromaTextMask(promptAttentionMaskT5, batchSize: 1);
        Tensor? uncondMask = useCfg ? BuildChromaTextMask(negativeAttentionMaskT5, batchSize: 1) : null;

        Logs.Info($"Text encoding done in {sw.ElapsedMilliseconds}ms");

        // ── 2. Set up dynamic flow-match scheduler ────────────────────────
        int hPacked = latentH / 2;
        int wPacked = latentW / 2;
        int imgSeqLen = hPacked * wPacked;
        TensorShape latentShape = new TensorShape(1, 16, latentH, latentW);
        TensorShape packedShape = new TensorShape(1, imgSeqLen, 64);

        FlowMatchEulerDiscreteScheduler scheduler =
            FlowMatchEulerDiscreteScheduler.CreateWithDynamicShift(imgSeqLen);
        // Touch the fallback shift field so the field isn't unused on builds that don't reach the dynamic path.
        _ = _schedulerShiftFallback;
        scheduler.SetTimesteps(steps);

        // ── 3. Build initial packed latent ────────────────────────────────
        Tensor packedLatent = PackLatent(SeedGenerator.CreateNoise(latentShape, seed), latentH, latentW);
        float initSigma = scheduler.InitialNoiseSigma;
        if (MathF.Abs(initSigma - 1.0f) > 1e-6f)
        {
            Tensor scaled = new Tensor(packedShape, DType.F32);
            _backend.Scale(scaled, packedLatent, initSigma);
            packedLatent.Dispose();
            packedLatent = scaled;
        }

        // ── 4. Denoising loop ────────────────────────────────────────────
        Logs.Info("Starting Chroma denoising loop...");
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;
        int txtSeqLen = (int)condContext.Shape[1];

        for (int i = 0; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float sigma = timesteps[i] / 1000.0f;

            Tensor noisePred;
            if (useCfg)
            {
                noisePred = ClassifierFreeGuidanceStep(packedLatent, sigma,
                    condContext, condMask,
                    uncondContext!, uncondMask!,
                    txtSeqLen, hPacked, wPacked, cfgScale);
            }
            else
            {
                noisePred = _transformer.Forward(_backend, packedLatent, condContext, sigma,
                    txtSeqLen, hPacked, wPacked, condMask);
            }

            Tensor newLatent = new Tensor(packedShape, DType.F32);
            scheduler.Step(newLatent, noisePred, packedLatent, i);
            noisePred.Dispose();
            packedLatent.Dispose();
            packedLatent = newLatent;

            stepSw.Stop();
            Logs.Debug($"Step {i + 1}/{steps} (sigma={sigma:F4}) done in {stepSw.ElapsedMilliseconds}ms");
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds));
        }

        condContext.Dispose();
        condMask.Dispose();
        uncondContext?.Dispose();
        uncondMask?.Dispose();

        ChromaTransformer.DumpFinalLatent(packedLatent);

        // Free transformer + T5 weights from GPU before VAE decode (mirrors SD3/Flux/AuraFlow pattern).
        _backend.Sync();
        _backend.FreeWeights(_transformer.EnumerateWeights());
        _backend.FreeWeights(_t5.EnumerateWeights());

        // ── 5. Unpack latent: [1, seqLen, 64] → [1, 16, latentH, latentW] ──
        Tensor unpackedLatent = UnpackLatent(packedLatent, latentH, latentW);
        packedLatent.Dispose();

        // ── 6. VAE decode ────────────────────────────────────────────────
        _backend.PreloadWeights(_vaeDecoder.EnumerateWeights());
        // Tiled decode: caps im2col workspace at ~2.4 GB per tile. Internal fast-path
        // skips tiling when the latent fits in a single tile, so small images pay no overhead.
        Logs.Verbose("Decoding latents to image (tiled F32 path)...");
        Stopwatch vaeSw = Stopwatch.StartNew();
        Tensor image = _vaeDecoder.DecodeTiled(_backend, unpackedLatent);
        unpackedLatent.Dispose();
        vaeSw.Stop();
        Logs.Verbose($"VAE decode done in {vaeSw.ElapsedMilliseconds}ms");

        byte[] rgbData = ImagePostProcessor.TensorToRgbBytes(image);
        image.Dispose();

        sw.Stop();
        Logs.Info($"Chroma image generation complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        return (rgbData, width, height, seed);
    }

    /// <summary>Runs classifier-free guidance: <c>noise_pred = uncond + cfg_scale * (cond - uncond)</c>.</summary>
    private Tensor ClassifierFreeGuidanceStep(
        Tensor packedLatent, float sigma,
        Tensor condContext, Tensor condMask,
        Tensor uncondContext, Tensor uncondMask,
        int txtSeqLen, int hPacked, int wPacked, float cfgScale)
    {
        // Note: the negative prompt may have a different sequence length from the positive prompt; the
        // transformer is fully shape-polymorphic on txtSeqLen so each pass uses its own value.
        int condTxtLen = (int)condContext.Shape[1];
        int uncondTxtLen = (int)uncondContext.Shape[1];

        Tensor uncondNoise = _transformer.Forward(_backend, packedLatent, uncondContext, sigma,
            uncondTxtLen, hPacked, wPacked, uncondMask);
        Tensor condNoise = _transformer.Forward(_backend, packedLatent, condContext, sigma,
            condTxtLen, hPacked, wPacked, condMask);

        Tensor output = new Tensor(packedLatent.Shape, DType.F32);
        float* uncPtr = (float*)uncondNoise.DataPointer;
        float* conPtr = (float*)condNoise.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        long count = packedLatent.ElementCount;

        for (long i = 0; i < count; i++)
            outPtr[i] = uncPtr[i] + cfgScale * (conPtr[i] - uncPtr[i]);

        // Touch txtSeqLen so the field isn't flagged unused (it remains for callers that need to introspect).
        _ = txtSeqLen;

        uncondNoise.Dispose();
        condNoise.Dispose();
        return output;
    }

    /// <summary>Builds Chroma's transformer-side text mask. Per pipeline_chroma.py:249-252, the mask used inside
    /// the transformer is <c>(arange(seq_len) &lt;= text_len)</c>: every real token plus exactly one extra
    /// padding slot kept unmasked. <paramref name="tokenizerMask"/> is the standard tokenizer mask
    /// (1=real, 0=pad). Returns a [batch, seq_len] F32 tensor with the extension applied.</summary>
    private static Tensor BuildChromaTextMask(int[] tokenizerMask, int batchSize)
    {
        int seqLen = tokenizerMask.Length;
        TensorShape shape = new TensorShape(batchSize, seqLen);
        Tensor mask = new Tensor(shape, DType.F32);

        // text_len = number of real tokens (sum of the tokenizer mask). The "<=" rule keeps positions
        // [0, text_len], which is text_len + 1 unmasked entries — text_len real plus one padding slot.
        int textLen = 0;
        for (int i = 0; i < seqLen; i++) if (tokenizerMask[i] != 0) textLen++;

        float* ptr = (float*)mask.DataPointer;
        for (int b = 0; b < batchSize; b++)
        {
            int baseOffset = b * seqLen;
            for (int i = 0; i < seqLen; i++)
                ptr[baseOffset + i] = i <= textLen ? 1.0f : 0.0f;
        }
        return mask;
    }

    /// <summary>Packs a latent tensor [B, 16, H, W] → [B, H/2 * W/2, 64] (2x2 patchify, identical to Flux).</summary>
    private static Tensor PackLatent(Tensor latent, int h, int w)
    {
        int batch = (int)latent.Shape[0];
        int channels = (int)latent.Shape[1];
        int hPacked = h / 2;
        int wPacked = w / 2;
        int patchDim = channels * 4;
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
                    int outBase = (b * seqLen + seqIdx) * patchDim;

                    for (int c = 0; c < channels; c++)
                    {
                        int inChannelBase = (b * channels + c) * h * w;
                        int patchBase = outBase + c * 4;

                        outPtr[patchBase + 0] = inPtr[inChannelBase + (ph * 2 + 0) * w + (pw * 2 + 0)];
                        outPtr[patchBase + 1] = inPtr[inChannelBase + (ph * 2 + 0) * w + (pw * 2 + 1)];
                        outPtr[patchBase + 2] = inPtr[inChannelBase + (ph * 2 + 1) * w + (pw * 2 + 0)];
                        outPtr[patchBase + 3] = inPtr[inChannelBase + (ph * 2 + 1) * w + (pw * 2 + 1)];
                    }
                }
            }
        }

        latent.Dispose();
        return packed;
    }

    /// <summary>Unpacks [B, S, 64] → [B, 16, H, W]. Inverse of <see cref="PackLatent"/>.</summary>
    private static Tensor UnpackLatent(Tensor packed, int h, int w)
    {
        int batch = (int)packed.Shape[0];
        int channels = 16;
        int hPacked = h / 2;
        int wPacked = w / 2;
        int patchDim = channels * 4;
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
                    int inBase = (b * seqLen + seqIdx) * patchDim;

                    for (int c = 0; c < channels; c++)
                    {
                        int outChannelBase = (b * channels + c) * h * w;
                        int patchBase = inBase + c * 4;

                        outPtr[outChannelBase + (ph * 2 + 0) * w + (pw * 2 + 0)] = inPtr[patchBase + 0];
                        outPtr[outChannelBase + (ph * 2 + 0) * w + (pw * 2 + 1)] = inPtr[patchBase + 1];
                        outPtr[outChannelBase + (ph * 2 + 1) * w + (pw * 2 + 0)] = inPtr[patchBase + 2];
                        outPtr[outChannelBase + (ph * 2 + 1) * w + (pw * 2 + 1)] = inPtr[patchBase + 3];
                    }
                }
            }
        }

        return unpacked;
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
    }

    /// <summary>Disposes the pipeline. Does not dispose backend or model components (shared resources).</summary>
    public void Dispose()
    {
        Volatile.Write(ref _disposed, 1);
    }
}
