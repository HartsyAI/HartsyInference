using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>Boogu-Image-0.1 pipeline (Base / Turbo text-to-image and Edit text+image-to-image). Caller supplies the
/// Qwen3-VL instruction embeddings (final hidden state <c>[1, T, 4096]</c>) — see
/// <c>docs/Research/BOOGU_IMAGE.md §5/§7</c> for how those are produced (text-only for T2I; via the Qwen3-VL vision
/// tower for edit). This mirrors the established <see cref="OmniGen2Pipeline"/> contract where the encoder forward is
/// owned outside the pipeline.
///
/// <para>T2I uses single CFG <c>pred = neg + tg·(cond − neg)</c>. Edit uses Boogu double guidance
/// <c>pred = cond + (tg − 1)·(cond − drop_text) + (ig − 1)·(drop_text − drop_all)</c> with three transformer passes per
/// step (cond = text+ref, drop_text = neg+ref, drop_all = neg+no-ref). Sampling is the ascending v1
/// flow-match schedule (<see cref="BooguFlowMatchScheduler"/>).</para></summary>
public sealed class BooguImagePipeline : DiffusionPipelineBase
{
    private readonly BooguImageTransformer _transformer;
    private readonly VaeDecoder _vaeDecoder;
    private readonly VaeEncoder? _vaeEncoder;
    private readonly BooguImageConfig _config;

    /// <summary>Creates a Boogu-Image pipeline. Pass <paramref name="vaeEncoder"/> to enable the edit path (it encodes
    /// reference images into the DiT latent stream); null restricts the pipeline to T2I. Caller owns each component.</summary>
    public BooguImagePipeline(IBackend backend, BooguImageTransformer transformer, VaeDecoder vaeDecoder,
        VaeEncoder? vaeEncoder, BooguImageConfig config)
        : base(backend)
    {
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _vaeEncoder = vaeEncoder;
        _config = config;
    }

    /// <summary>Text-to-image. <paramref name="instructionEmbeddings"/> is the Qwen3-VL final hidden state
    /// <c>[1, T, 4096]</c> for the (positive) instruction; <paramref name="negativeInstructionEmbeddings"/> is required
    /// when <paramref name="textGuidanceScale"/> &gt; 1 (use the empty-string encode). Turbo: pass
    /// <c>textGuidanceScale = 1</c> and ~4 steps.</summary>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromEmbeddings(
        Tensor instructionEmbeddings,
        TextToImageRequest request,
        float textGuidanceScale = 4.0f,
        Tensor? negativeInstructionEmbeddings = null,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        if (textGuidanceScale > 1.0f && negativeInstructionEmbeddings is null)
            throw new ArgumentException("negativeInstructionEmbeddings is required when textGuidanceScale > 1.0.",
                nameof(negativeInstructionEmbeddings));

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        (int steps, _, int width, int height) = GenerationDefaults.Generic.Resolve(request);
        int latentH = height / 8;
        int latentW = width / 8;
        int seqLen = (latentH / _config.PatchSize) * (latentW / _config.PatchSize);

        Logs.Info($"Boogu-Image t2i: {width}x{height}, {steps} steps, tg={textGuidanceScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        TensorShape latentShape = new(1, _config.InChannels, latentH, latentW);
        Tensor latent = SeedGenerator.CreateNoise(latentShape, seed);

        BooguFlowMatchScheduler scheduler = new(seqLen);
        scheduler.SetTimesteps(steps);
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;

        Backend.PreloadWeights(_transformer.EnumerateWeights());

        for (int i = 0; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float t = timesteps[i];

            Tensor velocity = _transformer.Forward(Backend, latent, t, instructionEmbeddings);
            if (textGuidanceScale > 1.0f)
            {
                Tensor uncond = _transformer.Forward(Backend, latent, t, negativeInstructionEmbeddings!);
                Tensor combined = CfgHelper.ApplyCfg(uncond, velocity, textGuidanceScale);
                uncond.Dispose();
                velocity.Dispose();
                velocity = combined;
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
        Logs.Info($"Boogu-Image t2i complete in {sw.ElapsedMilliseconds}ms (seed={seed})");
        return (rgb, width, height, seed);
    }

    /// <summary>Edit (text+image-to-image) with Boogu double guidance. The three instruction embeddings are produced by
    /// the caller from the Qwen3-VL encoder: <paramref name="condEmbeddings"/> (positive instruction with the reference
    /// image seen by the vision tower), <paramref name="dropTextEmbeddings"/> (negative instruction, image still seen)
    /// and <paramref name="dropAllEmbeddings"/> (negative instruction, no image). Reference images are RGB tensors
    /// <c>[1, 3, Hr, Wr]</c> in <c>[-1, 1]</c>; they are VAE-encoded into the DiT latent stream. When
    /// <paramref name="imageGuidanceScale"/> ≤ 1 the drop-all pass is skipped (text-only guidance, reference kept).</summary>
    public (byte[] rgbData, int width, int height, int seed) EditFromEmbeddings(
        Tensor condEmbeddings,
        Tensor dropTextEmbeddings,
        Tensor? dropAllEmbeddings,
        IReadOnlyList<Tensor> referenceImages,
        TextToImageRequest request,
        float textGuidanceScale = 4.0f,
        float imageGuidanceScale = 1.0f,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        if (_vaeEncoder is null)
            throw new InvalidOperationException("Edit requires a VAE encoder; construct the pipeline with vaeEncoder != null.");
        if (referenceImages.Count == 0)
            throw new ArgumentException("Edit requires at least one reference image.", nameof(referenceImages));
        bool doubleGuide = imageGuidanceScale > 1.0f;
        if (doubleGuide && dropAllEmbeddings is null)
            throw new ArgumentException("dropAllEmbeddings is required when imageGuidanceScale > 1.0.", nameof(dropAllEmbeddings));

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        (int steps, _, int width, int height) = GenerationDefaults.Generic.Resolve(request);
        int latentH = height / 8;
        int latentW = width / 8;
        int seqLen = (latentH / _config.PatchSize) * (latentW / _config.PatchSize);

        Logs.Info($"Boogu-Image edit: {width}x{height}, {steps} steps, tg={textGuidanceScale}, ig={imageGuidanceScale}, refs={referenceImages.Count}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        // Encode reference images into latents once.
        Tensor[] refLatents = new Tensor[referenceImages.Count];
        for (int j = 0; j < referenceImages.Count; j++)
            refLatents[j] = _vaeEncoder.Encode(Backend, referenceImages[j]);

        TensorShape latentShape = new(1, _config.InChannels, latentH, latentW);
        Tensor latent = SeedGenerator.CreateNoise(latentShape, seed);

        BooguFlowMatchScheduler scheduler = new(seqLen);
        scheduler.SetTimesteps(steps);
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;

        Backend.PreloadWeights(_transformer.EnumerateWeights());

        for (int i = 0; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float t = timesteps[i];

            Tensor cond = _transformer.Forward(Backend, latent, t, condEmbeddings, refLatents);
            Tensor dropText = _transformer.Forward(Backend, latent, t, dropTextEmbeddings, refLatents);

            Tensor velocity;
            if (doubleGuide)
            {
                Tensor dropAll = _transformer.Forward(Backend, latent, t, dropAllEmbeddings!, null);
                velocity = CombineDoubleGuidance(cond, dropText, dropAll, textGuidanceScale, imageGuidanceScale);
                dropAll.Dispose();
            }
            else
            {
                // Text-only guidance with the reference kept: cond + (tg-1)*(cond - dropText).
                velocity = CfgHelper.ApplyCfg(dropText, cond, textGuidanceScale);
            }
            cond.Dispose();
            dropText.Dispose();

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
        foreach (Tensor r in refLatents) r.Dispose();

        Tensor decoded = _vaeDecoder.Decode(Backend, latent);
        latent.Dispose();
        byte[] rgb = ImagePostProcessor.TensorToRgbBytes(decoded);
        decoded.Dispose();

        sw.Stop();
        Logs.Info($"Boogu-Image edit complete in {sw.ElapsedMilliseconds}ms (seed={seed})");
        return (rgb, width, height, seed);
    }

    /// <summary><c>pred = cond + (tg − 1)·(cond − dropText) + (ig − 1)·(dropText − dropAll)</c>, elementwise.</summary>
    private static unsafe Tensor CombineDoubleGuidance(Tensor cond, Tensor dropText, Tensor dropAll, float tg, float ig)
    {
        Tensor output = new(cond.Shape, DType.F32);
        float* c = (float*)cond.DataPointer;
        float* dt = (float*)dropText.DataPointer;
        float* da = (float*)dropAll.DataPointer;
        float* o = (float*)output.DataPointer;
        long n = cond.ElementCount;
        float tgm = tg - 1.0f;
        float igm = ig - 1.0f;
        for (long i = 0; i < n; i++)
            o[i] = c[i] + tgm * (c[i] - dt[i]) + igm * (dt[i] - da[i]);
        return output;
    }
}
