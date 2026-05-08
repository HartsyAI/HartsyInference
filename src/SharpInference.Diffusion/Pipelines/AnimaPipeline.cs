using System.Diagnostics;
using SharpInference.Core.Backends;
using SharpInference.Core.Logging;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.Vae;
using SharpInference.Diffusion.Requests;
using SharpInference.Diffusion.Schedulers;
using SharpInference.Diffusion.Utilities;

namespace SharpInference.Diffusion.Pipelines;

/// <summary>Anima (Cosmos-Predict2 family) text-to-image pipeline. Image-only path: <c>p_t = 1</c>, no
/// video / world-model conditioning. Caller supplies pre-computed text encoder embeddings — Anima's
/// text encoder hasn't been ported as a first-class C# component yet.</summary>
public sealed unsafe class AnimaPipeline : IDisposable
{
    private readonly IBackend _backend;
    private readonly AnimaTransformer _transformer;
    private readonly VaeDecoder _vaeDecoder;
    private readonly AnimaConfig _config;
    private int _disposed;

    /// <summary>Creates an Anima t2i pipeline. Caller owns the components.</summary>
    public AnimaPipeline(IBackend backend, AnimaTransformer transformer, VaeDecoder vaeDecoder, AnimaConfig config)
    {
        _backend = backend;
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _config = config;
    }

    /// <summary>Generates an image from pre-computed text embeddings <c>[1, T, dim]</c>. CFG dual-pass when
    /// <paramref name="cfgScale"/> &gt; 1 and a negative-prompt embedding is provided.</summary>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromEmbeddings(
        Tensor textEmbeddings,
        TextToImageRequest request,
        float cfgScale = 1.0f,
        Tensor? negativeTextEmbeddings = null,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();

        if (cfgScale > 1.0f && negativeTextEmbeddings is null)
            throw new ArgumentException(
                "negativeTextEmbeddings is required when cfgScale > 1.0.",
                nameof(negativeTextEmbeddings));

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int width = request.Width;
        int height = request.Height;
        int latentH = height / 8;
        int latentW = width / 8;
        int steps = request.Steps;

        Logs.Info($"Anima t2i: {width}x{height}, {steps} steps, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        TensorShape latentShape = new(1, _config.InChannels, latentH, latentW);
        Tensor latent = SeedGenerator.CreateNoise(latentShape, seed);

        FlowMatchEulerDiscreteScheduler scheduler = new(3.0f);
        scheduler.SetTimesteps(steps);
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;

        for (int i = 0; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float t = timesteps[i];

            Tensor velocity = _transformer.Forward(_backend, latent, t, textEmbeddings);

            if (cfgScale > 1.0f)
            {
                Tensor uncond = _transformer.Forward(_backend, latent, t, negativeTextEmbeddings!);
                Tensor combined = new(velocity.Shape, DType.F32);
                ApplyStandardCfg(combined, velocity, uncond, cfgScale);
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

        _backend.Sync();
        _backend.FreeWeights(_transformer.EnumerateWeights());

        Tensor decoded = _vaeDecoder.Decode(_backend, latent);
        latent.Dispose();

        byte[] rgb = ImagePostProcessor.TensorToRgbBytes(decoded);
        decoded.Dispose();

        sw.Stop();
        Logs.Info($"Anima t2i complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        return (rgb, width, height, seed);
    }

    private static void ApplyStandardCfg(Tensor output, Tensor cond, Tensor uncond, float cfg)
    {
        float* condPtr = (float*)cond.DataPointer;
        float* uncondPtr = (float*)uncond.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        long count = output.Shape.ElementCount;
        for (long i = 0; i < count; i++)
        {
            float u = uncondPtr[i];
            outPtr[i] = u + cfg * (condPtr[i] - u);
        }
    }

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    public void Dispose() => Volatile.Write(ref _disposed, 1);
}
