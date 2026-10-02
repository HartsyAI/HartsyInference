using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Auk;

/// <summary>Per-channel latent statistics of the AuK VAE: <c>(z - mean) / sqrt(var)</c> to normalize and <c>z * sqrt(var) + mean</c> to denormalize, over time-major latents <c>[1, T, D]</c>.</summary>
/// <remarks>The checkpoint buffer <c>global_log_std</c> is misnamed: it is a positive variance-like quantity and is used under a square root, never exponentiated.</remarks>
public sealed unsafe class AukVaeStats : IDisposable
{
    private readonly Tensor _normScale;
    private readonly Tensor _normShift;
    private readonly Tensor _denormScale;
    private readonly Tensor _denormShift;

    private AukVaeStats(Tensor normScale, Tensor normShift, Tensor denormScale, Tensor denormShift)
    {
        _normScale = normScale;
        _normShift = normShift;
        _denormScale = denormScale;
        _denormShift = denormShift;
    }

    /// <summary>Latent channel count.</summary>
    public int Dim => (int)_normScale.ElementCount;

    /// <summary>Builds the statistics from <c>{prefix}global_mean</c> and <c>{prefix}global_log_std</c> (both <c>[D]</c>).</summary>
    public static AukVaeStats Load(IReadOnlyDictionary<string, Tensor> w, string prefix = "")
    {
        Tensor rawMean = w[$"{prefix}global_mean"];
        Tensor rawVariance = w[$"{prefix}global_log_std"];
        Tensor mean = WhisperOps.EnsureF32(rawMean);
        Tensor variance = WhisperOps.EnsureF32(rawVariance);
        try
        {
            return Create(mean, variance);
        }
        finally
        {
            if (!ReferenceEquals(mean, rawMean)) mean.Dispose();
            if (!ReferenceEquals(variance, rawVariance)) variance.Dispose();
        }
    }

    /// <summary>Builds the statistics from explicit per-channel mean and variance-like tensors.</summary>
    public static AukVaeStats Create(Tensor mean, Tensor variance)
    {
        int d = (int)mean.ElementCount;
        if (variance.ElementCount != d) throw new ArgumentException("mean and variance must have the same length.");
        Tensor ns = new(new TensorShape(d), DType.F32);
        Tensor nb = new(new TensorShape(d), DType.F32);
        Tensor ds = new(new TensorShape(d), DType.F32);
        Tensor db = new(new TensorShape(d), DType.F32);
        float* m = (float*)mean.DataPointer;
        float* v = (float*)variance.DataPointer;
        for (int i = 0; i < d; i++)
        {
            float sd = MathF.Sqrt(v[i]);
            ((float*)ns.DataPointer)[i] = 1f / sd;
            ((float*)nb.DataPointer)[i] = -m[i] / sd;
            ((float*)ds.DataPointer)[i] = sd;
            ((float*)db.DataPointer)[i] = m[i];
        }
        return new AukVaeStats(ns, nb, ds, db);
    }

    /// <summary>Normalizes time-major latents <c>[1, T, D]</c> into a new tensor.</summary>
    public Tensor Normalize(IBackend backend, Tensor latents) => Apply(backend, latents, _normScale, _normShift);

    /// <summary>Maps normalized time-major latents <c>[1, T, D]</c> back to raw VAE latents in a new tensor.</summary>
    public Tensor Denormalize(IBackend backend, Tensor latents) => Apply(backend, latents, _denormScale, _denormShift);

    public IEnumerable<Tensor> EnumerateWeights() => [_normScale, _normShift, _denormScale, _denormShift];

    public void Dispose()
    {
        _normScale.Dispose();
        _normShift.Dispose();
        _denormScale.Dispose();
        _denormShift.Dispose();
    }

    private Tensor Apply(IBackend backend, Tensor latents, Tensor scale, Tensor shift)
    {
        if (latents.Shape.Rank != 3 || (int)latents.Shape[2] != Dim)
            throw new ArgumentException($"Expected time-major latents [B, T, {Dim}]; got {latents.Shape}.", nameof(latents));
        Tensor o = new(latents.Shape, DType.F32);
        backend.AffineBroadcastLastDim(o, latents, scale, shift);
        return o;
    }
}
