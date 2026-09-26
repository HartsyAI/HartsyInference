using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>Brownian-motion increments over sigma, ComfyUI's <c>BrownianTreeNoiseSampler</c> role: draws over
/// overlapping intervals are correlated exactly as one Brownian path would make them.</summary>
/// <remarks>Built by sequential Brownian-bridge conditioning, so it is distributionally exact and reproducible for a
/// fixed query order, but not bit-compatible with torchsde's tree.</remarks>
public sealed class BrownianNoiseSource : INoiseSource
{
    private readonly TensorShape _shape;
    private readonly int _seed;
    private readonly float _tMin;
    private readonly float _tMax;
    private readonly List<(float T, Tensor W)> _points = [];
    private bool _disposed;

    /// <summary>Creates the path on <c>[sigmaMin, sigmaMax]</c>, pinned at <c>W(sigmaMin) = 0</c>.</summary>
    public BrownianNoiseSource(TensorShape shape, int seed, float sigmaMin, float sigmaMax)
    {
        if (!(sigmaMin >= 0f) || !(sigmaMax > sigmaMin))
        {
            throw new ArgumentException($"Brownian interval needs 0 <= sigmaMin < sigmaMax; got [{sigmaMin}, {sigmaMax}].");
        }
        _shape = shape;
        _seed = seed;
        _tMin = sigmaMin;
        _tMax = sigmaMax;
        _points.Add((sigmaMin, new Tensor(shape, DType.F32)));
        _points.Add((sigmaMax, Draw(sigmaMax, MathF.Sqrt(sigmaMax - sigmaMin))));
    }

    /// <summary>Builds a source spanning the positive range of <paramref name="sigmas"/>, as ComfyUI does.</summary>
    public static BrownianNoiseSource ForSchedule(TensorShape shape, int seed, float[] sigmas)
    {
        ArgumentNullException.ThrowIfNull(sigmas);
        float min = float.MaxValue;
        float max = 0f;
        foreach (float s in sigmas)
        {
            if (s > 0f)
            {
                min = MathF.Min(min, s);
                max = MathF.Max(max, s);
            }
        }
        return min < max ? new BrownianNoiseSource(shape, seed, min, max) : new BrownianNoiseSource(shape, seed, 0f, MathF.Max(max, 1f));
    }

    /// <inheritdoc/>
    public unsafe Tensor Sample(int stepIndex, int subDraw, float sigma, float sigmaNext)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        float a = Math.Clamp(sigma, _tMin, _tMax);
        float b = Math.Clamp(sigmaNext, _tMin, _tMax);
        Tensor result = new Tensor(_shape, DType.F32);
        float* dst = (float*)result.DataPointer;
        int count = (int)_shape.ElementCount;
        if (a == b)
        {
            new Span<float>(dst, count).Clear();
            return result;
        }
        float* wa = (float*)Query(a).DataPointer;
        float* wb = (float*)Query(b).DataPointer;
        float inv = 1f / MathF.Sqrt(MathF.Abs(b - a));
        for (int i = 0; i < count; i++)
        {
            dst[i] = (wb[i] - wa[i]) * inv;
        }
        return result;
    }

    /// <summary>W(t), sampled from the bridge between its stored neighbours on first request.</summary>
    private unsafe Tensor Query(float t)
    {
        int hi = 0;
        while (hi < _points.Count && _points[hi].T < t)
        {
            hi++;
        }
        if (hi < _points.Count && _points[hi].T == t)
        {
            return _points[hi].W;
        }
        (float ta, Tensor wa) = _points[hi - 1];
        (float tb, Tensor wb) = _points[hi];
        float span = tb - ta;
        float w = (t - ta) / span;
        float std = MathF.Sqrt((t - ta) * (tb - t) / span);
        Tensor wt = Draw(t, std);
        float* dst = (float*)wt.DataPointer;
        float* pa = (float*)wa.DataPointer;
        float* pb = (float*)wb.DataPointer;
        int count = (int)_shape.ElementCount;
        for (int i = 0; i < count; i++)
        {
            dst[i] += pa[i] + (w * (pb[i] - pa[i]));
        }
        _points.Insert(hi, (t, wt));
        return wt;
    }

    private unsafe Tensor Draw(float t, float scale)
    {
        Tensor z = SeedGenerator.CreateNoise(_shape, SamplerOps.StepSeed(_seed, BitConverter.SingleToInt32Bits(t), 7));
        float* p = (float*)z.DataPointer;
        int count = (int)_shape.ElementCount;
        for (int i = 0; i < count; i++)
        {
            p[i] *= scale;
        }
        return z;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }
        _disposed = true;
        foreach ((float _, Tensor w) in _points)
        {
            w.Dispose();
        }
        _points.Clear();
    }
}
