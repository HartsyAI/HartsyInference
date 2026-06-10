using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Vae;

/// <summary>Wan2.2 VAE channel-wise RMS norm (<c>RMS_norm</c> in <c>vae2_2.py</c>: <c>F.normalize(x, dim=channel)·√dim·gamma</c>, which equals RMS-over-channels × gamma). Normalizes the channel vector at each spatial-temporal position of a <c>[B, C, T, H, W]</c> tensor and scales by a per-channel learnable <c>gamma</c>. Reusable by any Wan-family video VAE.</summary>
public sealed unsafe class WanRmsNorm
{
    private readonly int _channels;
    private readonly float _eps;
    private Tensor? _gamma;   // [C] (flattened from [C,1,1,1])

    public WanRmsNorm(int channels, float eps = 1e-12f)
    {
        _channels = channels;
        _eps = eps;
    }

    /// <summary>Loads the per-channel scale (<c>gamma</c>), cast to F32. Accepts any shape whose element count is the channel count.</summary>
    public void LoadWeights(Tensor gamma)
    {
        if (gamma.Shape.ElementCount != _channels)
            throw new ArgumentException($"gamma has {gamma.Shape.ElementCount} elements, expected {_channels}.", nameof(gamma));
        _gamma = gamma.DType == DType.F32 ? gamma : gamma.CastTo(DType.F32);
    }

    /// <summary>Enumerates the gamma weight for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_gamma is not null) yield return _gamma;
    }

    /// <summary>Returns a new <c>[B, C, T, H, W]</c> tensor: per position, <c>out[c] = x[c] / rms_over_C(x) · gamma[c]</c>.</summary>
    public Tensor Forward(Tensor x)
    {
        int b = (int)x.Shape[0], c = (int)x.Shape[1];
        if (c != _channels) throw new ArgumentException($"input channels {c} != {_channels}.", nameof(x));
        long spatial = x.Shape.ElementCount / ((long)b * c);
        float sqrtC = MathF.Sqrt(c);

        Tensor outT = new Tensor(x.Shape, DType.F32);
        float* xp = (float*)x.DataPointer;
        float* op = (float*)outT.DataPointer;
        float* g = _gamma is null ? null : (float*)_gamma.DataPointer;

        for (int bi = 0; bi < b; bi++)
        {
            long baseB = (long)bi * c * spatial;
            for (long s = 0; s < spatial; s++)
            {
                // L2 over channels at this position.
                double sumSq = 0;
                for (int ci = 0; ci < c; ci++)
                {
                    float v = xp[baseB + (long)ci * spatial + s];
                    sumSq += (double)v * v;
                }
                float denom = MathF.Max((float)Math.Sqrt(sumSq), _eps);
                float scale = sqrtC / denom;
                for (int ci = 0; ci < c; ci++)
                {
                    long off = baseB + (long)ci * spatial + s;
                    float gv = g is null ? 1f : g[ci];
                    op[off] = xp[off] * scale * gv;
                }
            }
        }
        return outT;
    }
}
