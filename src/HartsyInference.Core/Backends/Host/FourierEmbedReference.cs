using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Host;

/// <summary>The executable form of the Fourier embedding contract (<c>docs/BACKEND_OPS.md</c>): per point, the coordinates (when
/// included), then <c>sin(x · 2^b)</c> coordinate-major over the bands, then the matching cosines. The <see cref="IBackend"/>
/// default and the oracle the kernel is checked against.</summary>
public static unsafe class FourierEmbedReference
{
    public static void Apply(Tensor output, Tensor coords, in FourierEmbedSpec spec)
    {
        long count = FourierEmbedContract.Validate(output, coords, spec);
        int dims = spec.InputDims, bands = spec.Bands, width = spec.Width;
        int sinBase = spec.IncludeInput ? dims : 0, cosBase = sinBase + dims * bands;
        float* dp = (float*)output.DataPointer, cp = (float*)coords.DataPointer;
        for (long i = 0; i < count; i++)
        {
            long o = i * width;
            for (int c = 0; c < dims; c++)
            {
                float x = cp[i * dims + c];
                if (spec.IncludeInput) dp[o + c] = x;
                for (int band = 0; band < bands; band++)
                {
                    float a = x * (1 << band);
                    dp[o + sinBase + c * bands + band] = MathF.Sin(a);
                    dp[o + cosBase + c * bands + band] = MathF.Cos(a);
                }
            }
        }
    }
}
