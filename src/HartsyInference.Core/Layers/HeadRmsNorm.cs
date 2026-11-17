using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Layers;

/// <summary>Per-head query/key normalization <c>F.normalize(x) · sqrt(headDim) · gamma</c> (upstream <c>MultiHeadRMSNorm</c>), as an
/// RMS norm without a scale followed by the learned per-channel <c>gamma</c>. <c>F.normalize</c> clamps the norm at 1e-12, which is
/// 1e-24 / headDim on the mean square, so an all-zero head stays zero. Owns one <c>[headDim]</c> tensor of ones.</summary>
public sealed class HeadRmsNorm : IDisposable
{
    private const float NormClampSquared = 1e-24f;

    private readonly float _eps;
    private int _disposed;

    /// <param name="headDim">Width of one head, the last axis the norm runs over.</param>
    public HeadRmsNorm(int headDim)
    {
        if (headDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(headDim), headDim, "The head width must be positive.");
        _eps = NormClampSquared / headDim;
        Ones = new Tensor(new TensorShape(headDim), DType.F32);
        Ones.AsSpan<float>().Fill(1f);
    }

    /// <summary>The unit scale the RMS norm reads; a block that preloads its weights on the device lists it with them.</summary>
    public Tensor Ones { get; }

    /// <summary>Normalizes every head of <paramref name="heads"/> (last axis <c>headDim</c>) and scales by <paramref name="gamma"/>,
    /// which <c>AffineBroadcastLastDim</c> broadcasts over the last axis of <paramref name="gammaShape"/> (the shape of
    /// <paramref name="heads"/> when null; pass the flat <c>[1, tokens, heads·headDim]</c> shape for a flat <c>[heads·headDim]</c> gamma).
    /// Consumes <paramref name="heads"/>; the caller owns the result.</summary>
    public Tensor Apply(IBackend backend, Tensor heads, Tensor gamma, TensorShape? gammaShape = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(heads);
        ArgumentNullException.ThrowIfNull(gamma);
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        TensorShape shape = gammaShape ?? heads.Shape;
        if (shape.ElementCount != heads.Shape.ElementCount)
            throw new ArgumentException($"The gamma shape {shape} does not hold the {heads.Shape.ElementCount} elements of the heads {heads.Shape}.", nameof(gammaShape));
        Tensor rms = new(heads.Shape, DType.F32);
        try
        {
            backend.RmsNorm(rms, heads, Ones, _eps);
            heads.Dispose();
            Tensor o = new(shape, DType.F32);
            try
            {
                backend.AffineBroadcastLastDim(o, shape.Equals(rms.Shape) ? rms : rms.Reshape(shape), gamma, null);
            }
            catch
            {
                o.Dispose();
                throw;
            }

            return o;
        }
        finally
        {
            rms.Dispose();
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        Ones.Dispose();
    }
}
