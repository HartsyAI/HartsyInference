using HartsyInference.Core.Backends;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.TrellisV2;

/// <summary>Thin <see cref="IBackend"/> wrappers over the shape-VAE layout <c>[1, N, C]</c>, and the input checks both networks share.</summary>
internal static class ShapeVaeOps
{
    /// <summary>Upstream block norms use eps 1e-6; the encoder and decoder output norms use <c>F.layer_norm</c>'s 1e-5.</summary>
    internal const float BlockNormEps = 1e-6f;
    internal const float OutputNormEps = 1e-5f;

    internal static Tensor Alloc(int rows, int channels) => new(new TensorShape(1, rows, channels), DType.F32);

    internal static Tensor Linear(IBackend backend, Tensor input, Tensor weight, Tensor bias, int rows, int outChannels)
    {
        Tensor o = Alloc(rows, outChannels);
        backend.Linear(o, SparseOps.As3D(input), weight, bias);
        return o;
    }

    internal static Tensor LayerNorm(IBackend backend, Tensor input, Tensor weight, Tensor bias, int rows, int channels, float eps)
    {
        Tensor o = Alloc(rows, channels);
        backend.LayerNorm(o, SparseOps.As3D(input), weight, bias, eps);
        return o;
    }

    internal static Tensor LayerNormNoAffine(IBackend backend, Tensor input, int rows, int channels, float eps)
    {
        Tensor o = Alloc(rows, channels);
        backend.LayerNormNoAffine(o, SparseOps.As3D(input), eps);
        return o;
    }

    internal static Tensor Silu(IBackend backend, Tensor input, int rows, int channels)
    {
        Tensor o = Alloc(rows, channels);
        backend.Silu(o, SparseOps.As3D(input));
        return o;
    }

    internal static Tensor Add(IBackend backend, Tensor a, Tensor b, int rows, int channels)
    {
        Tensor o = Alloc(rows, channels);
        backend.Add(o, SparseOps.As3D(a), SparseOps.As3D(b));
        return o;
    }

    /// <summary>Checks a sparse input against a block's expectations; fails before any work is queued.</summary>
    internal static void RequireInput(SparseTensor x, int channels, string name)
    {
        if (x.Count == 0) throw new ArgumentException("Sparse input has no voxels.", name);
        if (x.Channels != channels) throw new ArgumentException($"Sparse input has {x.Channels} channels, expected {channels}.", name);
        if (x.Feats.DType != DType.F32) throw new NotSupportedException($"Shape VAE CPU reference runs in F32; '{name}' is {x.Feats.DType}.");
        SparseSpatialChannel.ValidateCoords(x.Coords, x.Resolution);
    }
}
