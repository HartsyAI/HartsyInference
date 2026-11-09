using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.Utonia;

namespace HartsyInference.ThreeD.Models.CoLa3D;

/// <summary>
/// The three weight sets of the CoLa3D scene pipeline, each converted to the names its module reads: the decomposer
/// by <see cref="CoLa3DKeyMap"/>, the shape VAE by its checkpoint keys, and Utonia by <see cref="UtoniaCheckpointKeys"/>.
/// Owns the loaders that back the tensors, so it must outlive the modules built from it.
/// </summary>
public sealed class CoLa3DWeights : IDisposable
{
    private readonly List<IDisposable> _loaders;
    private bool _disposed;

    internal CoLa3DWeights(
        IReadOnlyDictionary<string, Tensor> decomposer,
        IReadOnlyDictionary<string, Tensor> shapeEncoder,
        IReadOnlyDictionary<string, Tensor> shapeDecoder,
        UtoniaWeights utonia,
        List<IDisposable> loaders)
    {
        Decomposer = decomposer;
        ShapeEncoder = shapeEncoder;
        ShapeDecoder = shapeDecoder;
        Utonia = utonia;
        _loaders = loaders;
    }

    /// <summary>Decomposer parameters by name, as <see cref="CoLa3DDecomposer"/> expects.</summary>
    public IReadOnlyDictionary<string, Tensor> Decomposer { get; }

    /// <summary>Shape VAE encoder tensors by checkpoint key.</summary>
    public IReadOnlyDictionary<string, Tensor> ShapeEncoder { get; }

    /// <summary>Shape VAE decoder tensors by checkpoint key.</summary>
    public IReadOnlyDictionary<string, Tensor> ShapeDecoder { get; }

    /// <summary>Utonia encoder parameters by C# name.</summary>
    public UtoniaWeights Utonia { get; }

    /// <summary>Releases the checkpoint loaders and the tensors they back.</summary>
    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        foreach (IDisposable loader in _loaders)
            loader.Dispose();
        _loaders.Clear();
    }
}
