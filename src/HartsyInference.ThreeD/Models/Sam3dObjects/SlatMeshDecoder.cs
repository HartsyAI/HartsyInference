using HartsyInference.Core.Backends;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.Trellis;

namespace HartsyInference.ThreeD.Models.Sam3dObjects;

/// <summary>
/// The feature stage of SAM 3D's mesh decoder (<c>SLatMeshDecoder</c>): the swin trunk shared with the Gaussian decoder, two subdivide blocks (768 to 192 and 192 to 96 channels, each doubling
/// the resolution) and a linear head of <see cref="OutChannels"/> channels per voxel of the 4 x finer grid. The head's layout is, per voxel, 8 corner SDF values, 8 x 3 corner
/// deformations, 21 FlexiCubes weights and 8 x 6 corner colours; turning it into a mesh is the FlexiCubes extraction, which this class does not do.
/// </summary>
public sealed class SlatMeshDecoder
{
    /// <summary>Channels of the head with colours (<c>use_color: true</c>): 8 + 24 + 21 + 48.</summary>
    public const int OutChannels = 101;

    private const int TrunkChannels = 768;

    private readonly SlatGaussianDecoder _trunk = new();
    private SamSparseSubdivideBlock? _first, _second;
    private Tensor? _headWeight, _headBias;

    /// <summary>Reads the decoder's tensors (<c>input_layer</c>, <c>blocks.i</c>, <c>upsample.0</c>, <c>upsample.1</c>, <c>out_layer</c>), all F32 except the trunk's GEMM weights.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        _trunk.LoadTrunkWeights(weights);
        _first = new SamSparseSubdivideBlock(weights, "upsample.0", TrunkChannels, TrunkChannels / 4);
        _second = new SamSparseSubdivideBlock(weights, "upsample.1", TrunkChannels / 4, TrunkChannels / 8);
        _headWeight = SamWeights.Require(weights, "out_layer.weight", OutChannels, TrunkChannels / 8);
        _headBias = SamWeights.Require(weights, "out_layer.bias", OutChannels);
    }

    /// <summary>Every tensor the decoder reads, for <c>IBackend.PreloadWeights</c> and <c>FreeWeights</c>.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _trunk.EnumerateWeights())
            yield return t;
        if (_first is null)
            yield break;
        foreach (Tensor t in _first.Weights())
            yield return t;
        foreach (Tensor t in _second!.Weights())
            yield return t;
        yield return _headWeight!;
        yield return _headBias!;
    }

    /// <summary>
    /// The per-voxel features of the mesh head: <paramref name="x"/> is the denormalised SLAT (features <c>[1, N, 8]</c>, 64-grid coordinates); the result has <c>64 N</c> voxels on the
    /// 256-grid, features <c>[1, 64 N, 101]</c>, owned by the caller.
    /// </summary>
    public SparseTensor Forward(IBackend backend, SparseTensor x)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (_first is null)
            throw new InvalidOperationException("Load the decoder's weights first.");
        SparseTensor trunk = _trunk.Trunk(backend, x);
        SparseTensor coarse = _first.Forward(backend, trunk);
        trunk.Feats.Dispose();
        SparseTensor fine = _second!.Forward(backend, coarse);
        coarse.Feats.Dispose();
        Tensor head = new(new TensorShape(1, fine.Count, OutChannels), DType.F32);
        backend.Linear(head, SparseOps.As3D(fine.Feats), _headWeight!, _headBias!);
        fine.Feats.Dispose();
        return fine.Replace(head);
    }
}
