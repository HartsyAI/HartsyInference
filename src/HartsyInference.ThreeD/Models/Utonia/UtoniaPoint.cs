using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Utonia;

/// <summary>
/// One point set in a Utonia level: features plus the geometry the encoder reads (Pointcept's Point structure).
/// Samples are contiguous in <see cref="Batch"/>, which is non-decreasing.
/// </summary>
public sealed class UtoniaPoint : IDisposable
{
    private Tensor? _feat;
    private bool _disposed;

    public UtoniaPoint(Tensor feat, float[] coord, int[] gridCoord, int[] batch, UtoniaPoint? poolingParent = null, int[]? poolingInverse = null)
    {
        int n = batch.Length;
        if (feat.Shape.Rank != 2 || feat.Shape[0] != n)
        {
            throw new HartsyInferenceException($"Utonia point features must be [{n}, C], got {feat.Shape}.");
        }

        if (coord.Length != n * 3 || gridCoord.Length != n * 3)
        {
            throw new HartsyInferenceException("Utonia point coords and grid coords need three values per point.");
        }

        for (int i = 1; i < n; i++)
        {
            if (batch[i] < batch[i - 1])
            {
                throw new HartsyInferenceException("Utonia points must be grouped by non-decreasing batch index.");
            }
        }

        _feat = feat;
        Coord = coord;
        GridCoord = gridCoord;
        Batch = batch;
        PoolingParent = poolingParent;
        PoolingInverse = poolingInverse;
    }

    /// <summary>Features <c>[N, C]</c>, owned by this point.</summary>
    public Tensor Feat => _feat ?? throw new InvalidOperationException("Utonia point features were taken.");

    public float[] Coord { get; }
    public int[] GridCoord { get; }
    public int[] Batch { get; }

    /// <summary>The finer point this one was pooled from, when it is a pooled level.</summary>
    public UtoniaPoint? PoolingParent { get; }

    /// <summary>For each parent point, its row in this level.</summary>
    public int[]? PoolingInverse { get; }

    /// <summary>Serialization of this point set. Set by <see cref="Serialize"/>.</summary>
    public UtoniaSerialized? Serialized { get; private set; }

    /// <summary>Voxel coords <c>(batch, x, y, z)</c> per point, the rulebook key for submanifold conv. Set by <see cref="Sparsify"/>.</summary>
    public int[]? SparseCoords { get; private set; }

    /// <summary>Grid extent of <see cref="SparseCoords"/>; the largest coordinate plus one.</summary>
    public int Resolution { get; private set; }

    public int Count => Batch.Length;

    /// <summary>Points per sample, indexed by batch id from zero, zeros included.</summary>
    public int[] SampleCounts()
    {
        int samples = Count == 0 ? 0 : Batch[Count - 1] + 1;
        int[] counts = new int[samples];
        for (int i = 0; i < Count; i++)
        {
            counts[Batch[i]]++;
        }

        return counts;
    }

    public void Serialize(UtoniaSerialOrder[] orders)
    {
        Serialized = UtoniaSerialization.Serialize(GridCoord, Batch, orders);
    }

    public void Sparsify()
    {
        int n = Count;
        int[] coords = new int[n * 4];
        int max = 0;
        for (int i = 0; i < n; i++)
        {
            coords[i * 4] = Batch[i];
            for (int a = 0; a < 3; a++)
            {
                coords[i * 4 + 1 + a] = GridCoord[i * 3 + a];
                max = Math.Max(max, GridCoord[i * 3 + a]);
            }
        }

        SparseCoords = coords;
        Resolution = max + 1;
    }

    /// <summary>The sparse view used by submanifold convolution; requires <see cref="Sparsify"/> first.</summary>
    public SparseTensor AsSparse(Tensor feat)
    {
        if (SparseCoords is null)
        {
            throw new InvalidOperationException("Utonia point must be sparsified before sparse convolution.");
        }

        return new SparseTensor(feat, SparseCoords, Resolution);
    }

    /// <summary>Replaces the features and releases the previous tensor. Callers must not keep a reference to it.</summary>
    public void ReplaceFeat(Tensor next)
    {
        Tensor? old = _feat;
        _feat = next;
        old?.Dispose();
    }

    /// <summary>Hands the features to the caller; this point no longer owns them.</summary>
    public Tensor TakeFeat()
    {
        Tensor feat = Feat;
        _feat = null;
        return feat;
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _feat?.Dispose();
        _feat = null;
    }
}
