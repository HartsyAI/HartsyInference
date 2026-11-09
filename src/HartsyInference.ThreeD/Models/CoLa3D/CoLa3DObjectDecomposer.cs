using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Pipelines.Stages;

namespace HartsyInference.ThreeD.Models.CoLa3D;

/// <summary>
/// CoLa3D object decomposition: runs <see cref="CoLa3DDecomposer"/> with learnable part queries, assigns each scene voxel to
/// its highest-scoring part, and returns each part's voxels of the scene latent as an object latent.
/// </summary>
/// <remarks>Object latents are the input latent rows, not decomposer features: the decomposer's 768-wide point features have
/// no head back to the 32-channel latent in the released weights. The part assignment is this stage's choice.</remarks>
public sealed class CoLa3DObjectDecomposer : IObjectDecomposer, IDisposable
{
    private readonly CoLa3DConfig _config;
    private readonly CoLa3DDecomposer _decomposer;
    private bool _disposed;

    /// <param name="config">Architecture the decomposer was built with.</param>
    /// <param name="decomposer">Built from the decomposer weights; this stage owns it.</param>
    public CoLa3DObjectDecomposer(CoLa3DConfig config, CoLa3DDecomposer decomposer)
    {
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _decomposer = decomposer ?? throw new ArgumentNullException(nameof(decomposer));
    }

    /// <inheritdoc/>
    public IReadOnlyList<SparseTensor> Decompose(IBackend backend, SparseTensor latent, SparseTensor features)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(latent);
        ArgumentNullException.ThrowIfNull(features);
        int n = latent.Count;
        if (features.Count != n)
            throw new HartsyInferenceException($"Utonia features cover {features.Count} voxels; the latent holds {n}.");

        Tensor aligned = AlignRows(latent.Coords, features, _config.UtoniaFeatureDim);
        try
        {
            using CoLa3DOutput output = _decomposer.Forward(backend, latent, aligned);
            int parts = (int)output.MaskLogits.Shape[1];
            int[] owners = CoLa3DObjectSplit.AssignOwners(output.MaskLogits.AsReadOnlySpan<float>(), n, parts);
            List<SparseTensor> objects = [];
            try
            {
                foreach (int[] rows in CoLa3DObjectSplit.GroupRows(owners, parts))
                    objects.Add(Gather(latent, rows));
            }
            catch
            {
                foreach (SparseTensor part in objects)
                    part.Feats.Dispose();
                throw;
            }
            return objects;
        }
        finally
        {
            aligned.Dispose();
        }
    }

    /// <summary>Releases the decomposer and its weights.</summary>
    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        _decomposer.Dispose();
    }

    // Puts the Utonia rows in the latent's voxel order by coordinate, so the two inputs need not share a sort order.
    private static Tensor AlignRows(int[] latentCoords, SparseTensor features, int channels)
    {
        int n = latentCoords.Length / 4;
        Dictionary<long, int> rowOf = new(features.Count);
        for (int r = 0; r < features.Count; r++)
        {
            if (!rowOf.TryAdd(Key(features.Coords, r * 4), r))
                throw new HartsyInferenceException("Utonia features hold a duplicate voxel coordinate.");
        }

        Tensor aligned = new(new TensorShape(n, channels), DType.F32);
        ReadOnlySpan<float> src = features.Feats.AsReadOnlySpan<float>();
        Span<float> dst = aligned.AsSpan<float>();
        for (int i = 0; i < n; i++)
        {
            if (!rowOf.TryGetValue(Key(latentCoords, i * 4), out int r))
            {
                aligned.Dispose();
                throw new HartsyInferenceException($"Utonia features have no row for latent voxel {i}.");
            }

            src.Slice(r * channels, channels).CopyTo(dst.Slice(i * channels, channels));
        }

        return aligned;
    }

    private static SparseTensor Gather(SparseTensor latent, int[] rows)
    {
        int channels = latent.Channels;
        Tensor feats = new(new TensorShape(rows.Length, channels), DType.F32);
        ReadOnlySpan<float> src = latent.Feats.AsReadOnlySpan<float>();
        Span<float> dst = feats.AsSpan<float>();
        int[] coords = new int[rows.Length * 4];
        for (int k = 0; k < rows.Length; k++)
        {
            src.Slice(rows[k] * channels, channels).CopyTo(dst.Slice(k * channels, channels));
            Array.Copy(latent.Coords, rows[k] * 4, coords, k * 4, 4);
        }

        return new SparseTensor(feats, coords, latent.Resolution);
    }

    // Packs (batch, x, y, z) into one key; each component is below 2^16 for every grid this pipeline accepts.
    private static long Key(int[] coords, int at) =>
        (((long)coords[at] * 65536L + coords[at + 1]) * 65536L + coords[at + 2]) * 65536L + coords[at + 3];
}
