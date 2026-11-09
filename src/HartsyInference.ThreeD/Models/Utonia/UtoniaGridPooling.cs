using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Utonia;

/// <summary>
/// Grid pooling: projects features, takes the per-voxel max over each cell of the coarser grid, averages coordinates,
/// then applies LayerNorm and GELU. The output is serialized and sparsified, and keeps a link back to its parent.
/// </summary>
internal sealed class UtoniaGridPooling
{
    private const int CoordBits = 16;
    private const int BatchShift = 48;

    private readonly IBackend _backend;
    private readonly UtoniaLinear _proj;
    private readonly UtoniaLayerNorm _norm;
    private readonly int _stride;
    private readonly long _chunkBytes;

    public UtoniaGridPooling(IBackend backend, UtoniaLinear proj, UtoniaLayerNorm norm, int stride,
        long chunkBytes = UtoniaRowChunks.ChunkBytes)
    {
        _backend = backend;
        _proj = proj;
        _norm = norm;
        _stride = stride;
        _chunkBytes = chunkBytes;
    }

    /// <summary>Pools <paramref name="point"/> into the coarser level; the result is ordered by (batch, x, y, z).</summary>
    public UtoniaPoint Forward(UtoniaPoint point, UtoniaSerialOrder[] orders)
    {
        int n = point.Count;
        if (n == 0)
        {
            throw new HartsyInferenceException("Utonia grid pooling needs at least one point.");
        }

        int c = _proj.OutChannels;
        Tensor projected = UtoniaRowChunks.Map(point.Feat, c, UtoniaRowChunks.RowsPerChunk(_chunkBytes, Math.Max(c, point.Feat.Shape[1])),
            (src, res) => _proj.Forward(src, res));
        long[] keys = new long[n];
        for (int i = 0; i < n; i++)
        {
            int x = point.GridCoord[i * 3] / _stride, y = point.GridCoord[i * 3 + 1] / _stride, z = point.GridCoord[i * 3 + 2] / _stride;
            if (x >= 1 << CoordBits || y >= 1 << CoordBits || z >= 1 << CoordBits)
            {
                throw new HartsyInferenceException("Utonia pooled grid coordinate exceeds 16 bits.");
            }

            keys[i] = ((long)point.Batch[i] << BatchShift) | ((long)x << (2 * CoordBits)) | ((long)y << CoordBits) | (long)z;
        }

        int[] perm = UtoniaSerialization.StableArgSort(keys);
        int[] cluster = new int[n];
        int clusters = 0;
        for (int j = 0; j < n; j++)
        {
            int i = perm[j];
            if (j == 0 || keys[i] != keys[perm[j - 1]])
            {
                clusters++;
            }

            cluster[i] = clusters - 1;
        }

        int[] clusterBatch = new int[clusters];
        int[] clusterGrid = new int[clusters * 3];
        float[] coordSum = new float[clusters * 3];
        int[] counts = new int[clusters];
        Tensor pooled = new(new TensorShape(clusters, c), DType.F32);
        unsafe
        {
            float* src = (float*)projected.DataPointer, dst = (float*)pooled.DataPointer;
            for (int j = 0; j < n; j++)
            {
                int i = perm[j], k = cluster[i];
                bool first = j == 0 || cluster[perm[j - 1]] != k;
                for (int ch = 0; ch < c; ch++)
                {
                    float v = src[(long)i * c + ch];
                    dst[(long)k * c + ch] = first ? v : Math.Max(dst[(long)k * c + ch], v);
                }

                for (int a = 0; a < 3; a++)
                {
                    coordSum[k * 3 + a] += point.Coord[i * 3 + a];
                }

                counts[k]++;
                if (first)
                {
                    long key = keys[i];
                    clusterBatch[k] = (int)(key >> BatchShift);
                    clusterGrid[k * 3] = (int)((key >> (2 * CoordBits)) & 0xFFFF);
                    clusterGrid[k * 3 + 1] = (int)((key >> CoordBits) & 0xFFFF);
                    clusterGrid[k * 3 + 2] = (int)(key & 0xFFFF);
                }
            }
        }

        projected.Dispose();
        float[] coord = new float[clusters * 3];
        for (int k = 0; k < clusters; k++)
        {
            for (int a = 0; a < 3; a++)
            {
                coord[k * 3 + a] = coordSum[k * 3 + a] / counts[k];
            }
        }

        Tensor activated = UtoniaRowChunks.Map(pooled, c, UtoniaRowChunks.RowsPerChunk(_chunkBytes, c), (src, res) =>
        {
            using Tensor normed = new(src.Shape, DType.F32);
            _norm.Forward(src, normed);
            _backend.GeluErf(res, normed);
        });
        pooled.Dispose();

        UtoniaPoint next = new(activated, coord, clusterGrid, clusterBatch, point, cluster);
        next.Serialize(orders);
        next.Sparsify();
        return next;
    }
}
