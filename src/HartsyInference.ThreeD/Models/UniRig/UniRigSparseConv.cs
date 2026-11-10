using HartsyInference.Core.Backends;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.UniRig;

/// <summary>Submanifold 3D convolution of any odd kernel size over a point set on an integer grid: the output has the input's points, neighbours come from a coordinate hash.</summary>
/// <remarks>Duplicate grid cells resolve to the last point inserted, which is what spconv's hash does. The k=3 convolutions of the PTv3 blocks use <see cref="SparseOps"/>; this one exists for the k=5 stem.</remarks>
internal static class UniRigSparseConv
{
    private const long ChunkBytes = 64L << 20;

    /// <param name="features"><c>[N, Cin]</c> features.</param>
    /// <param name="gridCoord">Grid cell per point, length <c>3 * N</c>.</param>
    /// <param name="slices">Per-offset <c>[Cout, Cin]</c> weights from <see cref="SparseOps.ConvWeightSlices"/>, <c>k³</c> of them in (x, y, z) order.</param>
    /// <param name="kernel">Odd kernel size.</param>
    public static Tensor Run(IBackend backend, Tensor features, int[] gridCoord, Tensor[] slices, int kernel)
    {
        int n = (int)features.Shape[0];
        int cin = (int)features.Shape[1];
        int cout = (int)slices[0].Shape[0];
        int half = kernel / 2;
        Dictionary<(int, int, int), int> hash = new(n);
        for (int i = 0; i < n; i++)
        {
            hash[(gridCoord[3 * i], gridCoord[3 * i + 1], gridCoord[3 * i + 2])] = i;
        }

        Tensor output = new(new TensorShape(1, n, cout), DType.F32);
        backend.Fill(output, 0f);
        using Tensor input = features.Reshape(new TensorShape(1, n, cin));
        int chunkRows = (int)Math.Max(1, ChunkBytes / (sizeof(float) * Math.Max(cin, cout)));
        int[] inBuffer = new int[n];
        int[] outBuffer = new int[n];
        for (int kx = 0; kx < kernel; kx++)
        {
            for (int ky = 0; ky < kernel; ky++)
            {
                for (int kz = 0; kz < kernel; kz++)
                {
                    int count = 0;
                    for (int o = 0; o < n; o++)
                    {
                        int x = gridCoord[3 * o] + kx - half;
                        int y = gridCoord[3 * o + 1] + ky - half;
                        int z = gridCoord[3 * o + 2] + kz - half;
                        if (x >= 0 && y >= 0 && z >= 0 && hash.TryGetValue((x, y, z), out int neighbour))
                        {
                            inBuffer[count] = neighbour;
                            outBuffer[count] = o;
                            count++;
                        }
                    }

                    Tensor slice = slices[(kx * kernel + ky) * kernel + kz];
                    for (int start = 0; start < count; start += chunkRows)
                    {
                        int rows = Math.Min(chunkRows, count - start);
                        using Tensor inIndex = IndexTensor.Rows(inBuffer.AsSpan(start, rows));
                        using Tensor outIndex = IndexTensor.Rows(outBuffer.AsSpan(start, rows));
                        using Tensor gathered = new(new TensorShape(1, rows, cin), DType.F32);
                        backend.GatherRows(gathered, input, inIndex);
                        using Tensor product = new(new TensorShape(1, rows, cout), DType.F32);
                        backend.Linear(product, gathered, slice, null);
                        backend.ScatterAddRows(output, product, outIndex);
                        backend.Sync();
                    }
                }
            }
        }

        Tensor flat = new(new TensorShape(n, cout), DType.F32);
        output.AsSpan<float>().CopyTo(flat.AsSpan<float>());
        output.Dispose();
        return flat;
    }
}
