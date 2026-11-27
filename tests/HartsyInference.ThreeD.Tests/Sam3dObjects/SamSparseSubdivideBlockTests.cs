using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.Sam3dObjects;

namespace HartsyInference.ThreeD.Tests.Sam3dObjects;

public sealed class SamSparseSubdivideBlockTests
{
    private static Tensor Filled(Random rng, double scale, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        foreach (ref float v in t.AsSpan<float>())
            v = (float)((rng.NextDouble() - 0.5) * scale);
        return t;
    }

    private static Dictionary<string, Tensor> Weights(Random rng, int channels, int outChannels)
    {
        Dictionary<string, Tensor> w = new()
        {
            ["b.act_layers.0.weight"] = Filled(rng, 1.0, channels),
            ["b.act_layers.0.bias"] = Filled(rng, 0.5, channels),
            ["b.out_layers.0.conv.weight"] = Filled(rng, 0.4, outChannels, 3, 3, 3, channels),
            ["b.out_layers.0.conv.bias"] = Filled(rng, 0.3, outChannels),
            ["b.out_layers.1.weight"] = Filled(rng, 1.0, outChannels),
            ["b.out_layers.1.bias"] = Filled(rng, 0.5, outChannels),
            ["b.out_layers.3.conv.weight"] = Filled(rng, 0.4, outChannels, 3, 3, 3, outChannels),
            ["b.out_layers.3.conv.bias"] = Filled(rng, 0.3, outChannels),
        };
        if (outChannels != channels)
        {
            w["b.skip_connection.conv.weight"] = Filled(rng, 0.6, outChannels, 1, 1, 1, channels);
            w["b.skip_connection.conv.bias"] = Filled(rng, 0.3, outChannels);
        }

        return w;
    }

    private static double[,] GroupNorm(double[,] x, float[] weight, float[] bias)
    {
        int n = x.GetLength(0), c = x.GetLength(1), perGroup = c / 32;
        double[,] y = new double[n, c];
        for (int g = 0; g < 32; g++)
        {
            double sum = 0, squares = 0;
            for (int i = 0; i < n; i++)
            {
                for (int k = 0; k < perGroup; k++)
                {
                    double v = x[i, g * perGroup + k];
                    sum += v;
                    squares += v * v;
                }
            }

            double count = (double)n * perGroup, mean = sum / count, variance = squares / count - mean * mean;
            for (int i = 0; i < n; i++)
            {
                for (int k = 0; k < perGroup; k++)
                {
                    int ch = g * perGroup + k;
                    y[i, ch] = (x[i, ch] - mean) / Math.Sqrt(variance + 1e-5) * weight[ch] + bias[ch];
                }
            }
        }

        return y;
    }

    private static double Silu(double v) => v / (1 + Math.Exp(-v));

    // Submanifold conv of [n, cin] features at (x, y, z) coordinates with weight [cout, 3, 3, 3, cin] (spconv layout, cross-correlation).
    private static double[,] Conv(double[,] x, int[][] xyz, float[] weight, float[] bias, int cin, int cout)
    {
        int n = x.GetLength(0);
        Dictionary<(int, int, int), int> index = [];
        for (int i = 0; i < n; i++)
            index[(xyz[i][0], xyz[i][1], xyz[i][2])] = i;
        double[,] y = new double[n, cout];
        for (int i = 0; i < n; i++)
        {
            for (int o = 0; o < cout; o++)
            {
                double acc = bias[o];
                for (int kx = 0; kx < 3; kx++)
                {
                    for (int ky = 0; ky < 3; ky++)
                    {
                        for (int kz = 0; kz < 3; kz++)
                        {
                            if (!index.TryGetValue((xyz[i][0] + kx - 1, xyz[i][1] + ky - 1, xyz[i][2] + kz - 1), out int j))
                                continue;
                            for (int ci = 0; ci < cin; ci++)
                                acc += weight[(((o * 3 + kx) * 3 + ky) * 3 + kz) * cin + ci] * x[j, ci];
                        }
                    }
                }

                y[i, o] = acc;
            }
        }

        return y;
    }

    private static float[] Values(Tensor t) => t.AsReadOnlySpan<float>().ToArray();

    [Theory]
    [InlineData(32, 32)]
    [InlineData(32, 64)]
    public void Forward_MatchesTheNaiveReference(int channels, int outChannels)
    {
        Random rng = new(channels * 7 + outChannels);
        Dictionary<string, Tensor> w = Weights(rng, channels, outChannels);
        // Four voxels, three of them neighbours of the first, so the children of different parents touch each other.
        int[][] coarse = [[2, 3, 3], [3, 3, 3], [2, 4, 3], [5, 1, 1]];
        int n = coarse.Length;
        int[] coords = new int[n * 4];
        for (int i = 0; i < n; i++)
        {
            coords[i * 4 + 1] = coarse[i][0];
            coords[i * 4 + 2] = coarse[i][1];
            coords[i * 4 + 3] = coarse[i][2];
        }

        using Tensor feats = Filled(rng, 4.0, 1, n, channels);
        using CpuBackend backend = new();
        SamSparseSubdivideBlock block = new(w, "b", channels, outChannels);

        SparseTensor result = block.Forward(backend, new SparseTensor(feats, coords, 4));

        // Reference.
        double[,] x = new double[n, channels];
        float[] f = Values(feats);
        for (int i = 0; i < n; i++)
        {
            for (int c = 0; c < channels; c++)
                x[i, c] = f[i * channels + c];
        }

        double[,] normed = GroupNorm(x, Values(w["b.act_layers.0.weight"]), Values(w["b.act_layers.0.bias"]));
        int m = n * 8;
        int[][] fine = new int[m][];
        double[,] activated = new double[m, channels], skipInput = new double[m, channels];
        for (int i = 0; i < n; i++)
        {
            for (int k = 0; k < 8; k++)
            {
                int row = i * 8 + k;
                fine[row] = [coarse[i][0] * 2 + ((k >> 2) & 1), coarse[i][1] * 2 + ((k >> 1) & 1), coarse[i][2] * 2 + (k & 1)];
                for (int c = 0; c < channels; c++)
                {
                    activated[row, c] = Silu(normed[i, c]);
                    skipInput[row, c] = x[i, c];
                }
            }
        }

        double[,] c1 = Conv(activated, fine, Values(w["b.out_layers.0.conv.weight"]), Values(w["b.out_layers.0.conv.bias"]), channels, outChannels);
        double[,] n2 = GroupNorm(c1, Values(w["b.out_layers.1.weight"]), Values(w["b.out_layers.1.bias"]));
        double[,] a2 = new double[m, outChannels];
        for (int i = 0; i < m; i++)
        {
            for (int c = 0; c < outChannels; c++)
                a2[i, c] = Silu(n2[i, c]);
        }

        double[,] c2 = Conv(a2, fine, Values(w["b.out_layers.3.conv.weight"]), Values(w["b.out_layers.3.conv.bias"]), outChannels, outChannels);

        Assert.Equal(m, result.Count);
        Assert.Equal(8, result.Resolution);
        Assert.Equal(outChannels, result.Channels);
        float[] got = Values(result.Feats);
        float[] skipWeight = outChannels != channels ? Values(w["b.skip_connection.conv.weight"]) : [];
        float[] skipBias = outChannels != channels ? Values(w["b.skip_connection.conv.bias"]) : [];
        for (int i = 0; i < m; i++)
        {
            Assert.Equal(fine[i][0], result.Coords[i * 4 + 1]);
            Assert.Equal(fine[i][1], result.Coords[i * 4 + 2]);
            Assert.Equal(fine[i][2], result.Coords[i * 4 + 3]);
            for (int o = 0; o < outChannels; o++)
            {
                double skip;
                if (outChannels == channels)
                {
                    skip = skipInput[i, o];
                }
                else
                {
                    skip = skipBias[o];
                    for (int c = 0; c < channels; c++)
                        skip += skipWeight[o * channels + c] * skipInput[i, c];
                }

                double expected = c2[i, o] + skip;
                Assert.True(Math.Abs(expected - got[i * outChannels + o]) < 2e-3 * (1 + Math.Abs(expected)), $"[{i},{o}] expected {expected} got {got[i * outChannels + o]}");
            }
        }
    }

    [Fact]
    public void Subdivide_ListsTheEightChildrenInRowMajorOrderAndCopiesTheFeatures()
    {
        int[] coords = [0, 1, 2, 3, 0, 4, 5, 6];
        using Tensor feats = new(new TensorShape(1, 2, 2), DType.F32);
        float[] values = [1f, 2f, 3f, 4f];
        for (int i = 0; i < 4; i++)
            feats.AsSpan<float>()[i] = values[i];

        SparseTensor result = SamSparseSubdivideBlock.Subdivide(new SparseTensor(feats, coords, 8));

        Assert.Equal(16, result.Resolution);
        Assert.Equal(16, result.Count);
        // Voxel 0 (1, 2, 3): children (2 + dx, 4 + dy, 6 + dz) with k = 4 dx + 2 dy + dz.
        Assert.Equal([0, 2, 4, 6], result.Coords[0..4]);
        Assert.Equal([0, 2, 4, 7], result.Coords[4..8]);
        Assert.Equal([0, 2, 5, 6], result.Coords[8..12]);
        Assert.Equal([0, 3, 4, 6], result.Coords[16..20]);
        Assert.Equal([0, 3, 5, 7], result.Coords[28..32]);
        // Voxel 1 starts at row 8.
        Assert.Equal([0, 8, 10, 12], result.Coords[32..36]);
        float[] got = result.Feats.AsReadOnlySpan<float>().ToArray();
        for (int k = 0; k < 8; k++)
        {
            Assert.Equal([1f, 2f], got[(k * 2)..(k * 2 + 2)]);
            Assert.Equal([3f, 4f], got[((8 + k) * 2)..((8 + k) * 2 + 2)]);
        }
    }

    [Fact]
    public void Constructor_RejectsChannelCountsThatTheGroupNormCannotSplit()
    {
        Dictionary<string, Tensor> w = Weights(new Random(1), 32, 32);
        Assert.Throws<ArgumentException>(() => new SamSparseSubdivideBlock(w, "b", 30, 30));
    }

    [Fact]
    public void SlatMeshDecoder_EnumeratesNothingBeforeItsWeightsAreLoadedAndRejectsAForwardPass()
    {
        SlatMeshDecoder decoder = new();

        Assert.Empty(decoder.EnumerateWeights());
        using CpuBackend backend = new();
        using Tensor feats = new(new TensorShape(1, 1, 8), DType.F32);
        Assert.Throws<InvalidOperationException>(() => decoder.Forward(backend, new SparseTensor(feats, [0, 0, 0, 0], 64)));
    }
}
