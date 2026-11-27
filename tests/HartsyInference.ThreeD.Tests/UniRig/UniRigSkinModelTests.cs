using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.UniRig;

namespace HartsyInference.ThreeD.Tests.UniRig;

/// <summary>Tiny-config checks of the skin network's building blocks. Synthetic: they test mechanics and invariants, not fidelity.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class UniRigSkinModelTests
{
    private static (float[] Points, float[] Normals) Cloud(int count, int seed)
    {
        Random random = new(seed);
        float[] points = new float[3 * count];
        float[] normals = new float[3 * count];
        for (int i = 0; i < points.Length; i++)
        {
            points[i] = (float)(random.NextDouble() * 2 - 1);
            normals[i] = (float)(random.NextDouble() * 2 - 1);
        }

        return (points, normals);
    }

    [Fact]
    public void SparseConv_KernelFive_MatchesBruteForce()
    {
        Random random = new(8);
        int n = 60;
        int[] grid = new int[3 * n];
        HashSet<(int, int, int)> seen = [];
        for (int i = 0; i < n; i++)
        {
            (int, int, int) cell;
            do
            {
                cell = (random.Next(6), random.Next(6), random.Next(6));
            }
            while (!seen.Add(cell));
            (grid[3 * i], grid[3 * i + 1], grid[3 * i + 2]) = cell;
        }

        int cin = 3, cout = 4;
        using Tensor weight = UniRigTestWeights.Random(random, 0.5, cout, 5, 5, 5, cin);
        using Tensor features = UniRigTestWeights.Random(random, 1.0, n, cin);
        Tensor[] slices = SparseOps.ConvWeightSlices(weight);
        using CpuBackend backend = new();
        using Tensor result = UniRigSparseConv.Run(backend, features, grid, slices, 5);
        foreach (Tensor slice in slices)
        {
            slice.Dispose();
        }

        ReadOnlySpan<float> w = weight.AsReadOnlySpan<float>();
        ReadOnlySpan<float> f = features.AsReadOnlySpan<float>();
        ReadOnlySpan<float> r = result.AsReadOnlySpan<float>();
        Dictionary<(int, int, int), int> index = [];
        for (int i = 0; i < n; i++)
        {
            index[(grid[3 * i], grid[3 * i + 1], grid[3 * i + 2])] = i;
        }

        for (int o = 0; o < n; o++)
        {
            for (int co = 0; co < cout; co++)
            {
                double sum = 0;
                for (int kx = 0; kx < 5; kx++)
                {
                    for (int ky = 0; ky < 5; ky++)
                    {
                        for (int kz = 0; kz < 5; kz++)
                        {
                            if (!index.TryGetValue((grid[3 * o] + kx - 2, grid[3 * o + 1] + ky - 2, grid[3 * o + 2] + kz - 2), out int neighbour))
                            {
                                continue;
                            }

                            for (int ci = 0; ci < cin; ci++)
                            {
                                sum += w[(((co * 5 + kx) * 5 + ky) * 5 + kz) * cin + ci] * f[neighbour * cin + ci];
                            }
                        }
                    }
                }

                Assert.True(Math.Abs(sum - r[o * cout + co]) < 1e-4, $"point {o} channel {co}: {r[o * cout + co]} vs {sum}");
            }
        }
    }

    [Fact]
    public void PointTransformer_IsEquivariantToPointOrder_WithOneAttentionPatch()
    {
        Dictionary<string, Tensor> weights = UniRigSkinTestWeights.Point(UniRigSkinTestWeights.TinyPoint, "enc", 3);
        try
        {
            (float[] points, float[] normals) = Cloud(40, 5);
            // Distinct grid cells keep the sparse hash unambiguous, so only the order of points may differ.
            using CpuBackend backend = new();
            using UniRigPtv3Encoder encoder = new(backend, UniRigSkinTestWeights.TinyPoint, weights, "enc");
            using Tensor a = encoder.Encode(points, normals);
            Assert.Equal(new TensorShape(40, 16), a.Shape);
            Assert.All(a.AsReadOnlySpan<float>().ToArray(), v => Assert.True(float.IsFinite(v)));

            int[] perm = Enumerable.Range(0, 40).OrderBy(i => (i * 7919) % 40).ToArray();
            float[] p2 = new float[points.Length];
            float[] n2 = new float[normals.Length];
            for (int i = 0; i < 40; i++)
            {
                Array.Copy(points, 3 * perm[i], p2, 3 * i, 3);
                Array.Copy(normals, 3 * perm[i], n2, 3 * i, 3);
            }

            using Tensor b = encoder.Encode(p2, n2);
            ReadOnlySpan<float> fa = a.AsReadOnlySpan<float>();
            ReadOnlySpan<float> fb = b.AsReadOnlySpan<float>();
            for (int i = 0; i < 40; i++)
            {
                for (int c = 0; c < 16; c++)
                {
                    Assert.True(Math.Abs(fa[perm[i] * 16 + c] - fb[i * 16 + c]) < 1e-3f, $"point {perm[i]} channel {c}");
                }
            }
        }
        finally
        {
            UniRigTestWeights.Dispose(weights);
        }
    }

    [Fact]
    public void SkinModel_ProducesNonNegativeRowsSummingToOne_AndFollowsTheVoxelPrior()
    {
        Dictionary<string, Tensor> weights = UniRigSkinTestWeights.Skin(11);
        try
        {
            using UniRigCheckpoint checkpoint = UniRigCheckpoint.FromTensors(weights);
            (float[] points, float[] normals) = Cloud(70, 2);
            float[] joints = [0f, 0f, 0f, 0.3f, 0.1f, 0f, -0.3f, 0.2f, 0.1f];
            int[] parents = [-1, 0, 0];
            Random random = new(1);
            float[] prior = new float[70 * 3];
            for (int i = 0; i < 70; i++)
            {
                double a = random.NextDouble() + 0.01, b = random.NextDouble() + 0.01, c = random.NextDouble() + 0.01;
                double s = a + b + c;
                prior[3 * i] = (float)(a / s);
                prior[3 * i + 1] = (float)(b / s);
                prior[3 * i + 2] = (float)(c / s);
            }

            using CpuBackend backend = new();
            using UniRigSkinModel model = new(backend, checkpoint, UniRigSkinTestWeights.TinySkin, UniRigSkinTestWeights.TinyPoint, UniRigSkinTestWeights.TinyGlobal);
            float[] skin = model.Predict(points, normals, joints, parents, prior, seed: 1);
            Assert.Equal(70 * 3, skin.Length);
            for (int p = 0; p < 70; p++)
            {
                float total = 0f;
                for (int j = 0; j < 3; j++)
                {
                    float v = skin[p * 3 + j];
                    Assert.True(v >= 0f && float.IsFinite(v), $"point {p} joint {j}: {v}");
                    total += v;
                }

                Assert.Equal(1f, total, 4);
            }

            // A prior of zero for a joint (and its parent-summed mask) removes that joint everywhere.
            float[] zeroed = (float[])prior.Clone();
            for (int p = 0; p < 70; p++)
            {
                zeroed[3 * p + 2] = 0f;
            }

            float[] masked = model.Predict(points, normals, joints, parents, zeroed, seed: 1);
            for (int p = 0; p < 70; p++)
            {
                Assert.Equal(0f, masked[p * 3 + 2], 6);
            }
        }
        finally
        {
            UniRigTestWeights.Dispose(weights);
        }
    }

    [Fact]
    public void SkinModel_RejectsParentsThatDoNotPrecedeChildren()
    {
        Dictionary<string, Tensor> weights = UniRigSkinTestWeights.Skin(11);
        try
        {
            using UniRigCheckpoint checkpoint = UniRigCheckpoint.FromTensors(weights);
            using CpuBackend backend = new();
            using UniRigSkinModel model = new(backend, checkpoint, UniRigSkinTestWeights.TinySkin, UniRigSkinTestWeights.TinyPoint, UniRigSkinTestWeights.TinyGlobal);
            (float[] points, float[] normals) = Cloud(10, 2);
            Assert.Throws<ArgumentException>(() => model.Predict(points, normals, new float[6], [1, -1], new float[20], 0));
        }
        finally
        {
            UniRigTestWeights.Dispose(weights);
        }
    }
}
