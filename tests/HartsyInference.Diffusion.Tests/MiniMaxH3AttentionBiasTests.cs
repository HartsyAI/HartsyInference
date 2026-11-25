using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The Duo additive bias through the H3 block's dense attention, on a tiny geometry. These are self-consistency checks:
/// they show that the bias hides the rows it should, not parity with upstream Duo.</summary>
public sealed class MiniMaxH3AttentionBiasTests
{
    private const int Seq = 4, Hidden = 8, Heads = 2, HeadDim = 4, Inner = Heads * HeadDim;
    private const string Prefix = "blocks.0.attn";

    [Fact]
    public void CausalBiasKeepsEarlierRowsIndependentOfLaterRows()
    {
        using MiniMaxH3Transformer transformer = Transformer();
        using IBackend backend = new CpuBackend();
        float[] input = Values(1);
        float[] laterPerturbed = (float[])input.Clone();
        for (int i = (Seq - 1) * Hidden; i < Seq * Hidden; i++)
        {
            laterPerturbed[i] += 0.5f;
        }

        float[] baseline = Run(transformer, backend, input, Causal());
        float[] perturbed = Run(transformer, backend, laterPerturbed, Causal());

        // Rows 0..Seq-2 read only keys at or before themselves, so the last row cannot change them.
        AssertRowsEqual(baseline, perturbed, 0, Seq - 1);
        // The last row does change, so the perturbation is real.
        Assert.True(MaxDiff(baseline, perturbed, Seq - 1, Seq) > 1e-4f);
    }

    [Fact]
    public void CausalBiasLetsLaterRowsSeeAnEarlierRow()
    {
        using MiniMaxH3Transformer transformer = Transformer();
        using IBackend backend = new CpuBackend();
        float[] input = Values(2);
        float[] firstPerturbed = (float[])input.Clone();
        for (int i = 0; i < Hidden; i++)
        {
            firstPerturbed[i] += 0.5f;
        }

        float[] baseline = Run(transformer, backend, input, Causal());
        float[] perturbed = Run(transformer, backend, firstPerturbed, Causal());

        // Every later row may read row 0, so each one moves.
        for (int row = 1; row < Seq; row++)
        {
            Assert.True(MaxDiff(baseline, perturbed, row, row + 1) > 1e-4f, $"row {row} did not see row 0.");
        }
    }

    [Fact]
    public void WithoutABiasTheEarlierRowsSeeTheLaterRow()
    {
        // Negative control for the first test: with no bias, row 0 reads every key, so the last row moves it.
        using MiniMaxH3Transformer transformer = Transformer();
        using IBackend backend = new CpuBackend();
        float[] input = Values(1);
        float[] laterPerturbed = (float[])input.Clone();
        for (int i = (Seq - 1) * Hidden; i < Seq * Hidden; i++)
        {
            laterPerturbed[i] += 0.5f;
        }

        float[] baseline = Run(transformer, backend, input, bias: null);
        float[] perturbed = Run(transformer, backend, laterPerturbed, bias: null);

        Assert.True(MaxDiff(baseline, perturbed, 0, Seq - 1) > 1e-4f);
    }

    [Fact]
    public void BiasOfTheWrongShapeIsRefused()
    {
        using MiniMaxH3Transformer transformer = Transformer();
        using IBackend backend = new CpuBackend();
        using Tensor input = MakeTensor(new TensorShape(Seq, Hidden), Values(3));
        using Tensor wrong = new(new TensorShape(Seq, Seq - 1), DType.F32);

        Assert.Throws<ArgumentException>(() => transformer.Attention(backend, input, Prefix, null, null, wrong));
    }

    private static MiniMaxH3Transformer Transformer()
    {
        MiniMaxH3Transformer transformer = new(new MiniMaxH3Config
        {
            HiddenSize = Hidden,
            NumLayers = 1,
            TokenRefinerNumLayers = 0,
            NumAttentionHeads = Heads,
            AttentionHeadDim = HeadDim,
            FfnHiddenSize = 16,
            LatentsDim = 24,
            HasAudio = false,
            TextDim = 5,
            OutputHeadCount = 4,
        });
        transformer.LoadWeights(new Dictionary<string, Tensor>
        {
            ["video_patch_proj.weight"] = MakeTensor(new TensorShape(Hidden, 96), Values(10, 96 * Hidden)),
            ["final_layer.video_out.weight"] = MakeTensor(new TensorShape(576, Hidden), Values(11, 576 * Hidden)),
            [$"{Prefix}.qkv_proj.weight"] = MakeTensor(new TensorShape(3 * Inner, Hidden), Values(12, 3 * Inner * Hidden)),
            [$"{Prefix}.q_norm.weight"] = MakeTensor(new TensorShape(HeadDim), Ones(HeadDim)),
            [$"{Prefix}.k_norm.weight"] = MakeTensor(new TensorShape(HeadDim), Ones(HeadDim)),
            [$"{Prefix}.out_proj.weight"] = MakeTensor(new TensorShape(Hidden, Inner), Values(13, Hidden * Inner)),
        });
        return transformer;
    }

    private static float[] Run(MiniMaxH3Transformer transformer, IBackend backend, float[] input, float[]? bias)
    {
        using Tensor x = MakeTensor(new TensorShape(Seq, Hidden), input);
        using Tensor? biasTensor = bias is null ? null : MakeTensor(new TensorShape(Seq, Seq), bias);
        using Tensor y = transformer.Attention(backend, x, Prefix, null, null, biasTensor);
        return y.AsSpan<float>().ToArray();
    }

    /// <summary>Row-major [Seq, Seq] causal bias: 0 where a query may read a key, negative infinity above the diagonal.</summary>
    private static float[] Causal()
    {
        float[] bias = new float[Seq * Seq];
        for (int q = 0; q < Seq; q++)
        {
            for (int k = 0; k < Seq; k++)
            {
                bias[q * Seq + k] = k <= q ? 0f : float.NegativeInfinity;
            }
        }
        return bias;
    }

    private static Tensor MakeTensor(TensorShape shape, float[] values)
    {
        Tensor tensor = new(shape, DType.F32);
        values.AsSpan().CopyTo(tensor.AsSpan<float>());
        return tensor;
    }

    private static float[] Values(int seed, int count = Seq * Hidden)
    {
        float[] data = new float[count];
        for (int i = 0; i < count; i++)
        {
            data[i] = MathF.Sin(seed * 1.7f + i * 0.37f);
        }
        return data;
    }

    private static float[] Ones(int count)
    {
        float[] data = new float[count];
        Array.Fill(data, 1f);
        return data;
    }

    private static float MaxDiff(float[] a, float[] b, int fromRow, int toRow)
    {
        float max = 0f;
        for (int i = fromRow * Hidden; i < toRow * Hidden; i++)
        {
            max = MathF.Max(max, MathF.Abs(a[i] - b[i]));
        }
        return max;
    }

    private static void AssertRowsEqual(float[] a, float[] b, int fromRow, int toRow)
    {
        Assert.True(MaxDiff(a, b, fromRow, toRow) <= 1e-6f, $"rows [{fromRow}, {toRow}) differ by {MaxDiff(a, b, fromRow, toRow)}.");
    }
}
