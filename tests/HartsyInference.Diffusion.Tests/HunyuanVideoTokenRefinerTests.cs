using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Utilities;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The text refiner's batch handling. <c>SyntheticSmoke</c>: the refiner's gated residuals take a gate per batch row, broadcast over that row's
/// tokens (<c>GatedResidualLastDim</c>). A batch of two must give each row the same result as that row run alone, within a float tolerance.
/// The weights are random and small, so the check is about the batch axis, not the values.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class HunyuanVideoTokenRefinerTests
{
    private const int TextDim = 24;
    private const int Hidden = 32;
    private const int Heads = 2;
    private const int Blocks = 2;
    private const int Length = 5;
    private const int MlpDim = 4 * Hidden;
    private const int TimestepDim = 256;
    private const float Tolerance = 1e-5f;

    [Fact]
    public void BatchOfTwoMatchesEachRowRunAlone()
    {
        using CpuBackend backend = new();
        HunyuanVideoTokenRefiner refiner = new(TextDim, Hidden, Heads, Blocks);
        refiner.LoadWeights(RandomWeights());

        using Tensor text = TorchCompatibleNoise.Randn(new TensorShape(2, Length, TextDim), 9);
        using Tensor both = refiner.Forward(backend, text, 500f);
        Assert.Equal(new TensorShape(2, Length, Hidden), both.Shape);

        int rowElements = Length * TextDim;
        int outElements = Length * Hidden;
        for (int b = 0; b < 2; b++)
        {
            using Tensor single = new(new TensorShape(1, Length, TextDim), DType.F32);
            text.AsReadOnlySpan<float>().Slice(b * rowElements, rowElements).CopyTo(single.AsSpan<float>());
            using Tensor alone = refiner.Forward(backend, single, 500f);

            ReadOnlySpan<float> expected = alone.AsReadOnlySpan<float>();
            ReadOnlySpan<float> actual = both.AsReadOnlySpan<float>().Slice(b * outElements, outElements);
            float max = 0f;
            for (int i = 0; i < outElements; i++) max = Math.Max(max, Math.Abs(expected[i] - actual[i]));
            Assert.True(max <= Tolerance, $"row {b}: max abs difference {max:E3} exceeds {Tolerance:E0}.");
        }
    }

    /// <summary>Random weights under the <c>txt_in</c> names the refiner reads, scaled to 0.1 so activations stay in a moderate range.</summary>
    private static Dictionary<string, Tensor> RandomWeights()
    {
        Dictionary<string, Tensor> w = new(StringComparer.Ordinal);
        long seed = 100;
        void Add(string name, params int[] dims)
        {
            Tensor t = dims.Length == 1
                ? TorchCompatibleNoise.Randn(new TensorShape(dims[0]), seed++)
                : TorchCompatibleNoise.Randn(new TensorShape(dims[0], dims[1]), seed++);
            Span<float> s = t.AsSpan<float>();
            for (int i = 0; i < s.Length; i++) s[i] *= 0.1f;
            w[$"txt_in.{name}"] = t;
        }

        Add("input_embedder.weight", Hidden, TextDim);
        Add("input_embedder.bias", Hidden);
        Add("t_embedder.in_layer.weight", Hidden, TimestepDim);
        Add("t_embedder.in_layer.bias", Hidden);
        Add("t_embedder.out_layer.weight", Hidden, Hidden);
        Add("t_embedder.out_layer.bias", Hidden);
        Add("c_embedder.in_layer.weight", Hidden, TextDim);
        Add("c_embedder.in_layer.bias", Hidden);
        Add("c_embedder.out_layer.weight", Hidden, Hidden);
        Add("c_embedder.out_layer.bias", Hidden);
        for (int i = 0; i < Blocks; i++)
        {
            string p = $"individual_token_refiner.blocks.{i}.";
            Add(p + "norm1.weight", Hidden);
            Add(p + "norm1.bias", Hidden);
            Add(p + "norm2.weight", Hidden);
            Add(p + "norm2.bias", Hidden);
            Add(p + "self_attn.qkv.weight", 3 * Hidden, Hidden);
            Add(p + "self_attn.qkv.bias", 3 * Hidden);
            Add(p + "self_attn.proj.weight", Hidden, Hidden);
            Add(p + "self_attn.proj.bias", Hidden);
            Add(p + "mlp.0.weight", MlpDim, Hidden);
            Add(p + "mlp.0.bias", MlpDim);
            Add(p + "mlp.2.weight", Hidden, MlpDim);
            Add(p + "mlp.2.bias", Hidden);
            Add(p + "adaLN_modulation.1.weight", 2 * Hidden, Hidden);
            Add(p + "adaLN_modulation.1.bias", 2 * Hidden);
        }
        return w;
    }
}
