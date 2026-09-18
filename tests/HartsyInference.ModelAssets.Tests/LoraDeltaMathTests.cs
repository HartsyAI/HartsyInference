using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.Lora;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Pins every <see cref="LoraDelta"/> subclass against ComfyUI's own adapter math. Expected values come from
/// <c>tests/python-reference/lycoris_lora_reference.py</c>, which transcribes
/// <c>comfy/weight_adapter/{loha,lokr,lora}.py</c> and <c>base.py::weight_decompose</c>; the fixtures are small
/// exact-in-binary32 ramps so only the DoRA cases need a tolerance. A wrong factor order or scale rule here does not
/// fail — it produces a plausible image that is not the one the LoRA was trained for.</summary>
public sealed class LoraDeltaMathTests
{
    [Fact]
    public void StandardDelta_MatchesTheShippedMergePathBitwise()
    {
        using CpuBackend backend = new();
        using Tensor down = Ramp([2, 6], start: -3.0f);
        using Tensor up = Ramp([4, 2], start: 1.0f);
        using Tensor baseWeight = Ramp([4, 6], start: 0.5f, step: 0.25f);
        StandardLoraDelta delta = new() { Down = down, Up = up, Alpha = 3.0f };
        const float Strength = 0.75f;

        // The reference is the op sequence LoraStack.AccumulateDelta runs: MatMul, one Scale by
        // strength * (alpha / rank), then Add. Two roundings instead of one would already break this.
        using Tensor expected = baseWeight.CastTo(DType.F32);
        using (Tensor product = delta.ComputeF32(backend))
        {
            backend.Scale(product, product, Strength * delta.Scale);
            backend.Add(expected, expected, product);
        }

        LoraLayer layer = new()
        {
            TargetKey = "blocks.0.attn.to_q.weight",
            Target = LoraTarget.Transformer,
            Delta = delta,
        };
        using LoraFile file = new()
        {
            FilePath = "synthetic",
            Format = LoraFormat.DiffusersFlux,
            Layers = [layer],
        };
        using LoraStack stack = new();
        stack.Add(file, Strength);
        Dictionary<string, Tensor> weights = new(StringComparer.Ordinal)
        {
            ["blocks.0.attn.to_q.weight"] = baseWeight,
        };

        Assert.Equal(1, stack.ApplyTo(weights, LoraTarget.Transformer, backend));
        Assert.Equal(expected.AsReadOnlySpan<float>().ToArray(),
            weights["blocks.0.attn.to_q.weight"].AsReadOnlySpan<float>().ToArray());
    }

    [Fact]
    public void StandardDelta_ConvAdapterFlattensKernelAxesIntoColumns()
    {
        using CpuBackend backend = new();
        using Tensor up = Ramp([4, 2, 1, 1], start: 1.0f);
        using Tensor down = Ramp([2, 3, 3, 3], start: -5.0f);
        StandardLoraDelta delta = new() { Down = down, Up = up, Alpha = 2.0f };

        Assert.Equal(4, delta.OutFeatures);
        Assert.Equal(27, delta.InFeatures);
        using Tensor product = delta.ComputeF32(backend);
        Assert.Equal(new TensorShape(4, 27), product.Shape);
        // Spot-checked against conv_lora in the python reference: first row, first column, last column.
        ReadOnlySpan<float> values = product.AsReadOnlySpan<float>();
        Assert.Equal(39.0f, values[0]);
        Assert.Equal(117.0f, values[26]);
        Assert.Equal(531.0f, values[^1]);
    }

    [Fact]
    public void StandardDelta_MatchesConvWeightOfRankFour()
    {
        using Tensor up = Ramp([4, 2, 1, 1], start: 1.0f);
        using Tensor down = Ramp([2, 3, 3, 3], start: -5.0f);
        using Tensor convWeight = Ramp([4, 3, 3, 3], start: 0.0f);
        using Tensor linearWeight = Ramp([4, 27], start: 0.0f);
        using Tensor wrongWeight = Ramp([4, 9], start: 0.0f);
        StandardLoraDelta delta = new() { Down = down, Up = up, Alpha = 2.0f };

        Assert.True(delta.MatchesShape(convWeight));
        Assert.True(delta.MatchesShape(linearWeight));
        Assert.False(delta.MatchesShape(wrongWeight));
    }

    [Fact]
    public void LoHaDelta_PlainFactorsMatchComfyHadamardProduct()
    {
        using CpuBackend backend = new();
        using Tensor w1a = Ramp([4, 2], start: 1.0f);
        using Tensor w1b = Ramp([2, 6], start: -3.0f);
        using Tensor w2a = Ramp([4, 2], start: 2.0f);
        using Tensor w2b = Ramp([2, 6], start: 1.0f, step: -1.0f);
        LoHaDelta delta = new() { W1A = w1a, W1B = w1b, W2A = w2a, W2B = w2b, Alpha = 1.0f };

        Assert.Equal(LoraVariant.LoHa, delta.Variant);
        Assert.Equal(2, delta.Rank);
        Assert.Equal(0.5f, delta.Scale);
        Assert.Equal(4, delta.OutFeatures);
        Assert.Equal(6, delta.InFeatures);
        using Tensor product = delta.ComputeF32(backend);
        Assert.Equal(
            new float[]
            {
                -39, -108, -207, -336, -495, -684,
                -63, -300, -663, -1152, -1767, -2508,
                -87, -588, -1375, -2448, -3807, -5452,
                -111, -972, -2343, -4224, -6615, -9516,
            },
            product.AsReadOnlySpan<float>().ToArray());
    }

    [Fact]
    public void LoHaDelta_TuckerCoresRebuildThroughTheEinsumIndexOrder()
    {
        using CpuBackend backend = new();
        using Tensor t1 = Ramp([2, 2, 2, 2], start: 1.0f);
        using Tensor w1a = Ramp([2, 3], start: 1.0f);
        using Tensor w1b = Ramp([2, 2], start: -1.0f);
        using Tensor t2 = Ramp([2, 2, 2, 2], start: 2.0f);
        using Tensor w2a = Ramp([2, 3], start: -2.0f);
        using Tensor w2b = Ramp([2, 2], start: 1.0f, step: -1.0f);
        LoHaDelta delta = new() { W1A = w1a, W1B = w1b, W2A = w2a, W2B = w2b, T1 = t1, T2 = t2, Alpha = 2.0f };

        // In Tucker mode LyCORIS stores hada_w1_a transposed, so the out dim is its SECOND axis.
        Assert.Equal(3, delta.OutFeatures);
        Assert.Equal(8, delta.InFeatures);
        using Tensor product = delta.ComputeF32(backend);
        Assert.Equal(
            new float[]
            {
                80, 80, 80, 80, -456, -248, 0, 288,
                -112, -112, -112, -112, -6600, -7544, -8544, -9600,
                -432, -432, -432, -432, -15624, -18360, -21312, -24480,
            },
            product.AsReadOnlySpan<float>().ToArray());
    }

    [Fact]
    public void LoKrDelta_WholeFactorsProduceTheKroneckerProductAndIgnoreAlpha()
    {
        using CpuBackend backend = new();
        using Tensor w1 = Ramp([2, 2], start: 1.0f);
        using Tensor w2 = Ramp([2, 3], start: -2.0f);
        LoKrDelta delta = new() { W1 = w1, W2 = w2, Alpha = 8.0f };

        Assert.Equal(LoraVariant.LoKr, delta.Variant);
        Assert.Equal(4, delta.OutFeatures);
        Assert.Equal(6, delta.InFeatures);
        // ComfyUI's `dim` stays None when neither factor is low-rank, so the stored alpha is deliberately dropped.
        Assert.Equal(1.0f, delta.Scale);
        using Tensor product = delta.ComputeF32(backend);
        Assert.Equal(
            new float[]
            {
                -2, -1, 0, -4, -2, 0,
                1, 2, 3, 2, 4, 6,
                -6, -3, 0, -8, -4, 0,
                3, 6, 9, 4, 8, 12,
            },
            product.AsReadOnlySpan<float>().ToArray());
    }

    [Fact]
    public void LoKrDelta_FactoredSidesRebuildAndScaleByTheLastFactoredRank()
    {
        using CpuBackend backend = new();
        using Tensor w1a = Ramp([2, 1], start: 1.0f);
        using Tensor w1b = Ramp([1, 2], start: 3.0f);
        using Tensor w2a = Ramp([2, 1], start: -1.0f, step: 3.0f);
        using Tensor w2b = Ramp([1, 3], start: 2.0f);
        LoKrDelta delta = new() { W1A = w1a, W1B = w1b, W2A = w2a, W2B = w2b, Alpha = 4.0f };

        Assert.Equal(4.0f, delta.Scale);
        using Tensor product = delta.ComputeF32(backend);
        Assert.Equal(
            new float[]
            {
                -6, -9, -12, -8, -12, -16,
                12, 18, 24, 16, 24, 32,
                -12, -18, -24, -16, -24, -32,
                24, 36, 48, 32, 48, 64,
            },
            product.AsReadOnlySpan<float>().ToArray());
    }

    [Fact]
    public void LoKrDelta_ConvRightFactorFlattensKernelAxesIntoColumns()
    {
        using CpuBackend backend = new();
        using Tensor w1 = Ramp([2, 2], start: 1.0f);
        using Tensor w2 = Ramp([2, 1, 2, 2], start: -1.0f);
        LoKrDelta delta = new() { W1 = w1, W2 = w2, Alpha = 1.0f };

        Assert.Equal(4, delta.OutFeatures);
        Assert.Equal(8, delta.InFeatures);
        using Tensor product = delta.ComputeF32(backend);
        Assert.Equal(
            new float[]
            {
                -1, 0, 1, 2, -2, 0, 2, 4,
                3, 4, 5, 6, 6, 8, 10, 12,
                -3, 0, 3, 6, -4, 0, 4, 8,
                9, 12, 15, 18, 12, 16, 20, 24,
            },
            product.AsReadOnlySpan<float>().ToArray());
    }

    [Fact]
    public void Dora_OnOutputAxisNormalizesByTheOriginalWeightRows()
    {
        using Tensor weight = Ramp([3, 4], start: -4.0f, step: 0.5f);
        using Tensor delta = Ramp([3, 4], start: 1.0f, step: -0.25f);
        using Tensor magnitude = FromValues([3, 1], [2.0f, 3.0f, 0.5f]);

        LoraDoraDecompose.Apply(weight, delta, magnitude, scale: 0.5f, strength: 1.0f);

        AssertClose(
            [
                -1.0613372f, -0.9476226f, -0.83390784f, -0.72019315f,
                -2.1908901f, -1.7800982f, -1.3693063f, -0.95851439f,
                -0.1336306f, -0.033407651f, 0.066815302f, 0.16703826f,
            ],
            weight);
    }

    [Fact]
    public void Dora_StrengthInterpolatesBetweenTheOriginalAndDecomposedWeight()
    {
        using Tensor weight = Ramp([3, 4], start: -4.0f, step: 0.5f);
        using Tensor delta = Ramp([3, 4], start: 1.0f, step: -0.25f);
        using Tensor magnitude = FromValues([3, 1], [2.0f, 3.0f, 0.5f]);

        LoraDoraDecompose.Apply(weight, delta, magnitude, scale: 0.5f, strength: 0.5f);

        AssertClose(
            [
                -2.5306687f, -2.2238111f, -1.9169539f, -1.6100966f,
                -2.0954452f, -1.6400491f, -1.1846532f, -0.72925723f,
                -0.066815302f, 0.23329619f, 0.53340769f, 0.8335191f,
            ],
            weight);
    }

    [Fact]
    public void Dora_OnInputAxisNormalizesByTheLoraedWeightColumns()
    {
        using Tensor weight = Ramp([3, 4], start: -4.0f, step: 0.5f);
        using Tensor delta = Ramp([3, 4], start: 1.0f, step: -0.25f);
        using Tensor magnitude = FromValues([1, 4], [1.5f, 2.0f, 0.25f, 3.0f]);

        LoraDoraDecompose.Apply(weight, delta, magnitude, scale: 0.5f, strength: 1.0f);

        AssertClose(
            [
                -1.2924607f, -1.7733173f, -0.22681618f, -2.7329407f,
                -0.73854893f, -0.92212504f, -0.10309827f, -1.0068729f,
                -0.18463723f, -0.070932694f, 0.020619653f, 0.71919495f,
            ],
            weight);
    }

    private static void AssertClose(float[] expected, Tensor actual)
    {
        ReadOnlySpan<float> values = actual.AsReadOnlySpan<float>();
        Assert.Equal(expected.Length, values.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(MathF.Abs(expected[i] - values[i]) <= 1e-5f * MathF.Max(1.0f, MathF.Abs(expected[i])),
                $"index {i}: expected {expected[i]}, got {values[i]}");
        }
    }

    /// <summary>The <c>seq()</c> ramp the python reference builds its fixtures from.</summary>
    private static Tensor Ramp(long[] shape, float start, float step = 1.0f)
    {
        Tensor tensor = new(new TensorShape(shape), DType.F32);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = start + i * step;
        }
        return tensor;
    }

    private static Tensor FromValues(long[] shape, float[] values)
    {
        Tensor tensor = new(new TensorShape(shape), DType.F32);
        values.CopyTo(tensor.AsSpan<float>());
        return tensor;
    }
}
