using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Tests.Shared;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary><c>numerics.longcatInt8WeightOnly</c>: which weights the setting marks, and that CUDA then runs each marked linear as the dense BF16
/// linear on the dequantized weight, with the activations as they are.</summary>
public sealed class LongCatInt8WeightOnlyTests
{
    private const int Rows = 128, Columns = 256, Tokens = 64;

    /// <summary>The resident route must differ from the dense one by more than this (largest difference over largest value), or the identity below
    /// would prove nothing.</summary>
    private const float RouteSeparation = 1e-4f;

    [Fact]
    public void TheSettingIsOffByDefaultAndThenMarksNothing()
    {
        using Tensor weight = Int8Weight(seed: 1, out Tensor scale);
        using Tensor rowScale = scale;

        Assert.False(EngineKnobs.LongcatInt8WeightOnly.Default);
        // A whole-run lane that names the knob (HARTSY_GPU_TEST_KNOBS) turns it on for every test in the run, so then it marks by design.
        if (Environment.GetEnvironmentVariable(GpuRunKnobs.EnvVar) is { } run && run.Contains("numerics.longcatInt8WeightOnly", StringComparison.Ordinal))
            return;
        Assert.Equal(0, LongCatRecipeSupport.ApplyInt8WeightOnly(new Dictionary<string, Tensor> { ["linear.weight"] = weight }));
        Assert.False(weight.QuantInfo!.FullPrecisionMatMul);
    }

    [Fact]
    public void TheSettingMarksOnlyPackedInt8TensorwiseWeights()
    {
        using Tensor packed = Int8Weight(seed: 1, out Tensor scale);
        using Tensor rowScale = scale;
        using Tensor dense = LongCatTestData.Random(new TensorShape(4, 4), 2);
        using Tensor otherFormat = new(new TensorShape(4, 4), DType.I8);
        otherFormat.QuantInfo = new QuantWeightInfo { Format = "another_format", RowScale = scale };

        using IDisposable on = On();
        int marked = LongCatRecipeSupport.ApplyInt8WeightOnly(new Dictionary<string, Tensor>
        {
            ["packed.weight"] = packed, ["dense.weight"] = dense, ["other.weight"] = otherFormat,
        });

        Assert.Equal(1, marked);
        Assert.True(packed.QuantInfo!.FullPrecisionMatMul);
        Assert.Same(scale, packed.QuantInfo.RowScale);
        Assert.Null(dense.QuantInfo);
        Assert.False(otherFormat.QuantInfo.FullPrecisionMatMul);
    }

    /// <summary>The three entry points the LongCat DiT and the shared attention layer use, on a weight preloaded to the device as a streamed block's
    /// are.</summary>
    [CudaDeviceFact]
    [Trait("Category", "GpuIntegration")]
    public void Cuda_AMarkedWeightRunsAsTheDenseLinearOnTheDequantizedWeight()
    {
        using Tensor weight = Int8Weight(seed: 3, out Tensor scale);
        using Tensor rowScale = scale;
        using Tensor input = LongCatTestData.Random(new TensorShape(Tokens, Columns), 4);
        using Tensor bias = LongCatTestData.Random(new TensorShape(Rows), 5);
        // The engine's GEMM for this float32 input runs in BF16 (ResolveGemmDtype), so the marked weight is the BF16 nearest-even product.
        using Tensor dense = Int8ConvRotCodec.DequantToBf16(weight, scale, 0, referenceRounding: true);
        using CudaBackend backend = GpuDevices.OpenCuda();

        float[] resident = Linear(backend, input, weight, bias);
        float[] expected = Linear(backend, input, dense, bias);
        Assert.True(Separation(resident, expected) > RouteSeparation,
            $"the resident route is within {Separation(resident, expected):E2} of the dense one, so this case cannot tell them apart");

        using (On())
            Assert.Equal(1, LongCatRecipeSupport.ApplyInt8WeightOnly(new Dictionary<string, Tensor> { ["linear.weight"] = weight }));
        backend.PreloadWeights([weight]);

        Assert.Equal(expected, Linear(backend, input, weight, bias));
        Assert.Equal(Window(backend, input, dense, bias), Window(backend, input, weight, bias));
        (float[] first, float[] second) = Grouped(backend, input, weight, bias);
        Assert.Equal(expected, first);
        Assert.Equal(expected, second);
        backend.FreeWeights([weight, dense, bias]);
    }

    private static IDisposable On() => KnobProfile.Create("longcat-int8-weight-only").With(EngineKnobs.LongcatInt8WeightOnly, true).Push();

    private static Tensor Int8Weight(int seed, out Tensor scale)
    {
        Random rng = new(seed);
        Tensor weight = new(new TensorShape(Rows, Columns), DType.I8);
        Span<byte> levels = weight.AsSpan<byte>();
        for (int i = 0; i < levels.Length; i++)
            levels[i] = unchecked((byte)(sbyte)rng.Next(-127, 128));
        scale = new Tensor(new TensorShape(Rows), DType.F32);
        Span<float> scales = scale.AsSpan<float>();
        for (int i = 0; i < scales.Length; i++)
            scales[i] = (float)(0.001 + 0.01 * rng.NextDouble());
        weight.QuantInfo = new QuantWeightInfo { Format = "int8_tensorwise", RowScale = scale };
        return weight;
    }

    private static float[] Linear(IBackend backend, Tensor input, Tensor weight, Tensor bias)
    {
        using Tensor output = new(new TensorShape(Tokens, Rows), DType.F32);
        backend.Linear(output, input, weight, bias);
        return output.AsReadOnlySpan<float>().ToArray();
    }

    /// <summary>The second half of the weight's rows, as a fused key/value projection reads its values.</summary>
    private static float[] Window(IBackend backend, Tensor input, Tensor weight, Tensor bias)
    {
        using Tensor output = new(new TensorShape(Tokens, Rows / 2), DType.F32);
        backend.LinearWeightRows(output, input, weight, bias, Rows / 2, Rows / 2);
        return output.AsReadOnlySpan<float>().ToArray();
    }

    private static (float[] First, float[] Second) Grouped(IBackend backend, Tensor input, Tensor weight, Tensor bias)
    {
        using Tensor first = new(new TensorShape(Tokens, Rows), DType.F32);
        using Tensor second = new(new TensorShape(Tokens, Rows), DType.F32);
        backend.LinearMulti(input, [new LinearOp(first, weight, bias), new LinearOp(second, weight, bias)]);
        return (first.AsReadOnlySpan<float>().ToArray(), second.AsReadOnlySpan<float>().ToArray());
    }

    private static float Separation(float[] actual, float[] expected)
    {
        float scale = expected.Max(Math.Abs), worst = 0f;
        for (int i = 0; i < expected.Length; i++)
            worst = Math.Max(worst, Math.Abs(actual[i] - expected[i]));
        return worst / scale;
    }
}
