using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Core.Tests.Conv;

/// <summary>The CUDA BF16-output depthwise Conv2D against the F32 reference on an FFN-like shape (channels 256, 11x20, batch 2, 3x3).</summary>
[Trait("Category", "GpuIntegration")]
public sealed class DepthwiseConv2dBf16GpuTests
{
    private const int Batch = 2, Channels = 256, Height = 11, Width = 20, Kernel = 3;

    private readonly ITestOutputHelper _output;
    public DepthwiseConv2dBf16GpuTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Conv2dDepthwise_Bf16Output_MatchesF32ReferenceOnBf16Operands()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        TensorShape inShape = new(Batch, Channels, Height, Width);
        TensorShape weightShape = new(Channels, 1, Kernel, Kernel);
        TensorShape biasShape = new(Channels);
        using CudaBackend backend = new(0, PtxDir());
        IBackend reference = new CpuBackend();
        using Tensor input = new(inShape, DType.BF16);
        using Tensor weight = new(weightShape, DType.BF16);
        using Tensor bias = new(biasShape, DType.BF16);
        using Tensor refInput = new(inShape, DType.F32);
        using Tensor refWeight = new(weightShape, DType.F32);
        using Tensor refBias = new(biasShape, DType.F32);
        using Tensor refOutput = new(inShape, DType.F32);
        using Tensor output = new(inShape, DType.BF16);

        Random random = new(7);
        FillBf16(input, refInput, random, 1f);
        FillBf16(weight, refWeight, random, 0.5f);
        FillBf16(bias, refBias, random, 0.5f);

        // The F32 reference runs on the same BF16-exact operands, so the device result differs only by its final BF16 rounding.
        reference.Conv2dDepthwise(refOutput, refInput, refWeight, refBias, 1, 1, 1, 1);
        backend.Conv2dDepthwise(output, input, weight, bias, 1, 1, 1, 1);
        backend.OffloadActivation(output);

        ReadOnlySpan<float> expected = refOutput.AsSpan<float>();
        ReadOnlySpan<ushort> actual = output.AsSpan<ushort>();
        Assert.Equal(expected.Length, actual.Length);
        int mismatches = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float got = BFloat16Bits.ToFloat(actual[i]);
            float want = expected[i];
            if (MathF.Abs(got - want) <= MathF.Abs(want) / 128f + 1e-6f) continue;
            if (++mismatches <= 5) _output.WriteLine($"element {i}: bf16 {got} vs f32 {want}");
        }
        Assert.Equal(0, mismatches);
    }

    /// <summary>Fills the BF16 operand and its F32 twin with the same values, so the reference sees exactly what the kernel reads.</summary>
    private static void FillBf16(Tensor operand, Tensor twin, Random random, float scale)
    {
        Span<ushort> bits = operand.AsSpan<ushort>();
        Span<float> values = twin.AsSpan<float>();
        for (int i = 0; i < bits.Length; i++)
        {
            ushort raw = BFloat16Bits.FromFloat(((float)random.NextDouble() * 2f - 1f) * scale);
            bits[i] = raw;
            values[i] = BFloat16Bits.ToFloat(raw);
        }
    }

    private static string PtxDir()
    {
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(candidate)) return candidate;
        }
        throw new DirectoryNotFoundException("src/HartsyInference.Cuda/Ptx not found above the test output directory.");
    }
}
