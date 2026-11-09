using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Core.Tests.Elementwise;

/// <summary>The CUDA <c>elementwise_math</c> kernels against the host reference for every F32/BF16 input and output pair, over a
/// count that spans many blocks and ends mid-block.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class ElementwiseMathGpuTests
{
    private const int Count = 70_001;

    private readonly ITestOutputHelper _output;
    public ElementwiseMathGpuTests(ITestOutputHelper output) => _output = output;

    [Theory]
    [InlineData(ElementwiseMathFunction.Exp, false, false)]
    [InlineData(ElementwiseMathFunction.Exp, false, true)]
    [InlineData(ElementwiseMathFunction.Exp, true, false)]
    [InlineData(ElementwiseMathFunction.Exp, true, true)]
    [InlineData(ElementwiseMathFunction.Reciprocal, false, false)]
    [InlineData(ElementwiseMathFunction.Reciprocal, false, true)]
    [InlineData(ElementwiseMathFunction.Reciprocal, true, false)]
    [InlineData(ElementwiseMathFunction.Reciprocal, true, true)]
    public void Kernel_MatchesHostReference(ElementwiseMathFunction function, bool bf16Input, bool bf16Output)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        using CudaBackend backend = new(0, PtxDir());
        IBackend device = backend;
        TensorShape shape = new(Count);
        using Tensor input = new(shape, bf16Input ? DType.BF16 : DType.F32);
        using Tensor expected = new(shape, bf16Output ? DType.BF16 : DType.F32);
        using Tensor actual = new(shape, bf16Output ? DType.BF16 : DType.F32);
        Fill(input, function, new Random(23));

        if (function == ElementwiseMathFunction.Exp)
        {
            ElementwiseMathReference.Exp(expected, input);
            device.Exp(actual, input);
        }
        else
        {
            ElementwiseMathReference.Reciprocal(expected, input);
            device.Reciprocal(actual, input);
        }
        backend.OffloadActivation(actual);

        // expf is within 2 ulp of the host's exp; IEEE division is exact on both sides.
        float relative = function == ElementwiseMathFunction.Exp ? 4e-7f : 0f;
        int mismatches = 0;
        for (int i = 0; i < Count; i++)
        {
            float want = Read(expected, i);
            float got = Read(actual, i);
            float allowed = bf16Output ? Bf16Step(want) * (function == ElementwiseMathFunction.Exp ? 1f : 0f) : MathF.Abs(want) * relative;
            if (MathF.Abs(got - want) <= allowed) continue;
            if (++mismatches <= 5) _output.WriteLine($"element {i}: device {got} vs host {want}");
        }
        Assert.Equal(0, mismatches);
    }

    /// <summary>Exp inputs in [-20, 20]; reciprocal inputs of magnitude in [1e-3, 1e3] with both signs.</summary>
    private static void Fill(Tensor tensor, ElementwiseMathFunction function, Random random)
    {
        for (int i = 0; i < Count; i++)
        {
            float value = function == ElementwiseMathFunction.Exp
                ? ((float)random.NextDouble() * 2f - 1f) * 20f
                : (random.Next(2) == 0 ? -1f : 1f) * MathF.Pow(10f, (float)random.NextDouble() * 6f - 3f);
            if (tensor.DType == DType.BF16)
                tensor.AsSpan<ushort>()[i] = BFloat16Bits.FromFloat(value);
            else
                tensor.AsSpan<float>()[i] = value;
        }
    }

    private static float Read(Tensor tensor, int i) =>
        tensor.DType == DType.BF16 ? BFloat16Bits.ToFloat(tensor.AsSpan<ushort>()[i]) : tensor.AsSpan<float>()[i];

    /// <summary>One BF16 step at <paramref name="value"/>'s magnitude, the gap a one-ulp F32 difference can round across.</summary>
    private static float Bf16Step(float value) => MathF.Abs(value) / 128f;

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
