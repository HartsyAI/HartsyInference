using HartsyInference.Core.Backends;
using HartsyInference.Core.Moe;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu.Moe;
using HartsyInference.ModelAssets.Gguf;
using Xunit;
using static HartsyInference.Cpu.Tests.Moe.MoeTestData;

namespace HartsyInference.Cpu.Tests.Moe;

/// <summary>
/// The packed CPU runner inside the heterogeneous executor. A mixed plan runs its CPU experts on packed bytes and its GPU
/// expert on the F32 reference; the result is held to the kernel's tolerance against the all-F32 reference on the same
/// dequantized weights.
/// </summary>
public sealed class PackedExpertHostRunnerTests
{
    private const int Hidden = 256;
    private const int Intermediate = 512;

    [Theory]
    [InlineData("Q8_0", 2e-2f)]
    [InlineData("Q4_K", 2.5e-2f)]
    public void MixedPlan_MatchesTheF32ReferenceWithinTheKernelTolerance(string dtypeName, float tolerance)
    {
        DType dtype = dtypeName == "Q8_0" ? DType.Q8_0 : DType.Q4_K;
        ExpertKey[] keys = [new(0, 0, 0), new(0, 1, 0), new(0, 2, 0)];
        int[] rows = [2, 1, 3];
        List<Tensor> owned = [];
        try
        {
            Dictionary<ExpertKey, ExpertWeights> packed = [];
            Dictionary<ExpertKey, F32ExpertWeights> reference = [];
            for (int i = 0; i < keys.Length; i++)
            {
                Tensor gate = Quantize(Random(Intermediate * Hidden, 100 + i, 0.1f), dtype, Intermediate, Hidden, out float[] gateDeq);
                Tensor up = Quantize(Random(Intermediate * Hidden, 200 + i, 0.1f), dtype, Intermediate, Hidden, out float[] upDeq);
                Tensor down = Quantize(Random(Hidden * Intermediate, 300 + i, 0.1f), dtype, Hidden, Intermediate, out float[] downDeq);
                owned.AddRange([gate, up, down]);
                packed[keys[i]] = new ExpertWeights(keys[i], new ExpertMatrix(gate), new ExpertMatrix(down), new ExpertMatrix(up));
                reference[keys[i]] = new F32ExpertWeights(Hidden, Intermediate, gateDeq, upDeq, downDeq);
            }

            int total = rows.Sum();
            float[] gathered = Random(total * Hidden, seed: 9, scale: 1f);
            ExpertAssignment[] allCpu = [.. keys.Select((k, i) => new ExpertAssignment(k, ExpertPlacement.Cpu, rows[i]))];
            ExpertAssignment[] mixed = [.. keys.Select((k, i) =>
                new ExpertAssignment(k, i == 1 ? ExpertPlacement.Gpu : ExpertPlacement.Cpu, rows[i]))];

            float[] expected = new float[total * Hidden];
            HeterogeneousExpertExecutor.Execute(ExpertProgram.Swiglu, allCpu, gathered, Hidden, expected,
                key => reference[key], device: null);

            float[] actual = new float[total * Hidden];
            PackedExpertHostRunner runner = new PackedExpertHostRunner(dtype, Hidden, Intermediate, key => packed[key]);
            HeterogeneousExpertExecutor.Execute(ExpertProgram.Swiglu, mixed, gathered, Hidden, actual, runner,
                new ReferenceDevice(ExpertProgram.Swiglu, reference));

            float maxAbs = 0f;
            float maxRef = 0f;
            for (int i = 0; i < expected.Length; i++)
            {
                maxAbs = MathF.Max(maxAbs, MathF.Abs(actual[i] - expected[i]));
                maxRef = MathF.Max(maxRef, MathF.Abs(expected[i]));
            }
            float relative = maxAbs / maxRef;
            Assert.True(relative <= tolerance, $"{dtypeName}: relative error {relative:E4} exceeds {tolerance:E2}.");
        }
        finally
        {
            foreach (Tensor tensor in owned) tensor.Dispose();
        }
    }

    [Fact]
    public void RefusesAnExpertWhoseDtypeIsNotTheRunnersDtype()
    {
        ExpertKey key = new(0, 0, 0);
        List<Tensor> owned = [];
        try
        {
            Tensor gate = Quantize(Random(Intermediate * Hidden, 1, 0.1f), DType.Q8_0, Intermediate, Hidden, out _);
            Tensor up = Quantize(Random(Intermediate * Hidden, 2, 0.1f), DType.Q8_0, Intermediate, Hidden, out _);
            Tensor down = Quantize(Random(Hidden * Intermediate, 3, 0.1f), DType.Q8_0, Hidden, Intermediate, out _);
            owned.AddRange([gate, up, down]);
            ExpertWeights weights = new(key, new ExpertMatrix(gate), new ExpertMatrix(down), new ExpertMatrix(up));
            PackedExpertHostRunner runner = new PackedExpertHostRunner(DType.Q4_K, Hidden, Intermediate, _ => weights);
            float[] x = Random(Hidden, 4, 1f);
            float[] y = new float[Hidden];
            Assert.Throws<InvalidOperationException>(() => runner.Run(ExpertProgram.Swiglu, key, x, 1, y));
        }
        finally
        {
            foreach (Tensor tensor in owned) tensor.Dispose();
        }
    }

    [Fact]
    public void RefusesADtypeWithoutAPackedKernel()
    {
        Assert.Throws<NotSupportedException>(() => new PackedExpertHostRunner(DType.F32, Hidden, Intermediate, _ => null!));
    }

    /// <summary>Stands in for a GPU run: the F32 reference on the same weights.</summary>
    private sealed class ReferenceDevice(ExpertProgram program, Dictionary<ExpertKey, F32ExpertWeights> weights) : IExpertDeviceRunner
    {
        public void Run(ExpertKey key, ReadOnlySpan<float> x, int rows, Span<float> y) =>
            ExpertProgramReference.Apply(program, weights[key], x, rows, y);
    }

    /// <summary>Quantizes <paramref name="values"/> as <c>[rows, cols]</c> with the production quantizer. Returns the packed
    /// tensor, which the caller owns, and the dequantized values the reference uses.</summary>
    private static Tensor Quantize(float[] values, DType dtype, int rows, int cols, out float[] dequantized)
    {
        using Tensor source = F32(values, rows, cols);
        Tensor quant = GgufQuantizer.Quantize(source, dtype);
        using Tensor back = GgufDequantizer.Dequantize(quant, DType.F32);
        dequantized = ReadF32(back);
        return quant;
    }
}
