using HartsyInference.Core.Moe;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu.Moe;
using HartsyInference.ModelAssets.Gguf;
using Xunit;
using Xunit.Abstractions;
using static HartsyInference.Cpu.Tests.Moe.MoeTestData;

namespace HartsyInference.Cpu.Tests.Moe;

/// <summary>
/// <see cref="CpuExpertKernels"/> against the F32 oracle: weights are quantized with the production quantizer, the kernel
/// reads the packed bytes, and the reference runs the same expert on the dequantized weights. The scalar and AVX2 paths must
/// agree bit for bit. Errors are measured per format and program; see the tolerances below.
/// </summary>
public sealed unsafe class CpuExpertKernelsTests
{
    private const int Hidden = 256;
    private const int Intermediate = 512;

    private readonly ITestOutputHelper _output;

    public CpuExpertKernelsTests(ITestOutputHelper output) => _output = output;

    public static TheoryData<string, int, int> Cases()
    {
        TheoryData<string, int, int> data = new();
        foreach (string dtype in new[] { "Q8_0", "Q4_K" })
            foreach (int program in new[] { 0, 1, 2, 3 })
                foreach (int rows in new[] { 1, 3, 8 })
                    data.Add(dtype, program, rows);
        return data;
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void MatchesDequantizedReference(string dtypeName, int programKind, int rows)
    {
        DType dtype = ParseDType(dtypeName);
        ExpertProgram program = ProgramFor(programKind);
        Weights weights = MakeWeights(dtype, seed: 100 + programKind * 10 + rows);
        float[] x = Random(rows * Hidden, seed: 7 + rows, scale: 1f);

        float[] expected = new float[rows * Hidden];
        ExpertProgramReference.Apply(program, weights.Dequantized, x, rows, expected);
        float[] actual = new float[rows * Hidden];
        CpuExpertKernels.Apply(program, dtype, Hidden, Intermediate, weights.Gate, weights.Up, weights.Down, x, rows, actual);

        (float maxAbs, float relative) = Measure(expected, actual);
        _output.WriteLine($"{dtypeName} program={programKind} rows={rows} maxAbs={maxAbs:E4} maxRelative={relative:E4}");
        Assert.True(relative <= Tolerance(dtype),
            $"{dtypeName} program {programKind} rows {rows}: relative error {relative:E4} exceeds {Tolerance(dtype):E2}.");
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void AvxPathMatchesScalarBitForBit(string dtypeName, int programKind, int rows)
    {
        DType dtype = ParseDType(dtypeName);
        ExpertProgram program = ProgramFor(programKind);
        Weights weights = MakeWeights(dtype, seed: 300 + programKind * 10 + rows);
        float[] x = Random(rows * Hidden, seed: 17 + rows, scale: 1f);

        float[] simd = new float[rows * Hidden];
        float[] scalar = new float[rows * Hidden];
        CpuExpertKernels.Apply(program, dtype, Hidden, Intermediate, weights.Gate, weights.Up, weights.Down, x, rows, simd);
        CpuExpertKernels.ApplyScalar(program, dtype, Hidden, Intermediate, weights.Gate, weights.Up, weights.Down, x, rows, scalar);

        for (int i = 0; i < simd.Length; i++)
            Assert.Equal(BitConverter.SingleToInt32Bits(scalar[i]), BitConverter.SingleToInt32Bits(simd[i]));
    }

    [Theory]
    [InlineData("Q8_0", true)]
    [InlineData("Q8_0", false)]
    [InlineData("Q4_K", true)]
    [InlineData("Q4_K", false)]
    public void SteadyStateCallAllocatesNothing(string dtypeName, bool simd)
    {
        DType dtype = ParseDType(dtypeName);
        ExpertProgram program = ProgramFor(2);
        Weights weights = MakeWeights(dtype, seed: 500);
        const int rows = 8;
        float[] x = Random(rows * Hidden, seed: 9, scale: 1f);
        float[] y = new float[rows * Hidden];

        // The first call sizes the pooled scratch; only calls after it are measured.
        Run(program, dtype, weights, x, rows, y, simd);
        Run(program, dtype, weights, x, rows, y, simd);

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 20; i++) Run(program, dtype, weights, x, rows, y, simd);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        _output.WriteLine($"{dtypeName} simd={simd} allocatedBytesOver20Calls={allocated}");
        Assert.Equal(0L, allocated);
    }

    [Fact]
    public void RejectsFloatWeightsAndOversizedBatch()
    {
        float[] x = new float[Hidden];
        float[] y = new float[Hidden];
        byte[] bytes = new byte[Intermediate * Hidden * 4];
        Assert.Throws<NotSupportedException>(() => CpuExpertKernels.Apply(ExpertProgram.Swiglu, DType.F32, Hidden, Intermediate,
            bytes, bytes, bytes, x, 1, y));
        Assert.Throws<ArgumentOutOfRangeException>(() => CpuExpertKernels.Apply(ExpertProgram.Swiglu, DType.Q8_0, Hidden, Intermediate,
            new byte[1], new byte[1], new byte[1], new float[9 * Hidden], 9, new float[9 * Hidden]));
    }

    private static void Run(ExpertProgram program, DType dtype, Weights weights, float[] x, int rows, float[] y, bool simd)
    {
        if (simd) CpuExpertKernels.Apply(program, dtype, Hidden, Intermediate, weights.Gate, weights.Up, weights.Down, x, rows, y);
        else CpuExpertKernels.ApplyScalar(program, dtype, Hidden, Intermediate, weights.Gate, weights.Up, weights.Down, x, rows, y);
    }

    /// <summary>Largest allowed relative error (max abs error over max abs reference) per format. Measured over all 24
    /// format, program and batch cases on this suite's seeds: Q8_0 peaks at 1.06e-2 and Q4_K at 1.17e-2. Each bound is
    /// about twice its measured peak. The reference uses the dequantized weights, so the remaining error is the int8
    /// quantization of the activations (the input row and the hidden row before Down).</summary>
    private static float Tolerance(DType dtype) => dtype == DType.Q8_0 ? 2e-2f : 2.5e-2f;

    private static (float MaxAbs, float Relative) Measure(float[] expected, float[] actual)
    {
        float maxAbs = 0f;
        float maxRef = 0f;
        for (int i = 0; i < expected.Length; i++)
        {
            maxAbs = MathF.Max(maxAbs, MathF.Abs(actual[i] - expected[i]));
            maxRef = MathF.Max(maxRef, MathF.Abs(expected[i]));
        }
        return (maxAbs, maxRef > 0f ? maxAbs / maxRef : maxAbs);
    }

    private static DType ParseDType(string name) => name == "Q8_0" ? DType.Q8_0 : DType.Q4_K;

    private static ExpertProgram ProgramFor(int kind) => kind switch
    {
        0 => ExpertProgram.Swiglu,
        1 => ExpertProgram.GeGlu,
        2 => ExpertProgram.SwigluClamped(0.5f),
        _ => new ExpertProgram(ExpertActivation.ReluSquared, float.PositiveInfinity, float.NegativeInfinity, float.PositiveInfinity),
    };

    /// <summary>One expert: packed bytes for the kernel, and the dequantized F32 matrices for the reference.</summary>
    private sealed class Weights
    {
        public required byte[] Gate { get; init; }
        public required byte[] Up { get; init; }
        public required byte[] Down { get; init; }
        public required F32ExpertWeights Dequantized { get; init; }
    }

    private static Weights MakeWeights(DType dtype, int seed)
    {
        float[] gate = Random(Intermediate * Hidden, seed, scale: 0.1f);
        float[] up = Random(Intermediate * Hidden, seed + 1, scale: 0.1f);
        float[] down = Random(Hidden * Intermediate, seed + 2, scale: 0.1f);
        byte[] gatePacked = Pack(gate, dtype, Intermediate, Hidden, out float[] gateDeq);
        byte[] upPacked = Pack(up, dtype, Intermediate, Hidden, out float[] upDeq);
        byte[] downPacked = Pack(down, dtype, Hidden, Intermediate, out float[] downDeq);
        return new Weights
        {
            Gate = gatePacked,
            Up = upPacked,
            Down = downPacked,
            Dequantized = new F32ExpertWeights(Hidden, Intermediate, gateDeq, upDeq, downDeq),
        };
    }

    /// <summary>Quantizes <paramref name="values"/> as <c>[rows, cols]</c> with the production quantizer, and returns the packed
    /// bytes together with their dequantized values.</summary>
    private static byte[] Pack(float[] values, DType dtype, int rows, int cols, out float[] dequantized)
    {
        using Tensor source = F32(values, rows, cols);
        using Tensor quant = GgufQuantizer.Quantize(source, dtype);
        long bytes = dtype.ComputeByteCount(quant.ElementCount);
        byte[] packed = new byte[bytes];
        fixed (byte* dst = packed) Buffer.MemoryCopy((void*)quant.DataPointer, dst, bytes, bytes);
        using Tensor back = GgufDequantizer.Dequantize(quant, DType.F32);
        dequantized = ReadF32(back);
        return packed;
    }
}
