using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests.Parity;

/// <summary>Checks <see cref="Int8ConvRotCodec"/> and the dequant-fallback <c>Linear</c> against comfy-kitchen's own
/// eager INT8 backend (<c>comfy_kitchen/backends/eager/quantization.py</c>), which ComfyUI bit-checks its CUDA and
/// Triton backends against — so parity with eager is parity with ComfyUI. Regenerate the fixture with
/// <c>tests/python-reference/int8_convrot_reference.py</c>.</summary>
public sealed unsafe class Int8ConvRotParityTests
{
    private readonly ITestOutputHelper _output;
    public Int8ConvRotParityTests(ITestOutputHelper output) => _output = output;

    private static Tensor FromFloats(float[] data, params long[] shape)
    {
        Tensor tensor = new Tensor(new TensorShape(shape), DType.F32);
        data.AsSpan().CopyTo(new Span<float>((void*)tensor.DataPointer, data.Length));
        return tensor;
    }

    private static Tensor FromSBytes(sbyte[] data, params long[] shape)
    {
        Tensor tensor = new Tensor(new TensorShape(shape), DType.I8);
        data.AsSpan().CopyTo(new Span<sbyte>((void*)tensor.DataPointer, data.Length));
        return tensor;
    }

    [Fact]
    public void IsValidGroupSizeAcceptsOnlyPowersOfFourAtLeastFour()
    {
        foreach (int size in new[] { 4, 16, 64, 256, 1024, 4096, 65536 })
            Assert.True(Int8ConvRotCodec.IsValidGroupSize(size), $"{size} is a power of four and must be accepted.");
        foreach (int size in new[] { -256, 0, 1, 2, 3, 5, 8, 32, 128, 512, 2048, 255, 257 })
            Assert.False(Int8ConvRotCodec.IsValidGroupSize(size), $"{size} is not a power of four and must be rejected.");
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void BuildHadamardMatchesComfyKitchen()
    {
        Int8ConvRotReference? reference = Int8ConvRotReference.TryLoad();
        if (reference is null) return;   // tier-lint: guarded
        Assert.NotEmpty(reference.Hadamards);

        foreach ((int size, float[] expected) in reference.Hadamards)
        {
            float[] actual = Int8ConvRotCodec.BuildHadamard(size);
            Assert.Equal(expected.Length, actual.Length);
            double relL2 = Int8ConvRotReference.RelL2(actual, expected);

            // The entries are dyadic (±4^-d) and the butterfly's per-stage 0.5 is exact in binary floating point,
            // so anything above round-off means a different matrix, not a different rounding.
            double maxAbs = 0;
            for (int i = 0; i < actual.Length; i++) maxAbs = Math.Max(maxAbs, Math.Abs(actual[i] - expected[i]));

            double maxAsymmetry = 0;
            for (int row = 0; row < size; row++)
                for (int column = 0; column < size; column++)
                    maxAsymmetry = Math.Max(maxAsymmetry, Math.Abs(actual[row * size + column] - actual[column * size + row]));

            // H is its own inverse, which is the entire reason the rotation cancels between weight and activation.
            double maxIdentityError = 0;
            for (int row = 0; row < size; row++)
            {
                for (int column = 0; column < size; column++)
                {
                    double sum = 0;
                    for (int inner = 0; inner < size; inner++)
                        sum += (double)actual[row * size + inner] * actual[inner * size + column];
                    maxIdentityError = Math.Max(maxIdentityError, Math.Abs(sum - (row == column ? 1.0 : 0.0)));
                }
            }

            _output.WriteLine($"H[{size}]: relL2={relL2:E3} maxAbs={maxAbs:E3} asymmetry={maxAsymmetry:E3} " +
                $"maxAbs(H@H - I)={maxIdentityError:E3}");
            Assert.True(relL2 < 1e-6, $"H[{size}] diverged from comfy-kitchen: relL2={relL2:E3} maxAbs={maxAbs:E3}");
            Assert.True(maxAsymmetry == 0.0, $"H[{size}] is not symmetric: max |H - Hᵀ| = {maxAsymmetry:E3}");
            Assert.True(maxIdentityError < 1e-5, $"H[{size}] is not an involution: max |H@H - I| = {maxIdentityError:E3}");
        }
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void ApplyRotationInPlaceMatchesTheDumpedHadamardMatMul()
    {
        Int8ConvRotReference? reference = Int8ConvRotReference.TryLoad();
        if (reference is null) return;   // tier-lint: guarded

        int checkedCases = 0;
        foreach (Int8ConvRotReference.Case testCase in reference.Cases)
        {
            if (testCase.GroupSize == 0) continue;
            int group = testCase.GroupSize;
            float[] hadamard = reference.Hadamards[group];

            // x @ H, group by group, in double — the butterfly is the thing under test, so the reference must be
            // the literal matrix product against the matrix comfy-kitchen dumped.
            float[] expected = new float[testCase.Activation.Length];
            for (int groupStart = 0; groupStart < expected.Length; groupStart += group)
            {
                for (int column = 0; column < group; column++)
                {
                    double sum = 0;
                    for (int inner = 0; inner < group; inner++)
                        sum += (double)testCase.Activation[groupStart + inner] * hadamard[inner * group + column];
                    expected[groupStart + column] = (float)sum;
                }
            }

            float[] actual = (float[])testCase.Activation.Clone();
            Int8ConvRotCodec.ApplyRotationInPlace(actual, group);
            double relL2 = Int8ConvRotReference.RelL2(actual, expected);
            _output.WriteLine($"rotation {testCase}: relL2={relL2:E3}");
            Assert.True(relL2 < 1e-6, $"case {testCase}: butterfly rotation diverged from x@H, relL2={relL2:E3}");
            checkedCases++;
        }
        Assert.True(checkedCases > 0, "fixture carried no rotated case");
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void DequantToBf16ApproximatesTheOriginalWeight()
    {
        Int8ConvRotReference? reference = Int8ConvRotReference.TryLoad();
        if (reference is null) return;   // tier-lint: guarded

        // Error budget: the int8 step is scale[r] = rowAbsMax/127, so round-to-nearest leaves a uniform residual of
        // std scale/√12; against a Gaussian row whose absmax runs ~3.1-3.4σ that is ~3.2/(127·√12) ≈ 7.3e-3 of the
        // row RMS. Truncating BF16 storage adds 2^-8/√3 ≈ 2.3e-3. Combined ≈ 7.6e-3, and the six shapes measure
        // 7.56e-3 to 8.42e-3 — so 1.5e-2 keeps ~1.8× margin without going loose enough to hide a dropped rotation.
        const double Tolerance = 1.5e-2;
        foreach (Int8ConvRotReference.Case testCase in reference.Cases)
        {
            using Tensor quant = FromSBytes(testCase.Quant, testCase.OutFeatures, testCase.InFeatures);
            using Tensor rowScale = FromFloats(testCase.RowScale, testCase.OutFeatures);
            using Tensor dequantized = Int8ConvRotCodec.DequantToBf16(quant, rowScale, testCase.GroupSize);
            Assert.Equal(DType.BF16, dequantized.DType);
            using Tensor asF32 = dequantized.CastTo(DType.F32);

            ReadOnlySpan<float> actual = new ReadOnlySpan<float>((float*)asF32.DataPointer, testCase.Weight.Length);
            double relL2 = Int8ConvRotReference.RelL2(actual, testCase.Weight);
            _output.WriteLine($"dequant {testCase}: relL2={relL2:E3}");
            Assert.True(relL2 < Tolerance,
                $"case {testCase}: int8 dequant round-trip relL2={relL2:E3} exceeds the {Tolerance:E1} quantization budget");
        }
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void CpuLinearThroughTheDequantFallbackMatchesEagerInt8Linear()
    {
        Int8ConvRotReference? reference = Int8ConvRotReference.TryLoad();
        if (reference is null) return;   // tier-lint: guarded
        using CpuBackend cpuBackend = new CpuBackend();
        IBackend backend = cpuBackend;

        // The weight-quantization error CANCELS here — both sides consume the same q and the same scale — so what is
        // left is eager's per-row int8 ACTIVATION quantization, which the dequant fallback does not do: ~3.2/(127·√12)
        // ≈ 7.3e-3 of the activation RMS, plus BF16 weight truncation at 2.3e-3. Combined ≈ 7.6e-3, and the six shapes
        // measure 7.11e-3 to 8.52e-3; 2.5e-2 is a loose but still meaningful gate (a dropped rotation or a
        // mis-broadcast scale lands at O(1)).
        const double Tolerance = 2.5e-2;
        foreach (Int8ConvRotReference.Case testCase in reference.Cases)
        {
            using Tensor quant = FromSBytes(testCase.Quant, testCase.OutFeatures, testCase.InFeatures);
            using Tensor rowScale = FromFloats(testCase.RowScale, testCase.OutFeatures);
            using Tensor weight = Int8ConvRotCodec.DequantToBf16(quant, rowScale, testCase.GroupSize);
            using Tensor input = FromFloats(testCase.Activation, testCase.Rows, testCase.InFeatures);
            using Tensor bias = FromFloats(testCase.Bias, testCase.OutFeatures);
            using Tensor output = new Tensor(new TensorShape(testCase.Rows, testCase.OutFeatures), DType.F32);

            backend.Linear(output, input, weight, null);
            ReadOnlySpan<float> plain = new ReadOnlySpan<float>((float*)output.DataPointer, testCase.Output.Length);
            double relL2 = Int8ConvRotReference.RelL2(plain, testCase.Output);

            backend.Linear(output, input, weight, bias);
            ReadOnlySpan<float> biased = new ReadOnlySpan<float>((float*)output.DataPointer, testCase.OutputWithBias.Length);
            double relL2Bias = Int8ConvRotReference.RelL2(biased, testCase.OutputWithBias);

            _output.WriteLine($"cpu fallback {testCase}: relL2={relL2:E3} withBias={relL2Bias:E3}");
            Assert.True(relL2 < Tolerance,
                $"case {testCase}: dequant-fallback Linear relL2={relL2:E3} exceeds the {Tolerance:E1} activation-quantization budget");
            Assert.True(relL2Bias < Tolerance,
                $"case {testCase}: dequant-fallback Linear+bias relL2={relL2Bias:E3} exceeds the {Tolerance:E1} activation-quantization budget");
        }
    }
}
