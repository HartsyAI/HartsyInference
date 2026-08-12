using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Parity for the RESIDENT int8 <c>Linear</c> path — an I8 weight carrying a <see cref="QuantWeightInfo"/>
/// consumed straight on the INT8 tensor cores — against comfy-kitchen's eager <c>int8_linear</c>. Both sides quantize
/// the activation per row to int8 and accumulate in int32, so this gate is tight: unlike the dequant fallback (see
/// <c>Int8ConvRotParityTests</c> in Diffusion.Tests) there is no missing activation-quantization term to absorb.
/// Regenerate the fixture with <c>tests/python-reference/int8_convrot_reference.py</c>.</summary>
/// <remarks>Skips cleanly when the backend has no packed-int8 branch: without one the weight falls through to the
/// generic GEMM, whose I8→F32 cast is unsupported and throws. The skip line prints the exception so a branch that
/// landed but whose eligibility gate excludes these shapes (e.g. an m ≥ 32 floor, which would drop M=1 and M=8) shows
/// up as a reported skip instead of disappearing.</remarks>
[Collection("CudaSerial")]
[Trait("Category", "GpuIntegration")]
public sealed unsafe class Int8ConvRotCudaParityTests
{
    private readonly ITestOutputHelper _output;
    public Int8ConvRotCudaParityTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

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

    [Theory]
    [InlineData(false)]  // F32 activations
    [InlineData(true)]   // F16 activations (the DiT loop's dtype)
    public void ResidentInt8LinearMatchesEagerInt8Linear(bool f16Activations)
    {
        Int8ConvRotReference? reference = Int8ConvRotReference.TryLoad();
        if (reference is null) return;   // tier-lint: guarded
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }

        // F32: the int32 accumulation is exact and both sides pick the same int8 codes, so all that is left is the
        // F32 dequant epilogue — measured 5.1e-8 to 2.7e-7 across the six shapes, hence 1e-5. F16 additionally
        // rounds the activation before rotation/quantization and the output after, which moves a small fraction of
        // elements across an int8 code boundary: measured 2.2e-4 to 2.8e-3, hence 6e-3. Both stay far under the
        // O(1) a dropped rotation or a mis-broadcast row scale would produce.
        double tolerance = f16Activations ? 6e-3 : 1e-5;
        DType activationDType = f16Activations ? DType.F16 : DType.F32;

        using CudaBackend backend = new CudaBackend(0, PtxDir());
        _output.WriteLine($"device: {backend.Context.DeviceName} (SM {backend.Context.ComputeCapabilityMajor}.{backend.Context.ComputeCapabilityMinor})");

        int checkedCases = 0;
        foreach (Int8ConvRotReference.Case testCase in reference.Cases)
        {
            using Tensor weight = FromSBytes(testCase.Quant, testCase.OutFeatures, testCase.InFeatures);
            using Tensor rowScale = FromFloats(testCase.RowScale, testCase.OutFeatures);
            weight.QuantInfo = new QuantWeightInfo
            {
                Format = "int8_tensorwise",
                RowScale = rowScale,
                ConvRotGroupSize = testCase.GroupSize,
            };

            using Tensor inputF32 = FromFloats(testCase.Activation, testCase.Rows, testCase.InFeatures);
            using Tensor? inputF16 = f16Activations ? inputF32.CastTo(DType.F16) : null;
            Tensor input = inputF16 ?? inputF32;
            using Tensor bias = FromFloats(testCase.Bias, testCase.OutFeatures);
            using Tensor output = new Tensor(new TensorShape(testCase.Rows, testCase.OutFeatures), activationDType);

            double relL2, relL2Bias;
            try
            {
                relL2 = RunAndCompare(backend, output, input, weight, null, testCase.Output);
                relL2Bias = RunAndCompare(backend, output, input, weight, bias, testCase.OutputWithBias);
            }
            catch (NotSupportedException error)
            {
                _output.WriteLine($"SKIPPED at {testCase}: the packed-int8 Linear branch did not take this call — {error.Message}");
                return;
            }

            _output.WriteLine($"cuda int8 {testCase} [{activationDType.Name} act]: relL2={relL2:E3} withBias={relL2Bias:E3}");
            Assert.True(relL2 < tolerance,
                $"case {testCase} [{activationDType.Name} act]: resident int8 Linear relL2={relL2:E3} exceeds {tolerance:E1}");
            Assert.True(relL2Bias < tolerance,
                $"case {testCase} [{activationDType.Name} act]: resident int8 Linear+bias relL2={relL2Bias:E3} exceeds {tolerance:E1}");
            checkedCases++;
        }
        Assert.True(checkedCases == reference.Cases.Count, $"only {checkedCases} of {reference.Cases.Count} cases ran");
    }

    private static double RunAndCompare(CudaBackend backend, Tensor output, Tensor input, Tensor weight, Tensor? bias, float[] expected)
    {
        backend.Linear(output, input, weight, bias);
        backend.Sync();
        if (output.DType == DType.F32)
            return Int8ConvRotReference.RelL2(new ReadOnlySpan<float>((float*)output.DataPointer, expected.Length), expected);
        using Tensor asF32 = output.CastTo(DType.F32);
        return Int8ConvRotReference.RelL2(new ReadOnlySpan<float>((float*)asF32.DataPointer, expected.Length), expected);
    }
}
