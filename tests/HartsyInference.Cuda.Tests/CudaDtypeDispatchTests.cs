using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Validates floating-point dtype routing and shape contracts for CUDA activation and KV-repeat operations.</summary>
[Collection("CudaSerial")]
public sealed unsafe class CudaDtypeDispatchTests
{
    private readonly ITestOutputHelper _output;

    /// <summary>Captures numerical parity diagnostics for failed and detailed test runs.</summary>
    public CudaDtypeDispatchTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    private static Tensor RandomTensor(TensorShape shape, int seed, float min = -3f, float max = 3f)
    {
        Tensor tensor = new Tensor(shape, DType.F32);
        float* values = (float*)tensor.DataPointer;
        Random random = new Random(seed);
        for (long i = 0; i < shape.ElementCount; i++)
            values[i] = (float)(random.NextDouble() * (max - min) + min);
        return tensor;
    }

    private static DType HalfDtype(string name) => name == "F16" ? DType.F16 : DType.BF16;

    private static float ReadFloat(Tensor tensor, long index)
    {
        if (tensor.DType == DType.F32)
            return ((float*)tensor.DataPointer)[index];
        if (tensor.DType == DType.F16)
            return (float)((Half*)tensor.DataPointer)[index];
        if (tensor.DType == DType.BF16)
            return BitConverter.Int32BitsToSingle(((ushort*)tensor.DataPointer)[index] << 16);
        throw new NotSupportedException($"Test reader does not support {tensor.DType}.");
    }

    private void AssertClose(Tensor expected, Tensor actual, float tolerance, string operation)
    {
        Assert.Equal(expected.ElementCount, actual.ElementCount);
        float maxError = 0f;
        long maxIndex = 0;
        for (long i = 0; i < expected.ElementCount; i++)
        {
            float expectedValue = ReadFloat(expected, i);
            float actualValue = ReadFloat(actual, i);
            Assert.True(float.IsFinite(expectedValue), $"{operation}: reference is non-finite at {i}.");
            Assert.True(float.IsFinite(actualValue), $"{operation}: result is non-finite at {i}.");
            float error = MathF.Abs(expectedValue - actualValue);
            if (error > maxError)
            {
                maxError = error;
                maxIndex = i;
            }
        }

        _output.WriteLine($"{operation}: max error {maxError:E3} at {maxIndex} over {expected.ElementCount} elements");
        Assert.True(maxError <= tolerance, $"{operation}: max error {maxError:E3} exceeded {tolerance:E3} at {maxIndex}.");
    }

    /// <summary>GELU and Clamp use their typed launchers for non-block-aligned F16 and BF16 tensors.</summary>
    [Theory]
    [InlineData("F16")]
    [InlineData("BF16")]
    [Trait("Category", "GpuIntegration")]
    public void UnaryHalfPrecisionPaths_MatchF32Twins(string dtypeName)
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        DType dtype = HalfDtype(dtypeName);
        TensorShape shape = new TensorShape(3, 257);
        using Tensor original = RandomTensor(shape, 101, -6f, 6f);
        using Tensor halfInput = original.CastTo(dtype);
        using Tensor f32Input = halfInput.CastTo(DType.F32);
        using Tensor geluF32 = new Tensor(shape, DType.F32);
        using Tensor geluHalf = new Tensor(shape, dtype);
        using Tensor clampF32 = new Tensor(shape, DType.F32);
        using Tensor clampHalf = new Tensor(shape, dtype);
        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        IBackend backend = cuda;

        backend.Gelu(geluF32, f32Input);
        backend.Gelu(geluHalf, halfInput);
        backend.Clamp(clampF32, f32Input, -1.25f, 2.75f);
        backend.Clamp(clampHalf, halfInput, -1.25f, 2.75f);
        cuda.Sync();

        float geluTolerance = dtype == DType.F16 ? 8e-3f : 5e-2f;
        float clampTolerance = dtype == DType.F16 ? 1e-3f : 2e-2f;
        AssertClose(geluF32, geluHalf, geluTolerance, $"Gelu {dtypeName}");
        AssertClose(clampF32, clampHalf, clampTolerance, $"Clamp {dtypeName}");
    }

    /// <summary>GEGLU preserves the logical last-dimension split across multiple rows for F16 and BF16.</summary>
    [Theory]
    [InlineData("F16")]
    [InlineData("BF16")]
    [Trait("Category", "GpuIntegration")]
    public void GeGluHalfPrecisionPaths_MatchF32Twin(string dtypeName)
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        DType dtype = HalfDtype(dtypeName);
        TensorShape inputShape = new TensorShape(2, 3, 66);
        TensorShape outputShape = new TensorShape(2, 3, 33);
        using Tensor original = RandomTensor(inputShape, 202, -2f, 2f);
        using Tensor halfInput = original.CastTo(dtype);
        using Tensor f32Input = halfInput.CastTo(DType.F32);
        using Tensor f32Output = new Tensor(outputShape, DType.F32);
        using Tensor halfOutput = new Tensor(outputShape, dtype);
        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        IBackend backend = cuda;

        backend.GeGlu(f32Output, f32Input);
        backend.GeGlu(halfOutput, halfInput);
        cuda.Sync();

        float tolerance = dtype == DType.F16 ? 1e-2f : 5e-2f;
        AssertClose(f32Output, halfOutput, tolerance, $"GeGlu {dtypeName}");
    }

    /// <summary>KV repetition is a bit-preserving gather for both supported 16-bit floating-point formats.</summary>
    [Theory]
    [InlineData("F16")]
    [InlineData("BF16")]
    [Trait("Category", "GpuIntegration")]
    public void RepeatKvHalfPrecisionPaths_AreBitExact(string dtypeName)
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        const int Batch = 2;
        const int KvHeads = 3;
        const int GroupSize = 4;
        const int Sequence = 5;
        const int HeadDim = 7;
        DType dtype = HalfDtype(dtypeName);
        TensorShape inputShape = new TensorShape(Batch, KvHeads, Sequence, HeadDim);
        TensorShape outputShape = new TensorShape(Batch, KvHeads * GroupSize, Sequence, HeadDim);
        using Tensor input = new Tensor(inputShape, dtype);
        ushort[] patterns = [0x0000, 0x8000, 0x0001, 0x3c00, 0xbc00, 0x7c00, 0xfc00, 0x7e01, 0x7fc1, 0xffff];
        ushort* sourceBits = (ushort*)input.DataPointer;
        for (long i = 0; i < input.ElementCount; i++)
            sourceBits[i] = patterns[i % patterns.Length];
        using Tensor output = new Tensor(outputShape, dtype);
        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        IBackend backend = cuda;

        backend.RepeatKvHeads(output, input, KvHeads, GroupSize);
        cuda.Sync();

        ushort* inputBits = (ushort*)input.DataPointer;
        ushort* outputBits = (ushort*)output.DataPointer;
        for (int batch = 0; batch < Batch; batch++)
        {
            for (int outputHead = 0; outputHead < KvHeads * GroupSize; outputHead++)
            {
                int inputHead = outputHead / GroupSize;
                for (int sequence = 0; sequence < Sequence; sequence++)
                {
                    for (int dim = 0; dim < HeadDim; dim++)
                    {
                        long inputIndex = (((long)batch * KvHeads + inputHead) * Sequence + sequence) * HeadDim + dim;
                        long outputIndex = (((long)batch * KvHeads * GroupSize + outputHead) * Sequence + sequence) * HeadDim + dim;
                        Assert.Equal(inputBits[inputIndex], outputBits[outputIndex]);
                    }
                }
            }
        }
    }

    /// <summary>Unary activation contracts reject shape, dtype, unsupported-format, and bound mismatches before launch.</summary>
    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void UnaryContracts_RejectInvalidPairs()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        using Tensor input = new Tensor(new TensorShape(8), DType.F32);
        using Tensor wrongShape = new Tensor(new TensorShape(7), DType.F32);
        using Tensor wrongDtype = new Tensor(new TensorShape(8), DType.BF16);
        using Tensor i32Input = new Tensor(new TensorShape(8), DType.I32);
        using Tensor i32Output = new Tensor(new TensorShape(8), DType.I32);
        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        IBackend backend = cuda;

        Assert.Throws<HartsyInferenceException>(() => backend.Gelu(wrongShape, input));
        Assert.Throws<NotSupportedException>(() => backend.Gelu(wrongDtype, input));
        Assert.Throws<NotSupportedException>(() => backend.Gelu(i32Output, i32Input));
        Assert.Throws<HartsyInferenceException>(() => backend.Clamp(wrongShape, input, -1f, 1f));
        Assert.Throws<NotSupportedException>(() => backend.Clamp(wrongDtype, input, -1f, 1f));
        Assert.Throws<NotSupportedException>(() => backend.Clamp(i32Output, i32Input, -1f, 1f));
        Assert.Throws<ArgumentException>(() => backend.Clamp(input, input, 2f, -2f));
    }

    /// <summary>GEGLU rejects odd splits, wrong output geometry, mixed dtypes, and unsupported storage.</summary>
    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void GeGluContracts_RejectInvalidGeometryAndDtypes()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        using Tensor validInput = new Tensor(new TensorShape(2, 3, 10), DType.F32);
        using Tensor validOutput = new Tensor(new TensorShape(2, 3, 5), DType.F32);
        using Tensor oddInput = new Tensor(new TensorShape(2, 3, 9), DType.F32);
        using Tensor wrongPrefix = new Tensor(new TensorShape(3, 2, 5), DType.F32);
        using Tensor wrongLastDim = new Tensor(new TensorShape(2, 3, 4), DType.F32);
        using Tensor wrongDtype = new Tensor(new TensorShape(2, 3, 5), DType.BF16);
        using Tensor i32Input = new Tensor(new TensorShape(2, 3, 10), DType.I32);
        using Tensor i32Output = new Tensor(new TensorShape(2, 3, 5), DType.I32);
        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        IBackend backend = cuda;

        Assert.Throws<HartsyInferenceException>(() => backend.GeGlu(validOutput, oddInput));
        Assert.Throws<HartsyInferenceException>(() => backend.GeGlu(wrongPrefix, validInput));
        Assert.Throws<HartsyInferenceException>(() => backend.GeGlu(wrongLastDim, validInput));
        Assert.Throws<NotSupportedException>(() => backend.GeGlu(wrongDtype, validInput));
        Assert.Throws<NotSupportedException>(() => backend.GeGlu(i32Output, i32Input));
    }

    /// <summary>KV repetition rejects malformed rank, geometry, grouping, dtype, and output shape contracts.</summary>
    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void RepeatKvContracts_RejectInvalidGeometryAndDtypes()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        using Tensor input = new Tensor(new TensorShape(2, 3, 5, 7), DType.F32);
        using Tensor output = new Tensor(new TensorShape(2, 12, 5, 7), DType.F32);
        using Tensor rankThree = new Tensor(new TensorShape(3, 5, 7), DType.F32);
        using Tensor wrongOutput = new Tensor(new TensorShape(2, 11, 5, 7), DType.F32);
        using Tensor wrongDtype = new Tensor(new TensorShape(2, 12, 5, 7), DType.BF16);
        using Tensor i32Input = new Tensor(new TensorShape(2, 3, 5, 7), DType.I32);
        using Tensor i32Output = new Tensor(new TensorShape(2, 12, 5, 7), DType.I32);
        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        IBackend backend = cuda;

        Assert.Throws<HartsyInferenceException>(() => backend.RepeatKvHeads(output, rankThree, 3, 4));
        Assert.Throws<ArgumentOutOfRangeException>(() => backend.RepeatKvHeads(output, input, 3, 0));
        Assert.Throws<HartsyInferenceException>(() => backend.RepeatKvHeads(output, input, 2, 4));
        Assert.Throws<HartsyInferenceException>(() => backend.RepeatKvHeads(wrongOutput, input, 3, 4));
        Assert.Throws<NotSupportedException>(() => backend.RepeatKvHeads(wrongDtype, input, 3, 4));
        Assert.Throws<NotSupportedException>(() => backend.RepeatKvHeads(i32Output, i32Input, 3, 4));
    }
}
