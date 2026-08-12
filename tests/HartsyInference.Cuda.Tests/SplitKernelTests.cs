using System.Buffers.Binary;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>
/// Raw-payload, geometry, alias, and residency gates for Split. Random payload words deliberately include values
/// that are NaNs, infinities, and signed zeros under the floating-point interpretations: Split is a byte-exact
/// shape operation and must never canonicalize them.
/// </summary>
[Collection("CudaSerial")]
public sealed unsafe class SplitKernelTests
{
    private sealed record SplitCase(
        string Name,
        DType DType,
        TensorShape InputShape,
        int Dimension,
        long[] OutputDimensions);

    private readonly ITestOutputHelper _output;

    public SplitKernelTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    private static SplitCase[] Cases() =>
    [
        // Direct contiguous D2D/host-copy geometry.
        new("F32 rank-1 dim-0", DType.F32, new TensorShape(17), 0, [3, 5, 9]),
        // General arbitrary-axis kernel geometry, including unequal and singleton chunks.
        new("F32 middle axis", DType.F32, new TensorShape(3, 7, 5), 1, [2, 1, 4]),
        new("F16 max-rank interior axis", DType.F16, new TensorShape([2, 2, 3, 2, 3, 5]), 4, [1, 2]),
        new("BF16 last axis", DType.BF16, new TensorShape(3, 5, 11), 2, [4, 7]),
        // 65,537 logical blocks forces the capped 65,535-block kernel grid to execute its grid-stride tail.
        new("F16 logical-block grid tail", DType.F16, new TensorShape(65_537, 2), 1, [1, 1]),
        // Production VAE-style channel split: large contiguous slices select the bounded D2D fast path.
        new("BF16 VAE channel split", DType.BF16, new TensorShape(2, 8, 128, 128), 1, [3, 5]),
    ];

    [Fact]
    public void Cpu_AllSupportedDtypesRanksAndAxes_AreByteExact_AndPreserveInput()
    {
        using IBackend cpu = new CpuBackend();
        foreach (SplitCase splitCase in Cases())
            RunExactCase(cpu, null, splitCase);
    }

    [Fact]
    public void Cpu_StrictContractRejectsMalformedGeometryAndOverlappingStorage()
    {
        using IBackend cpu = new CpuBackend();
        AssertStrictContract(cpu);
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void Cuda_AllSupportedDtypesRanksAndAxes_AreByteExact_Resident_AndPreserveInput()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        using CudaBackend cuda = new(0, PtxDir());
        foreach (SplitCase splitCase in Cases())
            RunExactCase(cuda, cuda, splitCase);
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void Cuda_StrictContractRejectsMalformedGeometryAndOverlappingStorageBeforeDispatch()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        using CudaBackend cuda = new(0, PtxDir());
        AssertStrictContract(cuda);
    }

    private static void RunExactCase(IBackend backend, CudaBackend? cuda, SplitCase splitCase)
    {
        int byteCount = checked((int)(splitCase.InputShape.ElementCount * splitCase.DType.SizeInBytes));
        byte[] payload = PayloadBytes(
            byteCount, splitCase.Name.GetHashCode(StringComparison.Ordinal), splitCase.DType.SizeInBytes);
        byte[][] expected = SplitReference(
            payload,
            splitCase.InputShape,
            splitCase.Dimension,
            splitCase.OutputDimensions,
            splitCase.DType.SizeInBytes);

        using Tensor hostInput = TensorFromBytes(payload, splitCase.InputShape, splitCase.DType);
        using Tensor residentInput = new(splitCase.InputShape, splitCase.DType);
        Tensor input = hostInput;
        if (cuda is not null)
        {
            // A one-input dim-0 concat is an opaque D2D copy and makes residentInput device-authoritative without
            // performing floating-point arithmetic on the adversarial payload words.
            cuda.Concat(residentInput, [hostInput], 0);
            cuda.Sync();
            cuda.ResetD2hSyncCount();
            input = residentInput;
        }

        Tensor[] outputs = splitCase.OutputDimensions
            .Select(outputDimension => new Tensor(
                ShapeWithDimension(splitCase.InputShape, splitCase.Dimension, outputDimension),
                splitCase.DType))
            .ToArray();
        try
        {
            backend.Split(outputs, input, splitCase.Dimension);
            cuda?.Sync();
            if (cuda is not null)
                Assert.Equal(0, cuda.GetD2hSyncCount());

            for (int t = 0; t < outputs.Length; t++)
                AssertBytesEqual(expected[t], SnapshotBytes(outputs[t]), $"{splitCase.Name}, output {t}");
            AssertBytesEqual(payload, SnapshotBytes(input), $"{splitCase.Name}, input mutation");

            if (cuda is not null)
                Assert.Equal(outputs.Length + 1, cuda.GetD2hSyncCount());
        }
        finally
        {
            foreach (Tensor output in outputs)
                output.Dispose();
        }
    }

    private static void AssertStrictContract(IBackend backend)
    {
        using Tensor input = new(new TensorShape(2, 6, 3), DType.F32);
        using Tensor first = new(new TensorShape(2, 2, 3), DType.F32);
        using Tensor second = new(new TensorShape(2, 4, 3), DType.F32);
        using Tensor f16First = new(first.Shape, DType.F16);
        using Tensor rankMismatch = new(new TensorShape(2, 2), DType.F32);
        using Tensor nonsplitMismatch = new(new TensorShape(3, 2, 3), DType.F32);
        using Tensor badSum = new(new TensorShape(2, 3, 3), DType.F32);
        using Tensor zeroChunk = new(new TensorShape(2, 0, 3), DType.F32);

        Assert.Throws<ArgumentNullException>(() => backend.Split([first, second], null!, 1));
        Assert.Throws<ArgumentException>(() => backend.Split(Array.Empty<Tensor>(), input, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => backend.Split([first, second], input, -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => backend.Split([first, second], input, input.Shape.Rank));
        Assert.Throws<ArgumentNullException>(() => backend.Split([first, null!, second], input, 1));
        Assert.Throws<ArgumentException>(() => backend.Split([f16First, second], input, 1));
        Assert.Throws<ArgumentException>(() => backend.Split([rankMismatch, second], input, 1));
        Assert.Throws<ArgumentException>(() => backend.Split([nonsplitMismatch, second], input, 1));
        Assert.Throws<ArgumentException>(() => backend.Split([badSum, first], input, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => backend.Split([zeroChunk, input], input, 1));
        Assert.Throws<ArgumentException>(() => backend.Split([input], input, 1));

        using Tensor sharedOutput = new(first.Shape, DType.F32);
        using Tensor mutualAliasInput = new(new TensorShape(2, 4, 3), DType.F32);
        Assert.Throws<ArgumentException>(() => backend.Split([sharedOutput, sharedOutput],
            mutualAliasInput, 1));

        // Separate Tensor objects over the same/overlapping host storage must be rejected, not merely identical
        // object references. These checks happen before an implementation is allowed to touch payload data.
        using Tensor inputAlias = new(input.DataPointer, input.Shape, input.DType);
        Assert.Throws<ArgumentException>(() => backend.Split([inputAlias], input, 1));

        using Tensor overlapBacking = new(new TensorShape(24), DType.F32);
        byte* overlapBase = (byte*)overlapBacking.DataPointer;
        using Tensor overlapFirst = new(overlapBase, badSum.Shape, DType.F32);
        using Tensor overlapSecond = new(overlapBase + sizeof(float), badSum.Shape, DType.F32);
        Assert.Throws<ArgumentException>(() => backend.Split(
            [overlapFirst, overlapSecond], input, 1));

        using Tensor i32Input = new(new TensorShape(4), DType.I32);
        using Tensor i32Output = new(i32Input.Shape, DType.I32);
        Assert.Throws<NotSupportedException>(() => backend.Split([i32Output], i32Input, 0));

        // TensorShape currently stores unchecked products; Split must independently reject both element-count
        // overflow and a positive element count whose byte span exceeds signed 64-bit addressability.
        using Tensor productOverflow = new((void*)1, new TensorShape(long.MaxValue, 2), DType.F32);
        Assert.Throws<ArgumentOutOfRangeException>(() => backend.Split([first], productOverflow, 0));
        using Tensor byteSpanOverflow = new((void*)1, new TensorShape(long.MaxValue / sizeof(float) + 1), DType.F32);
        Assert.Throws<ArgumentOutOfRangeException>(() => backend.Split([first], byteSpanOverflow, 0));
    }

    private static TensorShape ShapeWithDimension(TensorShape input, int dimension, long size)
    {
        Span<long> dimensions = stackalloc long[input.Rank];
        input.CopyDimsTo(dimensions);
        dimensions[dimension] = size;
        return new TensorShape(dimensions);
    }

    private static byte[][] SplitReference(
        byte[] input,
        TensorShape inputShape,
        int dimension,
        long[] outputDimensions,
        int elementSize)
    {
        long outer = 1, inner = 1;
        for (int axis = 0; axis < dimension; axis++)
            outer = checked(outer * inputShape[axis]);
        for (int axis = dimension + 1; axis < inputShape.Rank; axis++)
            inner = checked(inner * inputShape[axis]);

        byte[][] outputs = outputDimensions
            .Select(size => new byte[checked((int)(outer * size * inner * elementSize))])
            .ToArray();
        long splitOffset = 0;
        for (int t = 0; t < outputs.Length; t++)
        {
            long sliceElements = checked(outputDimensions[t] * inner);
            int sliceBytes = checked((int)(sliceElements * elementSize));
            for (long outerIndex = 0; outerIndex < outer; outerIndex++)
            {
                int sourceByteOffset = checked((int)(
                    ((outerIndex * inputShape[dimension] + splitOffset) * inner) * elementSize));
                int outputByteOffset = checked((int)(outerIndex * sliceBytes));
                Buffer.BlockCopy(input, sourceByteOffset, outputs[t], outputByteOffset, sliceBytes);
            }
            splitOffset += outputDimensions[t];
        }
        return outputs;
    }

    private static byte[] PayloadBytes(int byteCount, int seed, int elementSize)
    {
        byte[] payload = new byte[byteCount];
        uint state = unchecked((uint)seed) | 1u;
        for (int i = 0; i < payload.Length; i++)
        {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            payload[i] = (byte)state;
        }
        if (elementSize == sizeof(uint) && payload.Length >= 3 * sizeof(uint))
        {
            BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(0, 4), 0x8000_0000u); // F32 -0
            BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(4, 4), 0x7fc1_2345u); // F32 NaN payload
            BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(8, 4), 0xff80_0000u); // F32 -infinity
        }
        else if (elementSize == sizeof(ushort) && payload.Length >= 3 * sizeof(ushort))
        {
            BinaryPrimitives.WriteUInt16LittleEndian(payload.AsSpan(0, 2), 0x8000); // F16/BF16 -0
            BinaryPrimitives.WriteUInt16LittleEndian(payload.AsSpan(2, 2), 0x7e21); // F16 NaN payload
            BinaryPrimitives.WriteUInt16LittleEndian(payload.AsSpan(4, 2), 0x7fc1); // BF16 NaN payload
        }
        return payload;
    }

    private static Tensor TensorFromBytes(byte[] payload, TensorShape shape, DType dtype)
    {
        Assert.Equal(checked(shape.ElementCount * dtype.SizeInBytes), payload.LongLength);
        Tensor tensor = new(shape, dtype);
        payload.CopyTo(tensor.AsSpan<byte>());
        return tensor;
    }

    private static byte[] SnapshotBytes(Tensor tensor) => tensor.AsReadOnlySpan<byte>().ToArray();

    private static void AssertBytesEqual(byte[] expected, byte[] actual, string description)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(expected[i] == actual[i],
                $"{description}: byte mismatch at {i}: expected 0x{expected[i]:x2}, actual 0x{actual[i]:x2}.");
    }
}
