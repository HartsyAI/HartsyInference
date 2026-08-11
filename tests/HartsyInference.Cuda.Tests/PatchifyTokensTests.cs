using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>
/// Contract, bit-exact layout, batching, inverse, and device-residency gates for the shared DiT patch shuffle.
/// Values are compared as raw F32/F16/BF16 payload bits because patchify/unpatchify must perform no arithmetic.
/// </summary>
[Collection("CudaSerial")]
public sealed unsafe class PatchifyTokensTests
{
    private readonly ITestOutputHelper _output;

    public PatchifyTokensTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    public static TheoryData<int, int, int, int, int, bool> Geometries() => new()
    {
        // p=1 plus an odd HxW grid; both layouts intentionally remain in the matrix even though p=1 converges.
        { 1, 3, 5,   7,   1, true  },
        { 1, 3, 5,   7,   1, false },
        // B>1 and odd 3x5 packed grid.
        { 2, 5, 6,   10,  2, true  },
        { 2, 5, 6,   10,  2, false },
        // p=4, B>1, and odd 3x5 packed grid.
        { 3, 2, 12,  20,  4, true  },
        { 3, 2, 12,  20,  4, false },
        // Sanitizer-friendly production-scale tail: 628,320 elements, 2,455 blocks, odd 33x35 grid.
        { 2, 17, 132, 140, 4, true  },
        { 2, 17, 132, 140, 4, false },
    };

    [Theory]
    [MemberData(nameof(Geometries))]
    public void CpuFallback_MatchesIndependentPatchAndUnpatchOracles(
        int batch, int channels, int height, int width, int patch, bool innerChannelFastest)
    {
        using IBackend cpu = new CpuBackend();
        RunIndependentOracleCase(cpu, null, batch, channels, height, width, patch, innerChannelFastest);
    }

    [Theory]
    [MemberData(nameof(Geometries))]
    [Trait("Category", "GpuIntegration")]
    public void Cuda_MatchesIndependentPatchAndUnpatchOracles_WithoutIntermediateReadback(
        int batch, int channels, int height, int width, int patch, bool innerChannelFastest)
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        using CudaBackend cuda = new(0, PtxDir());
        RunIndependentOracleCase(cuda, cuda, batch, channels, height, width, patch, innerChannelFastest);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    [Trait("Category", "GpuIntegration")]
    public void Cuda_BatchedPatchifyThenUnpatchify_RoundTripsResident(bool innerChannelFastest)
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        const int batch = 2, channels = 7, height = 12, width = 20, patch = 4;
        uint[] original = Payloads(batch * channels * height * width);
        using Tensor input = TensorFromBits(original, new TensorShape(batch, channels, height, width));
        using Tensor tokens = new(new TensorShape(batch, (height / patch) * (width / patch), channels * patch * patch), DType.F32);
        using Tensor restored = new(input.Shape, DType.F32);
        using CudaBackend cuda = new(0, PtxDir());

        cuda.ResetD2hSyncCount();
        cuda.PatchifyTokens(tokens, input, patch, innerChannelFastest);
        cuda.UnpatchifyTokens(restored, tokens, channels, height / patch, width / patch, patch, innerChannelFastest);
        cuda.Sync();
        Assert.Equal(0, cuda.GetD2hSyncCount());

        AssertBitsEqual(original, SnapshotBits(restored), "resident patchify/unpatchify round trip");
        Assert.Equal(1, cuda.GetD2hSyncCount());
    }

    [Theory]
    [InlineData(false, true)]
    [InlineData(false, false)]
    [InlineData(true, true)]
    [InlineData(true, false)]
    [Trait("Category", "GpuIntegration")]
    public void F16AndBf16_PreserveEveryPayloadBit_OnCpuAndCuda(bool bf16, bool innerChannelFastest)
    {
        DType dtype = bf16 ? DType.BF16 : DType.F16;
        using (IBackend cpu = new CpuBackend())
            RunU16OracleAndRoundTrip(cpu, null, dtype, innerChannelFastest);

        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED CUDA half-word case: CUDA unavailable");
            return;
        }
        using CudaBackend cuda = new(0, PtxDir());
        RunU16OracleAndRoundTrip(cuda, cuda, dtype, innerChannelFastest);
    }

    [Fact]
    public void MalformedAndOverflowingContracts_AreRejectedBeforeDataAccessOrDispatch()
    {
        using IBackend cpu = new CpuBackend();
        AssertMalformedContracts(cpu);

        if (!CudaContext.IsAvailable()) return;
        using CudaBackend cuda = new(0, PtxDir());
        AssertMalformedContracts(cuda);
    }

    private void RunIndependentOracleCase(
        IBackend backend,
        CudaBackend? cuda,
        int batch,
        int channels,
        int height,
        int width,
        int patch,
        bool innerChannelFastest)
    {
        int hPacked = height / patch;
        int wPacked = width / patch;
        int sequenceLength = hPacked * wPacked;
        int patchVolume = channels * patch * patch;
        uint[] inputBits = Payloads(batch * channels * height * width);
        uint[] expectedTokens = PatchOracle(
            inputBits, batch, channels, height, width, patch, innerChannelFastest);

        using Tensor input = TensorFromBits(inputBits, new TensorShape(batch, channels, height, width));
        using Tensor tokens = new(new TensorShape(batch, sequenceLength, patchVolume), DType.F32);

        cuda?.ResetD2hSyncCount();
        backend.PatchifyTokens(tokens, input, patch, innerChannelFastest);
        cuda?.Sync();
        if (cuda is not null) Assert.Equal(0, cuda.GetD2hSyncCount());
        AssertBitsEqual(expectedTokens, SnapshotBits(tokens), "patchify vs independent oracle");
        if (cuda is not null) Assert.Equal(1, cuda.GetD2hSyncCount());

        // Feed an independently constructed token tensor to unpatchify. This prevents a shared forward/inverse
        // addressing error from passing merely because the two kernels happen to undo the same wrong permutation.
        using Tensor oracleTokens = TensorFromBits(expectedTokens, tokens.Shape);
        using Tensor restored = new(input.Shape, DType.F32);
        cuda?.ResetD2hSyncCount();
        backend.UnpatchifyTokens(
            restored, oracleTokens, channels, hPacked, wPacked, patch, innerChannelFastest);
        cuda?.Sync();
        if (cuda is not null) Assert.Equal(0, cuda.GetD2hSyncCount());
        AssertBitsEqual(inputBits, SnapshotBits(restored), "unpatchify vs independent oracle");
        if (cuda is not null) Assert.Equal(1, cuda.GetD2hSyncCount());

        _output.WriteLine(
            $"B={batch}, C={channels}, HxW={height}x{width}, p={patch}, " +
            $"layout={(innerChannelFastest ? "ph,pw,c" : "c,ph,pw")}: {inputBits.Length} raw words exact");
    }

    private static void AssertMalformedContracts(IBackend backend)
    {
        using Tensor validInput = new(new TensorShape(2, 3, 6, 10), DType.F32);
        using Tensor validTokens = new(new TensorShape(2, 15, 12), DType.F32);
        using Tensor validOutput = new(validInput.Shape, DType.F32);
        using Tensor rank3Input = new(new TensorShape(2, 3, 60), DType.F32);
        using Tensor rank4Tokens = new(new TensorShape(2, 3, 3, 4), DType.F32);
        using Tensor wrongTokenShape = new(new TensorShape(2, 14, 12), DType.F32);
        using Tensor wrongPatchVolume = new(new TensorShape(2, 15, 13), DType.F32);
        using Tensor wrongOutputBatch = new(new TensorShape(1, 3, 6, 10), DType.F32);
        using Tensor nonDivisible = new(new TensorShape(1, 3, 5, 8), DType.F32);
        using Tensor arbitraryTokens = new(new TensorShape(1, 1, 1), DType.F32);
        using Tensor emptyInput = new(new TensorShape(0, 3, 6, 10), DType.F32);
        using Tensor f16Input = new(validInput.Shape, DType.F16);
        using Tensor f16Tokens = new(validTokens.Shape, DType.F16);
        using Tensor i32Input = new(validInput.Shape, DType.I32);
        using Tensor i32Tokens = new(validTokens.Shape, DType.I32);
        using Tensor i32Output = new(validOutput.Shape, DType.I32);

        Assert.Throws<ArgumentException>(() => backend.PatchifyTokens(validInput, validInput, 2, true));
        Assert.Throws<ArgumentException>(() => backend.PatchifyTokens(validTokens, rank3Input, 2, true));
        Assert.Throws<ArgumentException>(() => backend.PatchifyTokens(rank4Tokens, validInput, 2, true));
        Assert.Throws<ArgumentException>(() => backend.PatchifyTokens(wrongTokenShape, validInput, 2, true));
        Assert.Throws<ArgumentException>(() => backend.PatchifyTokens(wrongPatchVolume, validInput, 2, true));
        Assert.Throws<ArgumentException>(() => backend.PatchifyTokens(arbitraryTokens, nonDivisible, 2, true));
        Assert.Throws<ArgumentOutOfRangeException>(() => backend.PatchifyTokens(arbitraryTokens, emptyInput, 2, true));
        Assert.Throws<ArgumentOutOfRangeException>(() => backend.PatchifyTokens(validTokens, validInput, 0, true));
        Assert.Throws<ArgumentException>(() => backend.PatchifyTokens(validTokens, f16Input, 2, true));
        Assert.Throws<ArgumentException>(() => backend.PatchifyTokens(f16Tokens, validInput, 2, true));
        Assert.Throws<NotSupportedException>(() => backend.PatchifyTokens(i32Tokens, i32Input, 2, true));

        // Distinct borrowed views over overlapping host storage are aliases too. Identity-only validation would
        // let the host shuffle overwrite unread input and make accelerator behavior diverge from the CPU fallback.
        using Tensor overlapBacking = new(new TensorShape(validInput.ElementCount + 1), DType.F32);
        using Tensor overlappingInput = new(overlapBacking.DataPointer, validInput.Shape, DType.F32);
        using Tensor overlappingTokens = new(
            (byte*)overlapBacking.DataPointer + sizeof(float), validTokens.Shape, DType.F32);
        Assert.Throws<ArgumentException>(() =>
            backend.PatchifyTokens(overlappingTokens, overlappingInput, 2, true));

        Assert.Throws<ArgumentException>(() => backend.UnpatchifyTokens(validTokens, validTokens, 3, 3, 5, 2, true));
        Assert.Throws<ArgumentException>(() => backend.UnpatchifyTokens(validOutput, rank4Tokens, 3, 3, 5, 2, true));
        Assert.Throws<ArgumentException>(() => backend.UnpatchifyTokens(wrongOutputBatch, validTokens, 3, 3, 5, 2, true));
        Assert.Throws<ArgumentException>(() => backend.UnpatchifyTokens(validOutput, wrongTokenShape, 3, 3, 5, 2, true));
        Assert.Throws<ArgumentException>(() => backend.UnpatchifyTokens(validOutput, wrongPatchVolume, 3, 3, 5, 2, true));
        Assert.Throws<ArgumentOutOfRangeException>(() => backend.UnpatchifyTokens(validOutput, validTokens, 0, 3, 5, 2, true));
        Assert.Throws<ArgumentOutOfRangeException>(() => backend.UnpatchifyTokens(validOutput, validTokens, 3, 0, 5, 2, true));
        Assert.Throws<ArgumentOutOfRangeException>(() => backend.UnpatchifyTokens(validOutput, validTokens, 3, 3, 0, 2, true));
        Assert.Throws<ArgumentOutOfRangeException>(() => backend.UnpatchifyTokens(validOutput, validTokens, 3, 3, 5, 0, true));
        Assert.Throws<ArgumentException>(() => backend.UnpatchifyTokens(validOutput, f16Tokens, 3, 3, 5, 2, true));
        Assert.Throws<NotSupportedException>(() => backend.UnpatchifyTokens(i32Output, i32Tokens, 3, 3, 5, 2, true));
        Assert.Throws<ArgumentException>(() =>
            backend.UnpatchifyTokens(overlappingInput, overlappingTokens, 3, 3, 5, 2, true));

        // Borrowed one-byte sentinels prove oversized contracts fail entirely in validation: no allocation and no
        // pointer dereference are possible. The first crosses a kernel's signed dimension bound; the second makes
        // B*C*H*W overflow Int64 even though each individual axis still fits Int32.
        using Tensor oversizedAxis = Borrowed(new TensorShape(1, (long)int.MaxValue + 1, 1, 1));
        using Tensor dummyTokens = Borrowed(new TensorShape(1, 1, 1), pointer: 2);
        Assert.Throws<ArgumentOutOfRangeException>(() => backend.PatchifyTokens(dummyTokens, oversizedAxis, 1, true));

        using Tensor productOverflow = Borrowed(new TensorShape(int.MaxValue, int.MaxValue, 3, 1), pointer: 3);
        Assert.Throws<ArgumentOutOfRangeException>(() => backend.PatchifyTokens(dummyTokens, productOverflow, 1, true));

        using Tensor byteSpanOverflow = Borrowed(new TensorShape(int.MaxValue, int.MaxValue, 1, 1), pointer: 4);
        using Tensor byteSpanTokens = Borrowed(new TensorShape(int.MaxValue, 1, int.MaxValue), pointer: 5);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            backend.PatchifyTokens(byteSpanTokens, byteSpanOverflow, 1, true));

        using Tensor dummyOutput = Borrowed(new TensorShape(1, 1, 1, 1), pointer: 6);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            backend.UnpatchifyTokens(dummyOutput, dummyTokens, 1, int.MaxValue, 1, 2, true));
    }

    private void RunU16OracleAndRoundTrip(
        IBackend backend, CudaBackend? cuda, DType dtype, bool innerChannelFastest)
    {
        const int batch = 2, channels = 3, height = 6, width = 10, patch = 2;
        int hPacked = height / patch, wPacked = width / patch;
        TensorShape inputShape = new(batch, channels, height, width);
        TensorShape tokenShape = new(batch, hPacked * wPacked, channels * patch * patch);
        ushort[] inputBits = U16Payloads((int)inputShape.ElementCount);
        ushort[] expectedTokens = PatchOracle(
            inputBits, batch, channels, height, width, patch, innerChannelFastest);

        using Tensor input = TensorFromBits(inputBits, inputShape, dtype);
        using Tensor tokens = new(tokenShape, dtype);
        cuda?.ResetD2hSyncCount();
        backend.PatchifyTokens(tokens, input, patch, innerChannelFastest);
        cuda?.Sync();
        if (cuda is not null) Assert.Equal(0, cuda.GetD2hSyncCount());
        AssertBitsEqual(expectedTokens, SnapshotU16Bits(tokens), $"{dtype} patchify oracle");

        using Tensor oracleTokens = TensorFromBits(expectedTokens, tokenShape, dtype);
        using Tensor restoredFromOracle = new(inputShape, dtype);
        cuda?.ResetD2hSyncCount();
        backend.UnpatchifyTokens(
            restoredFromOracle, oracleTokens, channels, hPacked, wPacked, patch, innerChannelFastest);
        cuda?.Sync();
        if (cuda is not null) Assert.Equal(0, cuda.GetD2hSyncCount());
        AssertBitsEqual(inputBits, SnapshotU16Bits(restoredFromOracle), $"{dtype} unpatchify oracle");

        using Tensor directTokens = new(tokenShape, dtype);
        using Tensor roundTrip = new(inputShape, dtype);
        cuda?.ResetD2hSyncCount();
        backend.PatchifyTokens(directTokens, input, patch, innerChannelFastest);
        backend.UnpatchifyTokens(roundTrip, directTokens, channels, hPacked, wPacked, patch, innerChannelFastest);
        cuda?.Sync();
        if (cuda is not null) Assert.Equal(0, cuda.GetD2hSyncCount());
        AssertBitsEqual(inputBits, SnapshotU16Bits(roundTrip), $"{dtype} resident round trip");
    }

    private static T[] PatchOracle<T>(
        T[] input,
        int batch,
        int channels,
        int height,
        int width,
        int patch,
        bool innerChannelFastest)
    {
        int hPacked = height / patch, wPacked = width / patch;
        int sequenceLength = hPacked * wPacked, patchVolume = channels * patch * patch;
        T[] output = new T[input.Length];
        for (int b = 0; b < batch; b++)
            for (int hp = 0; hp < hPacked; hp++)
                for (int wp = 0; wp < wPacked; wp++)
                {
                    int tokenBase = (b * sequenceLength + hp * wPacked + wp) * patchVolume;
                    int feature = 0;
                    if (innerChannelFastest)
                    {
                        for (int ph = 0; ph < patch; ph++)
                            for (int pw = 0; pw < patch; pw++)
                                for (int c = 0; c < channels; c++)
                                    output[tokenBase + feature++] = input[
                                        ((b * channels + c) * height + hp * patch + ph) * width + wp * patch + pw];
                    }
                    else
                    {
                        for (int c = 0; c < channels; c++)
                            for (int ph = 0; ph < patch; ph++)
                                for (int pw = 0; pw < patch; pw++)
                                    output[tokenBase + feature++] = input[
                                        ((b * channels + c) * height + hp * patch + ph) * width + wp * patch + pw];
                    }
                }
        return output;
    }

    private static uint[] Payloads(int count)
    {
        uint[] bits = new uint[count];
        uint x = 0x9e3779b9u;
        for (int i = 0; i < bits.Length; i++)
        {
            // Xorshift payloads deliberately include arbitrary exponent/significand fields. The shuffle is tested
            // as raw words, so NaNs, infinities, subnormals, and signed zero cannot be canonicalized unnoticed.
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            bits[i] = x;
        }
        if (bits.Length > 3)
        {
            bits[0] = 0x80000000u; // -0
            bits[1] = 0x7fc12345u; // quiet NaN with payload
            bits[2] = 0xff800000u; // -infinity
        }
        return bits;
    }

    private static ushort[] U16Payloads(int count)
    {
        ushort[] bits = new ushort[count];
        uint x = 0x6d2b79f5u;
        for (int i = 0; i < bits.Length; i++)
        {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            bits[i] = (ushort)x;
        }
        if (bits.Length > 3)
        {
            bits[0] = 0x8000; // F16/BF16 -0
            bits[1] = 0x7e21; // F16 quiet NaN payload (also an arbitrary BF16 payload)
            bits[2] = 0x7fc1; // BF16 quiet NaN payload (also an arbitrary F16 payload)
        }
        return bits;
    }

    private static Tensor TensorFromBits(uint[] bits, TensorShape shape)
    {
        Assert.Equal(shape.ElementCount, bits.LongLength);
        Tensor tensor = new(shape, DType.F32);
        uint* dst = (uint*)tensor.DataPointer;
        for (int i = 0; i < bits.Length; i++) dst[i] = bits[i];
        return tensor;
    }

    private static Tensor TensorFromBits(ushort[] bits, TensorShape shape, DType dtype)
    {
        Assert.True(dtype == DType.F16 || dtype == DType.BF16);
        Assert.Equal(shape.ElementCount, bits.LongLength);
        Tensor tensor = new(shape, dtype);
        ushort* dst = (ushort*)tensor.DataPointer;
        for (int i = 0; i < bits.Length; i++) dst[i] = bits[i];
        return tensor;
    }

    private static uint[] SnapshotBits(Tensor tensor)
    {
        uint[] bits = new uint[checked((int)tensor.ElementCount)];
        uint* src = (uint*)tensor.DataPointer;
        for (int i = 0; i < bits.Length; i++) bits[i] = src[i];
        return bits;
    }

    private static ushort[] SnapshotU16Bits(Tensor tensor)
    {
        ushort[] bits = new ushort[checked((int)tensor.ElementCount)];
        ushort* src = (ushort*)tensor.DataPointer;
        for (int i = 0; i < bits.Length; i++) bits[i] = src[i];
        return bits;
    }

    private static void AssertBitsEqual(uint[] expected, uint[] actual, string operation)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(expected[i] == actual[i],
                $"{operation}: raw word mismatch at {i}: expected 0x{expected[i]:x8}, actual 0x{actual[i]:x8}.");
    }

    private static void AssertBitsEqual(ushort[] expected, ushort[] actual, string operation)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(expected[i] == actual[i],
                $"{operation}: raw half-word mismatch at {i}: expected 0x{expected[i]:x4}, actual 0x{actual[i]:x4}.");
    }

    private static Tensor Borrowed(TensorShape shape, nuint pointer = 1)
        => new((void*)pointer, shape, DType.F32);
}
