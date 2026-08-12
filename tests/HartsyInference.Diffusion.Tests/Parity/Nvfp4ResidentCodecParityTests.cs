using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.BlockScale;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests.Parity;

/// <summary>Pins <see cref="Nvfp4ResidentCodec"/> — the host dequant a backend falls back to when its packed path
/// refuses a layer — to <c>Nvfp4Linear.DequantBf16Core</c>, the nvfp4 reference already validated against the
/// Qwen3-VL AWQ checkpoint. The two are separate transcriptions of the same arithmetic in two packages (Core cannot
/// reference ModelAssets), so nothing but a test keeps them from drifting; a drift here is silent, because a weight
/// dequantized slightly wrong still produces plausible output.
///
/// <para>Also guards the swizzle now having a single home: <see cref="BlockScaleSwizzle.SwizzledIndex"/> forwards to
/// <see cref="Nvfp4ResidentCodec.SwizzledScaleIndex"/>, and every pre-existing nvfp4/mxfp8 reader still calls the
/// former.</para></summary>
public sealed unsafe class Nvfp4ResidentCodecParityTests
{
    private readonly ITestOutputHelper _output;

    public Nvfp4ResidentCodecParityTests(ITestOutputHelper output) => _output = output;

    /// <summary>BF16 bit pattern of negative zero — the one value a cuBLAS readback cannot observe, so it is gated
    /// here on the host instead.</summary>
    private const ushort NegativeZeroBf16 = 0x8000;

    [Theory]
    [InlineData(100, 48)]     // rows pad 100->128 AND block columns pad 3->4
    [InlineData(100, 256)]    // padded rows only
    [InlineData(128, 48)]     // padded block columns only
    [InlineData(256, 256)]    // sweeps all 256 E4M3 scale bytes
    public void DequantToBf16_IsBitIdenticalToTheNvfp4LinearReference(int n, int k)
    {
        int paddedRows = (n + 127) / 128 * 128;
        int paddedCols = (k / Nvfp4ResidentCodec.GroupSize + 3) / 4 * 4;

        byte[] packedBytes = new byte[(long)n * (k / 2)];
        new Random(20260812 + n * 31 + k).NextBytes(packedBytes);
        byte[] scaleBytes = new byte[(long)paddedRows * paddedCols];
        // Every E4M3 byte value, so subnormals, the 480 maximum and the negative half are all exercised rather than
        // the narrow band a real checkpoint happens to land in.
        for (int i = 0; i < scaleBytes.Length; i++) scaleBytes[i] = (byte)(i & 0xFF);

        using Tensor packed = FromBytes(packedBytes, new TensorShape(n, k / 2), DType.U8);
        using Tensor blockScale = FromBytes(scaleBytes, new TensorShape(paddedRows, paddedCols), DType.F8E4M3);
        using Tensor globalScale = Scalar(0.37f);

        ushort[] reference = Nvfp4HostReference.Bf16Words(packed, blockScale, globalScale);

        using Tensor relabelled = packed.ReinterpretAs(DType.F4E2M1, new TensorShape(n, k));
        using Tensor actual = Nvfp4ResidentCodec.DequantToBf16(relabelled, blockScale, globalScale);
        Assert.Equal(DType.BF16, actual.DType);
        Assert.Equal((long)n, actual.Shape[0]);
        Assert.Equal((long)k, actual.Shape[1]);

        ReadOnlySpan<ushort> got = actual.AsReadOnlySpan<ushort>();
        long mismatches = 0, negativeZeros = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            if (got[i] != reference[i]) mismatches++;
            if (reference[i] == NegativeZeroBf16) negativeZeros++;
        }
        _output.WriteLine($"[{n}x{k}] mismatched BF16 words = {mismatches} / {reference.Length}; "
            + $"reference -0.0 words = {negativeZeros}");
        Assert.Equal(0L, mismatches);
        // E2M1 nibble 8 is -0.0, so a uniformly random packing must land there; zero of them would mean the case
        // never covered signed zero and the bit-exactness claim above is weaker than it reads.
        Assert.True(negativeZeros > 0, "no negative zero in the reference — the signed-zero case went uncovered.");
    }

    [Fact]
    public void SignedZero_SurvivesTheDequantWithItsSignFromTheBlockScale()
    {
        // One row, one 16-element block: nibbles 0 and 8 are +0.0 and -0.0, so the output sign is the block scale's.
        byte[] packedBytes = new byte[8];
        packedBytes[0] = 0x08;   // even element = +0.0, odd element = -0.0
        for (int i = 1; i < 8; i++) packedBytes[i] = 0x22;

        foreach ((byte scaleByte, ushort evenBits, ushort oddBits) in
                 new (byte, ushort, ushort)[] { (0x40, 0x0000, NegativeZeroBf16), (0xC0, NegativeZeroBf16, 0x0000) })
        {
            using Tensor packed = FromBytes(packedBytes, new TensorShape(1, 8), DType.U8);
            using Tensor blockScale = FromBytes([scaleByte, 0, 0, 0], new TensorShape(1, 4), DType.F8E4M3);
            using Tensor globalScale = Scalar(1f);
            using Tensor relabelled = packed.ReinterpretAs(DType.F4E2M1, new TensorShape(1, 16));
            using Tensor actual = Nvfp4ResidentCodec.DequantToBf16(relabelled, blockScale, globalScale);

            ReadOnlySpan<ushort> got = actual.AsReadOnlySpan<ushort>();
            _output.WriteLine($"block scale 0x{scaleByte:X2}: element0=0x{got[0]:X4} element1=0x{got[1]:X4}");
            Assert.Equal(evenBits, got[0]);
            Assert.Equal(oddBits, got[1]);
        }
    }

    [Theory]
    [InlineData(128, 4)]
    [InlineData(128, 16)]
    [InlineData(256, 8)]
    [InlineData(256, 20)]
    [InlineData(384, 12)]
    public void SwizzledScaleIndex_MatchesBlockScaleSwizzle_AndIsABijection(int paddedRows, int paddedCols)
    {
        bool[] hit = new bool[paddedRows * paddedCols];
        for (long row = 0; row < paddedRows; row++)
        {
            for (long blockColumn = 0; blockColumn < paddedCols; blockColumn++)
            {
                long core = Nvfp4ResidentCodec.SwizzledScaleIndex(row, blockColumn, paddedCols);
                Assert.Equal(BlockScaleSwizzle.SwizzledIndex(row, blockColumn, paddedCols), core);
                Assert.InRange(core, 0L, hit.Length - 1L);
                // A permutation, not just a bounded map: a collision would silently make two logical blocks share
                // one scale, which reads as mild quantization noise rather than as a bug.
                Assert.False(hit[core], $"({row}, {blockColumn}) collided at flat index {core}.");
                hit[core] = true;
            }
        }
        Assert.DoesNotContain(false, hit);
    }

    private static Tensor FromBytes(ReadOnlySpan<byte> source, TensorShape shape, DType dtype)
    {
        Tensor tensor = new Tensor(shape, dtype);
        source.CopyTo(new Span<byte>(tensor.DataPointer, source.Length));
        return tensor;
    }

    private static Tensor Scalar(float value)
    {
        Tensor tensor = new Tensor(new TensorShape(1), DType.F32);
        ((float*)tensor.DataPointer)[0] = value;
        return tensor;
    }
}
