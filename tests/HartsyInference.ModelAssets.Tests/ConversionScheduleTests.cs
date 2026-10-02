using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;
using HartsyInference.ModelAssets.BlockScale;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.Lora;
using HartsyInference.ModelAssets.MiniMaxH3;
using HartsyInference.ModelAssets.Nvfp4;
using Xunit;
using static HartsyInference.Tests.Common.CpuSchedules;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>The load-time weight conversions fan their rows, tiles and ranges out through <see cref="CpuParallel"/>.
/// Each gives the same bits whether it ran over every core, under a <c>numerics.cpuThreads</c> cap of 1, or inside an
/// <see cref="CpuParallel.InlineScope"/>; a row window decodes the same as those rows of a whole decode; and the fp8
/// quantizer, whose passes nest inside a loader's own parallel loop, finishes and agrees there at every cap. Every case
/// is big enough that the default schedule really fans out.</summary>
[Collection(CpuThreadsCollection.Name)]
public sealed class ConversionScheduleTests
{
    /// <summary>Over the fp8 passes' 2^19-element threshold and the cast's 2^20, and a whole number of neither's
    /// ranges, so the last range of each is short.</summary>
    private const int Count = (1 << 20) + 4099;

    private const string SeedKey = "blocks.3.ffn.w1.weight";

    [Theory]
    [InlineData(null)]
    [InlineData(SeedKey)]
    public void QuantizeF32ToFp8Scaled_GivesTheSameBytesAndScale_UnderEverySchedule(string? stochasticSeedKey)
    {
        float[] values = Values(Count, seed: 23);
        // The largest magnitude sits in the short last range and a NaN in the first, so the scale shows that every
        // range's maximum was kept and that NaN was ignored across ranges.
        values[Count - 7] = -9.5f;
        values[12_345] = float.NaN;

        (byte[] parallel, byte[] capped, byte[] inline) = UnderEverySchedule(() => Quantize(values, stochasticSeedKey));

        Assert.Equal(parallel, capped);
        Assert.Equal(parallel, inline);
        Assert.Equal(9.5f / 448f, BitConverter.ToSingle(parallel, Count));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(0)]
    public void Fp8QuantizationInsideAParallelLoop_CompletesAndAgrees_AtEveryCap(int cap)
    {
        int threads = cap == 0 ? Environment.ProcessorCount : cap;
        WithCpuThreads(threads, () => RunBounded(QuantizeInsideAParallelLoop));
    }

    [Fact]
    public void Fp8QuantizationInsideParallelLoops_OnTwoForeignThreads_OneOfThemInline_Completes()
    {
        RunBounded(QuantizeInsideAParallelLoop, () =>
        {
            using CpuParallel.InlineScope scope = CpuParallel.EnterInline();
            QuantizeInsideAParallelLoop();
        });
    }

    [Fact]
    public void DequantNvfp4ToF16AndToFp8_GiveTheSameBytes_UnderEverySchedule()
    {
        const int rows = 256, cols = 1024;
        using Tensor packed = RandomBytes(new TensorShape(rows, cols / 2), DType.U8, seed: 29);
        using Tensor blockScales = EveryByte(new TensorShape(rows, cols / 16), DType.F8E4M3);

        (byte[] parallel, byte[] capped, byte[] inline) = UnderEverySchedule<byte[]>(() =>
        {
            using Tensor f16 = CheckpointConvertUtils.DequantNvfp4ToF16(packed, blockScales, 0.37f);
            using Tensor fp8 = CheckpointConvertUtils.DequantNvfp4ToFp8(packed, blockScales, 0.37f);
            return [.. f16.AsReadOnlySpan<byte>(), .. fp8.AsReadOnlySpan<byte>(), .. BitConverter.GetBytes(fp8.Fp8ScaleFactor)];
        });

        Assert.Equal(parallel, capped);
        Assert.Equal(parallel, inline);
    }

    [Theory]
    [InlineData("fp8-e8m0-block")]
    [InlineData("mxfp4-e8m0")]
    [InlineData("modelopt-nvfp4")]
    [InlineData("affine-int4")]
    [InlineData("affine-int8")]
    public void BlockScaleCodecs_GiveTheSameBits_UnderEverySchedule_AndForARowWindow(string codec)
    {
        const int rows = 256, cols = 512;
        Random rng = new(codec.Length * 7);
        List<Tensor> owned = [];
        try
        {
            (byte[] packed, QuantRecipe recipe) = BlockScaleCase(codec, rows, cols, rng, owned);
            float[] Decode(long offset, long count)
            {
                float[] dest = new float[count * cols];
                switch (codec)
                {
                    case "fp8-e8m0-block": Fp8BlockE8M0Codec.DequantRows(packed, recipe, offset, count, dest); break;
                    case "mxfp4-e8m0": Mxfp4E8M0Codec.DequantRows(packed, recipe, offset, count, dest); break;
                    case "modelopt-nvfp4": ModelOptNvfp4Codec.DequantRows(packed, recipe, offset, count, dest); break;
                    default: AffineIntCodec.DequantRows(packed, recipe, offset, count, dest); break;
                }
                return dest;
            }

            (float[] parallel, float[] capped, float[] inline) = UnderEverySchedule(() => Decode(0, rows));
            float[] window = WithCpuThreads(0, () => Decode(rows / 2, rows / 2));

            AssertSameBits(parallel, capped, $"{codec}, cap 1");
            AssertSameBits(parallel, inline, $"{codec}, inline");
            AssertSameBits(parallel[(rows / 2 * cols)..], window, $"{codec}, rows {rows / 2}..{rows}");
        }
        finally
        {
            foreach (Tensor tensor in owned) tensor.Dispose();
        }
    }

    [Fact]
    public void Exl3_GivesTheSameBits_UnderEverySchedule_AndForARowWindow()
    {
        using Exl3FixtureData fixture = new();
        QuantRecipe recipe = fixture.Recipe();
        float[] Decode(long offset, long count)
        {
            float[] dest = new float[count * Exl3FixtureData.InDim];
            Exl3Codec.DequantRows(fixture.Trellis, recipe, offset, count, dest);
            return dest;
        }

        (float[] parallel, float[] capped, float[] inline) = UnderEverySchedule(() => Decode(0, Exl3FixtureData.OutDim));
        float[] window = WithCpuThreads(0, () => Decode(128, 256));

        AssertSameBits(parallel, capped, "cap 1");
        AssertSameBits(parallel, inline, "inline");
        AssertSameBits(parallel[(128 * Exl3FixtureData.InDim)..], window, "rows 128..384");
    }

    [Fact]
    public void Nvfp4ExpertBank_GivesTheSameBits_UnderEverySchedule_AndTheSliceIsTheBankExpertTransposed()
    {
        const int experts = 2, outDim = 256, inDim = 512;
        using Tensor weight = RandomBytes(new TensorShape(experts, outDim, inDim / 2), DType.U8, seed: 31);
        using Tensor blockScale = EveryByte(new TensorShape(experts, outDim, inDim / 16), DType.F8E4M3);
        blockScale.Fp8ScaleFactor = 0.5f;
        using Tensor globalScale = FromFloats([0.37f, 1.25f], new TensorShape(experts));

        (float[] bank, float[] bankCapped, float[] bankInline) =
            UnderEverySchedule(() => Floats(Nvfp4Codec.DequantExpert(weight, blockScale, globalScale)));
        (float[] slice, float[] sliceCapped, float[] sliceInline) = UnderEverySchedule(() =>
        {
            using Tensor destination = new(new TensorShape(outDim, inDim), DType.F32);
            Nvfp4Codec.DequantExpertSlice(weight, blockScale, globalScale, 1, destination);
            return destination.AsReadOnlySpan<float>().ToArray();
        });

        AssertSameBits(bank, bankCapped, "bank, cap 1");
        AssertSameBits(bank, bankInline, "bank, inline");
        AssertSameBits(slice, sliceCapped, "slice, cap 1");
        AssertSameBits(slice, sliceInline, "slice, inline");
        // The bank's tile loop writes each expert transposed from a per-tile scale buffer; the slice's row loop
        // writes it row-major with no buffer. Agreeing bit for bit shows the tile buffer is filled before it is read.
        float[] transposed = new float[outDim * inDim];
        int expertOne = inDim * outDim;
        for (int column = 0; column < inDim; column++)
        {
            for (int row = 0; row < outDim; row++) transposed[row * inDim + column] = bank[expertOne + column * outDim + row];
        }
        AssertSameBits(slice, transposed, "slice against the bank's expert 1");
    }

    [Fact]
    public void LoraBakerMatMulFma_GivesTheSameBits_UnderEverySchedule()
    {
        const int rows = 512, rank = 16, columns = 256;
        using Tensor up = FromFloats(Values(rows * rank, seed: 37), new TensorShape(rows, rank));
        using Tensor down = FromFloats(Values(rank * columns, seed: 41), new TensorShape(rank, columns));

        (float[] parallel, float[] capped, float[] inline) = UnderEverySchedule(() => Floats(LoraBaker.MatMulFma(up, down)));

        AssertSameBits(parallel, capped, "cap 1");
        AssertSameBits(parallel, inline, "inline");
    }

    [Fact]
    public void MiniMaxH3ControlRebase_GivesTheSameBits_UnderEverySchedule()
    {
        const int output = 512, dense = 64, curve = 8;
        using Tensor denseWeight = FromFloats(Values(output * dense, seed: 43), new TensorShape(output, dense));
        using Tensor denseBias = FromFloats(Values(output, seed: 47), new TensorShape(output));
        using Tensor intercept = FromFloats(Values(dense, seed: 53), new TensorShape(dense));
        using Tensor projection = FromFloats(Values(dense * curve, seed: 59), new TensorShape(dense, curve));
        using MiniMaxH3PddAffineBasis basis = new(intercept, projection, 1e-6);

        (float[] parallel, float[] capped, float[] inline) = UnderEverySchedule<float[]>(() =>
        {
            (Tensor weight, Tensor bias) = MiniMaxH3ControlNetPrunedConverter.RebaseProjection(denseWeight, denseBias, basis);
            using (weight)
            using (bias)
            {
                return [.. weight.AsReadOnlySpan<float>(), .. bias.AsReadOnlySpan<float>()];
            }
        });

        AssertSameBits(parallel, capped, "cap 1");
        AssertSameBits(parallel, inline, "inline");
    }

    [Fact]
    public void MiniMaxH3PddRebase_GivesTheSameBits_UnderEverySchedule()
    {
        const int output = 512, rank = 16, dense = 8, curve = 8;
        using Tensor intercept = FromFloats(Values(dense, seed: 61), new TensorShape(dense));
        using Tensor projection = FromFloats(Values(dense * curve, seed: 67), new TensorShape(dense, curve));
        using MiniMaxH3PddAffineBasis basis = new(intercept, projection, 0.0);
        using Tensor down = FromFloats(Values(rank * dense, seed: 71), new TensorShape(rank, dense));
        using Tensor up = FromFloats(Values(output * rank, seed: 73), new TensorShape(output, rank));
        LoraLayer layer = new()
        {
            TargetKey = "blocks.0.adaln_proj.linear.weight",
            Target = LoraTarget.Transformer,
            Delta = new StandardLoraDelta { Down = down, Up = up, Alpha = 4.0f },
        };

        (float[] parallel, float[] capped, float[] inline) = UnderEverySchedule<float[]>(() =>
        {
            using MiniMaxH3PddRebaseResult result = MiniMaxH3PddPrunedRebaser.Rebase([layer], basis, requireCompleteAdapter: false);
            LoraFullWeightDiff weight = Assert.Single(result.FullWeightDiffs, diff => !diff.IsBias);
            LoraFullWeightDiff bias = Assert.Single(result.FullWeightDiffs, diff => diff.IsBias);
            return [.. weight.Diff.AsReadOnlySpan<float>(), .. bias.Diff.AsReadOnlySpan<float>()];
        });

        AssertSameBits(parallel, capped, "cap 1");
        AssertSameBits(parallel, inline, "inline");
    }

    /// <summary>An outer loop that fans out, each of whose iterations quantizes a tensor big enough that its absmax,
    /// scale, stochastic-round and cast passes all fan out inside it.</summary>
    private static void QuantizeInsideAParallelLoop()
    {
        const int Iterations = 4;
        float[] values = Values(Count, seed: 79);
        byte[] expected = Quantize(values, SeedKey);
        for (int round = 0; round < 2; round++)
        {
            bool[] intact = new bool[Iterations];
            CpuParallel.For(Iterations, CpuParallel.MinWorkForParallel * Iterations, i =>
            {
                intact[i] = Quantize(values, SeedKey).AsSpan().SequenceEqual(expected);
            });
            Assert.DoesNotContain(false, intact);
        }
    }

    /// <summary>The quantized bytes followed by the scale's four, from a fresh copy of <paramref name="values"/>,
    /// which the quantizer overwrites.</summary>
    private static byte[] Quantize(float[] values, string? stochasticSeedKey)
    {
        using Tensor f32 = FromFloats(values, new TensorShape(values.Length));
        using Tensor fp8 = CheckpointConvertUtils.QuantizeF32ToFp8Scaled(f32, stochasticSeedKey);
        return [.. fp8.AsReadOnlySpan<byte>(), .. BitConverter.GetBytes(fp8.Fp8ScaleFactor)];
    }

    /// <summary>Packed bytes and a recipe for one block-scale codec; the recipe's tensors are added to
    /// <paramref name="owned"/>.</summary>
    private static (byte[] Packed, QuantRecipe Recipe) BlockScaleCase(string codec, int rows, int cols, Random rng, List<Tensor> owned)
    {
        switch (codec)
        {
            case "fp8-e8m0-block":
            {
                BlockGeometry geometry = new(128, 128);
                (long scaleRows, long scaleCols) = geometry.ScaleShape(rows, cols);
                Tensor scale = E8M0Scales(scaleRows, scaleCols, rng, owned);
                return (Bytes(rows * cols, rng), new QuantRecipe
                {
                    Encoding = QuantEncoding.Fp8E4M3BlockE8M0, Geometry = geometry, ScaleDType = scale.DType,
                    LogicalRows = rows, LogicalCols = cols, Scale = scale,
                });
            }
            case "mxfp4-e8m0":
            {
                Tensor scale = E8M0Scales(rows, cols / 32, rng, owned);
                return (Bytes(rows * cols / 2, rng), new QuantRecipe
                {
                    Encoding = QuantEncoding.Mxfp4E8M0, Geometry = new BlockGeometry(1, 32), ScaleDType = scale.DType,
                    LogicalRows = rows, LogicalCols = cols, Scale = scale,
                });
            }
            case "modelopt-nvfp4":
            {
                Tensor scale = EveryByte(new TensorShape(rows, cols / 16), DType.F8E4M3);
                owned.Add(scale);
                Tensor global = FromFloats([0.37f], new TensorShape(1));
                owned.Add(global);
                return (Bytes(rows * cols / 2, rng), new QuantRecipe
                {
                    Encoding = QuantEncoding.Nvfp4, Geometry = new BlockGeometry(1, 16), ScaleDType = DType.F8E4M3,
                    LogicalRows = rows, LogicalCols = cols, Scale = scale, GlobalScale = global,
                });
            }
            default:
            {
                int bits = codec == "affine-int4" ? 4 : 8;
                Tensor scale = FromFloats(Values(rows * (cols / 64), rng.Next()), new TensorShape(rows, cols / 64));
                owned.Add(scale);
                Tensor bias = FromFloats(Values(rows * (cols / 64), rng.Next()), new TensorShape(rows, cols / 64));
                owned.Add(bias);
                return (Bytes(rows * cols * bits / 8, rng), new QuantRecipe
                {
                    Encoding = bits == 4 ? QuantEncoding.AffineInt4 : QuantEncoding.AffineInt8, Geometry = new BlockGeometry(1, 64),
                    ScaleDType = DType.F32, LogicalRows = rows, LogicalCols = cols, Scale = scale, Bias = bias,
                });
            }
        }
    }

    /// <summary>E8M0 exponents around 2^0, so every decoded value stays finite.</summary>
    private static Tensor E8M0Scales(long rows, long cols, Random rng, List<Tensor> owned)
    {
        Tensor scale = new(new TensorShape(rows, cols), DType.F8E8M0);
        owned.Add(scale);
        Span<byte> bytes = scale.AsSpan<byte>();
        for (int i = 0; i < bytes.Length; i++) bytes[i] = (byte)rng.Next(96, 160);
        return scale;
    }

    private static float[] Values(int count, int seed)
    {
        Random rng = new(seed);
        float[] values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)((rng.NextDouble() - 0.5) * 4.0);
        return values;
    }

    private static byte[] Bytes(int count, Random rng)
    {
        byte[] bytes = new byte[count];
        rng.NextBytes(bytes);
        return bytes;
    }

    private static Tensor RandomBytes(TensorShape shape, DType dtype, int seed)
    {
        Tensor tensor = new(shape, dtype);
        new Random(seed).NextBytes(tensor.AsSpan<byte>());
        return tensor;
    }

    /// <summary>Every byte value in turn, so an fp8 scale covers subnormals, the maximum, NaN and the negative half.</summary>
    private static Tensor EveryByte(TensorShape shape, DType dtype)
    {
        Tensor tensor = new(shape, dtype);
        Span<byte> bytes = tensor.AsSpan<byte>();
        for (int i = 0; i < bytes.Length; i++) bytes[i] = (byte)(i & 0xFF);
        return tensor;
    }

    private static Tensor FromFloats(float[] values, TensorShape shape)
    {
        Tensor tensor = new(shape, DType.F32);
        values.CopyTo(tensor.AsSpan<float>());
        return tensor;
    }

    private static float[] Floats(Tensor tensor)
    {
        using (tensor)
        {
            return tensor.AsReadOnlySpan<float>().ToArray();
        }
    }

    private static void AssertSameBits(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        ReadOnlySpan<uint> want = MemoryMarshal.Cast<float, uint>(expected.AsSpan());
        ReadOnlySpan<uint> got = MemoryMarshal.Cast<float, uint>(actual.AsSpan());
        int same = want.CommonPrefixLength(got);
        Assert.True(same == want.Length, $"{what}: first difference at element {same} of {want.Length}");
    }
}
