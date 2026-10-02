using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.ModelAssets.Nvfp4;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;
using static HartsyInference.Tests.Common.CpuSchedules;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The Diffusion host loops that fan out through <see cref="CpuParallel"/> give the same bits whether they
/// ran over every core, under a <c>numerics.cpuThreads</c> cap of 1, or inside an
/// <see cref="CpuParallel.InlineScope"/>, and match a reference that does not go through the same split: FluxRope's
/// rotation of every head at once against one head at a time, <c>Nvfp4Linear</c>'s BF16 dequant against
/// <see cref="Nvfp4ResidentCodec"/>, and GPT-OSS's CPU expert lanes, whose output must not depend on which lane ran
/// which expert. Each case is big enough that the default schedule really fans out.</summary>
public sealed class HostFanOutScheduleTests
{
    private const int TextTokens = 16, Side = 24, Heads = 8, HeadDim = 128;

    private readonly ITestOutputHelper _output;

    public HostFanOutScheduleTests(ITestOutputHelper output) => _output = output;

    /// <summary>Text plus a 24×24 image grid: under one 1024-vector range per head, so the one-head reference takes
    /// the serial path, while all eight heads together span several ranges.</summary>
    private const int Sequence = TextTokens + Side * Side;

    [Fact]
    public void FluxRope_Forward_AllHeadsAtOnce_MatchesOneHeadAtATime_UnderEverySchedule()
    {
        FluxRope rope = PreparedRope();
        float[] q = Values(Heads * Sequence * HeadDim, seed: 3);
        float[] k = Values(Heads * Sequence * HeadDim, seed: 5);
        float[] expected = new float[2 * q.Length];
        int head = Sequence * HeadDim;
        for (int h = 0; h < Heads; h++)
        {
            using Tensor oneQ = FromFloats(q.AsSpan(h * head, head), new TensorShape(1, 1, Sequence, HeadDim));
            using Tensor oneK = FromFloats(k.AsSpan(h * head, head), new TensorShape(1, 1, Sequence, HeadDim));
            rope.Forward(oneQ, oneK, 1, 1, Sequence);
            oneQ.AsReadOnlySpan<float>().CopyTo(expected.AsSpan(h * head));
            oneK.AsReadOnlySpan<float>().CopyTo(expected.AsSpan(q.Length + h * head));
        }

        (float[] parallel, float[] capped, float[] inline) = UnderEverySchedule<float[]>(() =>
        {
            using Tensor allQ = FromFloats(q, new TensorShape(1, Heads, Sequence, HeadDim));
            using Tensor allK = FromFloats(k, new TensorShape(1, Heads, Sequence, HeadDim));
            rope.Forward(allQ, allK, 1, Heads, Sequence);
            return [.. allQ.AsReadOnlySpan<float>(), .. allK.AsReadOnlySpan<float>()];
        });

        AssertSameBits(expected, parallel, "every core");
        AssertSameBits(expected, capped, "cap 1");
        AssertSameBits(expected, inline, "inline");
    }

    [Fact]
    public void FluxRope_ForwardSingle_AllHeadsAtOnce_MatchesOneHeadAtATime_UnderEverySchedule()
    {
        FluxRope rope = PreparedRope();
        float[] q = Values(Heads * Sequence * HeadDim, seed: 7);
        float[] expected = new float[q.Length];
        int head = Sequence * HeadDim;
        for (int h = 0; h < Heads; h++)
        {
            using Tensor one = FromFloats(q.AsSpan(h * head, head), new TensorShape(1, 1, Sequence, HeadDim));
            rope.ForwardSingle(one, 1, 1, Sequence);
            one.AsReadOnlySpan<float>().CopyTo(expected.AsSpan(h * head));
        }

        (float[] parallel, float[] capped, float[] inline) = UnderEverySchedule(() =>
        {
            using Tensor all = FromFloats(q, new TensorShape(1, Heads, Sequence, HeadDim));
            rope.ForwardSingle(all, 1, Heads, Sequence);
            return all.AsReadOnlySpan<float>().ToArray();
        });

        AssertSameBits(expected, parallel, "every core");
        AssertSameBits(expected, capped, "cap 1");
        AssertSameBits(expected, inline, "inline");
    }

    [Fact]
    public void Nvfp4LinearDequantBf16_MatchesTheResidentCodec_UnderEverySchedule()
    {
        const int n = 256, k = 512;
        int paddedCols = k / Nvfp4ResidentCodec.GroupSize;
        using Tensor packed = RandomBytes(new TensorShape(n, k / 2), DType.U8, seed: 11);
        using Tensor blockScale = new(new TensorShape(n, paddedCols), DType.F8E4M3);
        Span<byte> scaleBytes = blockScale.AsSpan<byte>();
        for (int i = 0; i < scaleBytes.Length; i++) scaleBytes[i] = (byte)(i & 0xFF);
        using Tensor globalScale = FromFloats([0.37f], new TensorShape(1));
        using Tensor relabelled = packed.ReinterpretAs(DType.F4E2M1, new TensorShape(n, k));
        ushort[] expected;
        using (Tensor resident = Nvfp4ResidentCodec.DequantToBf16(relabelled, blockScale, globalScale))
        {
            expected = resident.AsReadOnlySpan<ushort>().ToArray();
        }

        // Nvfp4HostReference reaches Nvfp4Linear's private BF16 dequant, the loop under test here.
        (ushort[] parallel, ushort[] capped, ushort[] inline) =
            UnderEverySchedule(() => Nvfp4HostReference.Bf16Words(packed, blockScale, globalScale));

        Assert.Equal(expected, parallel);
        Assert.Equal(expected, capped);
        Assert.Equal(expected, inline);
    }

    /// <summary>More experts than lanes, so lanes pull several each from the shared counter, and big enough that
    /// the lanes fan out; under a cap of 1 or inline the lanes run one after another. The output may not depend on
    /// which lane ran which expert, and it agrees with the dense experts to within rounding.</summary>
    [Fact]
    public void GptOssPackedExperts_GiveTheSameBits_UnderEverySchedule_AndAgreeWithTheDenseExperts()
    {
        const int hidden = 128, intermediate = 64, experts = 16, topK = 2, tokens = 24;
        Random rng = new(19);
        Dictionary<string, Tensor> packedWeights = new()
        {
            ["mlp.router.weight"] = FromFloats(Values(experts * hidden, seed: 23, scale: 0.5f), new TensorShape(experts, hidden)),
            ["mlp.router.bias"] = FromFloats(Values(experts, seed: 29, scale: 0.1f), new TensorShape(experts)),
            ["mlp.experts.gate_up_proj_bias"] = FromFloats(Values(experts * 2 * intermediate, seed: 31, scale: 0.1f), new TensorShape(experts, 2 * intermediate)),
            ["mlp.experts.down_proj_bias"] = FromFloats(Values(experts * hidden, seed: 37, scale: 0.1f), new TensorShape(experts, hidden)),
            ["mlp.experts.gate_up_proj.weight"] = RandomBytes(new TensorShape(experts, 2 * intermediate, hidden / 2), DType.U8, seed: 41),
            ["mlp.experts.gate_up_proj.weight_scale"] = PositiveE4M3(rng, experts, 2 * intermediate, hidden / 16),
            ["mlp.experts.gate_up_proj.weight_scale_2"] = FromFloats(Values(experts, seed: 43, scale: 0.25f, offset: 0.75f), new TensorShape(experts)),
            ["mlp.experts.down_proj.weight"] = RandomBytes(new TensorShape(experts, hidden, intermediate / 2), DType.U8, seed: 47),
            ["mlp.experts.down_proj.weight_scale"] = PositiveE4M3(rng, experts, hidden, intermediate / 16),
            ["mlp.experts.down_proj.weight_scale_2"] = FromFloats(Values(experts, seed: 53, scale: 0.25f, offset: 1.25f), new TensorShape(experts)),
        };
        try
        {
            GptOssMoeFfn packed = new(hidden, intermediate, experts, topK, clampLimit: 7.0f, alpha: 1.702f);
            packed.LoadWeights(packedWeights, "mlp");
            using Tensor input = FromFloats(Values(tokens * hidden, seed: 59), new TensorShape(1, tokens, hidden));
            using CpuBackend backend = new();

            (float[] parallel, float[] capped, float[] inline) =
                UnderEverySchedule(() => Floats(packed.Forward(backend, input)));

            AssertSameBits(parallel, capped, "cap 1");
            AssertSameBits(parallel, inline, "inline");

            using Tensor gateUpDense = Nvfp4Codec.DequantExpert(packedWeights["mlp.experts.gate_up_proj.weight"],
                packedWeights["mlp.experts.gate_up_proj.weight_scale"], packedWeights["mlp.experts.gate_up_proj.weight_scale_2"]);
            using Tensor downDense = Nvfp4Codec.DequantExpert(packedWeights["mlp.experts.down_proj.weight"],
                packedWeights["mlp.experts.down_proj.weight_scale"], packedWeights["mlp.experts.down_proj.weight_scale_2"]);
            Dictionary<string, Tensor> denseWeights = new()
            {
                ["mlp.router.weight"] = packedWeights["mlp.router.weight"],
                ["mlp.router.bias"] = packedWeights["mlp.router.bias"],
                ["mlp.experts.gate_up_proj_bias"] = packedWeights["mlp.experts.gate_up_proj_bias"],
                ["mlp.experts.down_proj_bias"] = packedWeights["mlp.experts.down_proj_bias"],
                ["mlp.experts.gate_up_proj"] = gateUpDense,
                ["mlp.experts.down_proj"] = downDense,
            };
            GptOssMoeFfn dense = new(hidden, intermediate, experts, topK, clampLimit: 7.0f, alpha: 1.702f);
            dense.LoadWeights(denseWeights, "mlp");
            float[] reference = Floats(dense.Forward(backend, input));
            double difference = 0, magnitude = 0;
            for (int i = 0; i < reference.Length; i++)
            {
                double d = reference[i] - parallel[i];
                difference += d * d;
                magnitude += (double)reference[i] * reference[i];
            }
            double relL2 = Math.Sqrt(difference / magnitude);
            _output.WriteLine($"packed vs dense relL2 = {relL2:E2}");
            // Same dequant and per-(token, expert) math, but the dense path sums each dot product token by token
            // and the packed one through the vectorized GEMM, so the two round differently; an expert skipped or run
            // twice would be off by tenths.
            Assert.True(relL2 < 1e-4, $"packed experts diverge from dense: relL2 {relL2:E2}");
        }
        finally
        {
            foreach (Tensor tensor in packedWeights.Values) tensor.Dispose();
        }
    }

    private static FluxRope PreparedRope()
    {
        FluxRope rope = new([16, 56, 56], theta: 10000);
        using Tensor positions = FluxRope.BuildPositionIds(TextTokens, Side, Side);
        rope.Precompute(positions);
        return rope;
    }

    /// <summary>Uniform values in <c>[offset - scale, offset + scale)</c>.</summary>
    private static float[] Values(int count, int seed, float scale = 1f, float offset = 0f)
    {
        Random rng = new(seed);
        float[] values = new float[count];
        for (int i = 0; i < count; i++) values[i] = offset + (float)((rng.NextDouble() * 2.0 - 1.0) * scale);
        return values;
    }

    /// <summary>Positive, finite E4M3 block scales, exponents 5 to 7: magnitudes from 2^-2 to just under 2.</summary>
    private static Tensor PositiveE4M3(Random rng, int experts, int rows, int blockColumns)
    {
        int paddedRows = (rows + 127) / 128 * 128, paddedColumns = (blockColumns + 3) / 4 * 4;
        Tensor scale = new(new TensorShape(experts, paddedRows, paddedColumns), DType.F8E4M3);
        Span<byte> bytes = scale.AsSpan<byte>();
        for (int i = 0; i < bytes.Length; i++) bytes[i] = (byte)((rng.Next(5, 8) << 3) | rng.Next(8));
        return scale;
    }

    private static Tensor RandomBytes(TensorShape shape, DType dtype, int seed)
    {
        Tensor tensor = new(shape, dtype);
        new Random(seed).NextBytes(tensor.AsSpan<byte>());
        return tensor;
    }

    private static Tensor FromFloats(ReadOnlySpan<float> values, TensorShape shape)
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
