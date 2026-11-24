using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.CrossFrame;
using HartsyInference.Vision.DepthPro2;
using HartsyInference.Vision.Dinov2;
using HartsyInference.Vision.Dinov3;

namespace HartsyInference.Vision.Tests.CrossFrame;

/// <summary>
/// Cross-frame attention on a tiny random DINOv3 (64 wide, one head, three blocks, the middle and last attending across frames): the frame order does not
/// matter, the frames do see each other, one frame alone is the per-frame path bit for bit, and the unbuffered ring's first frame (and every frame of a
/// one-frame ring) is that frame alone.
/// </summary>
public sealed class CrossFrameAttentionTests : IDisposable
{
    private const int Frames = 3, Height = 32, Width = 48;
    private static readonly int[] Taps = [0, 1, 2];
    private static readonly int[] CrossFrameBlocks = [1, 2];

    private readonly CpuBackend _backend = new();
    private readonly List<Tensor> _owned = [];
    private readonly Dinov3VisionEncoder _encoder;

    public CrossFrameAttentionTests()
    {
        Dinov2Preset preset = new()
        {
            Name = "tiny-dinov3", HiddenSize = 64, NumLayers = 3, NumHeads = 1, IntermediateSize = 128, ImageSize = 48, PatchSize = 16, NumRegisterTokens = 4,
            UsesRotaryPositionEmbedding = true, RoundRopePeriodsToBf16 = false, AttentionQkvBias = false, HasFinalNorm = false, RopeNormalizesByLongSide = true,
        };
        _encoder = new Dinov3VisionEncoder(preset);
        _encoder.LoadWeights(Weights(new Random(7)));
    }

    [Fact]
    public void ReorderingTheFramesReordersTheOutputOnly()
    {
        using Tensor frames = Pixels(new Random(11), Frames);
        using Tensor shuffled = Reorder(frames, [2, 0, 1]);

        Tensor[] a = _encoder.EncodeFrameIntermediates(_backend, frames, Taps, CrossFrameBlocks);
        Tensor[] b = _encoder.EncodeFrameIntermediates(_backend, shuffled, Taps, CrossFrameBlocks);
        try
        {
            for (int tap = 0; tap < Taps.Length; tap++)
            {
                using Tensor back = Reorder(b[tap], [1, 2, 0]);
                Assert.True(MaxAbs(a[tap], back) < 1e-5, $"tap {tap} differs after reordering the frames by {MaxAbs(a[tap], back):E2}");
            }
        }
        finally
        {
            DisposeAll(a);
            DisposeAll(b);
        }
    }

    [Fact]
    public void TheFramesSeeEachOtherOnlyInTheCrossFrameBlocks()
    {
        using Tensor frames = Pixels(new Random(12), Frames);
        Tensor[] joint = _encoder.EncodeFrameIntermediates(_backend, frames, Taps, CrossFrameBlocks);
        try
        {
            for (int t = 0; t < Frames; t++)
            {
                using Tensor one = Reorder(frames, [t]);
                Tensor[] alone = _encoder.EncodeIntermediates(_backend, one, Taps, CrossFrameBlocks);
                try
                {
                    using Tensor block0 = Reorder(joint[0], [t]);
                    using Tensor block2 = Reorder(joint[2], [t]);
                    Assert.True(MaxAbs(block0, alone[0]) < 1e-6);   // before any cross-frame block
                    Assert.True(MaxAbs(block2, alone[2]) > 1e-3, "the last block's output should depend on the other frames");
                    float[] prefixJoint = block2.AsReadOnlySpan<float>()[..(5 * 64)].ToArray();
                    float[] prefixInput;
                    using (Tensor input = Reorder(joint[1], [t])) prefixInput = input.AsReadOnlySpan<float>()[..(5 * 64)].ToArray();
                    Assert.Equal(prefixInput, prefixJoint);   // a cross-frame block keeps each frame's prefix
                }
                finally
                {
                    DisposeAll(alone);
                }
            }
        }
        finally
        {
            DisposeAll(joint);
        }
    }

    [Fact]
    public void OneFrameAloneIsThePerFramePathBitForBit()
    {
        using Tensor frame = Pixels(new Random(13), 1);
        Tensor[] joint = _encoder.EncodeFrameIntermediates(_backend, frame, Taps, CrossFrameBlocks);
        Tensor[] alone = _encoder.EncodeIntermediates(_backend, frame, Taps, CrossFrameBlocks);
        try
        {
            for (int tap = 0; tap < Taps.Length; tap++) Assert.Equal(0, MaxAbs(joint[tap], alone[tap]));
        }
        finally
        {
            DisposeAll(joint);
            DisposeAll(alone);
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    public void TheRingsFirstFrameAndEveryFrameOfAOneFrameRingIsThatFrameAlone(int window)
    {
        using Tensor frames = Pixels(new Random(14), Frames);
        using CrossFrameKvRing ring = new(window);
        for (int t = 0; t < Frames; t++)
        {
            using Tensor one = Reorder(frames, [t]);
            Tensor[] cached = _encoder.EncodeFrameIntermediates(_backend, one, Taps, CrossFrameBlocks, ring);
            Tensor[] alone = _encoder.EncodeIntermediates(_backend, one, Taps, CrossFrameBlocks);
            try
            {
                double diff = MaxAbs(cached[2], alone[2]);
                if (t == 0 || window == 1) Assert.True(diff < 1e-5, $"frame {t}, window {window}: cached differs from alone by {diff:E2}");
                else Assert.True(diff > 1e-3, $"frame {t}, window {window}: the cached frames should change the output");
            }
            finally
            {
                DisposeAll(cached);
                DisposeAll(alone);
            }
        }

        Assert.Equal(Frames, ring.Seen(CrossFrameBlocks[0]));
        Assert.Equal(5 + Math.Min(Frames, window) * 6, ring.LiveRows(CrossFrameBlocks[1]));
    }

    [Fact]
    public void TheBlendRampMatchesTorchLinspaceWithoutItsEndpoints()
    {
        Assert.Equal([0.8f, 0.6f, 0.4f, 0.2f], DepthPro2VideoStream.BlendRamp(4));
        Assert.Equal([0.5f], DepthPro2VideoStream.BlendRamp(1));
        Assert.Empty(DepthPro2VideoStream.BlendRamp(0));
    }

    public void Dispose()
    {
        foreach (Tensor t in _owned) t.Dispose();
        _backend.Dispose();
    }

    private Dictionary<string, Tensor> Weights(Random rng)
    {
        Dictionary<string, Tensor> w = new()
        {
            ["embeddings.patch_embeddings.weight"] = Filled(rng, 0.05, 64, 3, 16, 16),
            ["embeddings.patch_embeddings.bias"] = Filled(rng, 0.05, 64),
            ["embeddings.cls_token"] = Filled(rng, 0.5, 1, 1, 64),
            ["embeddings.register_tokens"] = Filled(rng, 0.5, 1, 4, 64),
        };
        for (int i = 0; i < 3; i++)
        {
            string p = $"layer.{i}";
            w[$"{p}.norm1.weight"] = Around(rng, 0.2, 1);
            w[$"{p}.norm1.bias"] = Filled(rng, 0.1, 64);
            foreach (string proj in new[] { "q_proj", "k_proj", "v_proj", "o_proj" }) w[$"{p}.attention.{proj}.weight"] = Filled(rng, 0.25, 64, 64);
            w[$"{p}.attention.o_proj.bias"] = Filled(rng, 0.1, 64);
            w[$"{p}.layer_scale1.lambda1"] = Around(rng, 0.2, 0.5);
            w[$"{p}.norm2.weight"] = Around(rng, 0.2, 1);
            w[$"{p}.norm2.bias"] = Filled(rng, 0.1, 64);
            w[$"{p}.mlp.up_proj.weight"] = Filled(rng, 0.2, 128, 64);
            w[$"{p}.mlp.up_proj.bias"] = Filled(rng, 0.1, 128);
            w[$"{p}.mlp.down_proj.weight"] = Filled(rng, 0.15, 64, 128);
            w[$"{p}.mlp.down_proj.bias"] = Filled(rng, 0.1, 64);
            w[$"{p}.layer_scale2.lambda1"] = Around(rng, 0.2, 0.5);
        }

        return w;
    }

    private Tensor Filled(Random rng, double scale, params long[] shape) => Filled(rng, scale, shape, 0);

    private Tensor Around(Random rng, double scale, double offset) => Filled(rng, scale, [64], offset);

    private Tensor Filled(Random rng, double scale, long[] shape, double offset)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        Span<float> s = t.AsSpan<float>();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(offset + scale * (rng.NextDouble() * 2 - 1));
        _owned.Add(t);
        return t;
    }

    private static Tensor Pixels(Random rng, int frames)
    {
        Tensor t = new(new TensorShape(frames, 3, Height, Width), DType.F32);
        Span<float> s = t.AsSpan<float>();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() * 4 - 2);
        return t;
    }

    // The items of the leading axis in the order given, as a new tensor.
    private static Tensor Reorder(Tensor x, int[] order)
    {
        long[] dims = new long[x.Shape.Rank];
        for (int d = 0; d < dims.Length; d++) dims[d] = x.Shape[d];
        int item = (int)(x.ElementCount / dims[0]);
        dims[0] = order.Length;
        Tensor result = new(new TensorShape(dims), DType.F32);
        for (int i = 0; i < order.Length; i++) x.AsReadOnlySpan<float>().Slice(order[i] * item, item).CopyTo(result.AsSpan<float>().Slice(i * item, item));
        return result;
    }

    private static double MaxAbs(Tensor a, Tensor b)
    {
        Assert.Equal(a.Shape, b.Shape);
        ReadOnlySpan<float> x = a.AsReadOnlySpan<float>(), y = b.AsReadOnlySpan<float>();
        double worst = 0;
        for (int i = 0; i < x.Length; i++) worst = Math.Max(worst, Math.Abs(x[i] - y[i]));
        return worst;
    }

    private static void DisposeAll(Tensor[] tensors)
    {
        foreach (Tensor t in tensors) t.Dispose();
    }
}
