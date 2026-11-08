using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Matting.BiRefNet;
using HartsyInference.Vision.Swin;

namespace HartsyInference.Vision.Tests.Matting;

public sealed unsafe class BiRefNetLayerTests
{
    private const int Channels = 2, OutChannels = 3, Size = 5, Kernel = 3, Pad = 1;

    [Fact]
    public void DeformConv_ZeroOffsetUnitMaskMatchesPlainConvolution()
    {
        using IBackend backend = new CpuBackend();
        Dictionary<string, Tensor> w = DeformWeights(offsetBias: 0f, regularSeed: 1);
        BiRefNetDeformConv layer = new(Kernel, Pad);
        layer.LoadWeights(w, "layer", []);
        Tensor x = Ramp(new TensorShape(1, Channels, Size, Size));

        using Tensor actual = layer.Forward(backend, x);
        using Tensor expected = PlainConv(backend, x, w["layer.atrous_conv.regular_conv.weight"], Pad);

        AssertClose(expected, actual, 1e-4f);
    }

    [Fact]
    public void DeformConv_UnitXOffsetSamplesOneColumnRight()
    {
        using IBackend backend = new CpuBackend();
        Dictionary<string, Tensor> w = DeformWeights(offsetBias: 0f, regularSeed: 2);
        // Channel 2k+1 is the x-offset of tap k: a bias of 1 shifts every tap one column to the right.
        Tensor offsetBias = w["layer.atrous_conv.offset_conv.bias"];
        for (int k = 0; k < Kernel * Kernel; k++) offsetBias.AsSpan<float>()[2 * k + 1] = 1f;
        BiRefNetDeformConv layer = new(Kernel, Pad);
        layer.LoadWeights(w, "layer", []);
        Tensor x = Ramp(new TensorShape(1, Channels, Size, Size));

        using Tensor actual = layer.Forward(backend, x);
        using Tensor expected = NaiveColumnOffsetConv(x, w["layer.atrous_conv.regular_conv.weight"]);

        AssertClose(expected, actual, 1e-4f);
    }

    [Fact]
    public void Patchify_TilesImageIntoChannelsInUpstreamOrder()
    {
        Tensor image = Ramp(new TensorShape(1, 3, 4, 4));

        using Tensor patches = BiRefNetOps.Patchify(image, 2);

        Assert.Equal(new long[] { 1, 12, 2, 2 }, Dims(patches));
        float* p = (float*)patches.DataPointer;
        float* src = (float*)image.DataPointer;
        for (int c = 0; c < 3; c++)
            for (int gi = 0; gi < 2; gi++)
                for (int gj = 0; gj < 2; gj++)
                    for (int y = 0; y < 2; y++)
                        for (int xx = 0; xx < 2; xx++)
                        {
                            float got = p[((c * 4 + gi * 2 + gj) * 2 + y) * 2 + xx];
                            float want = src[(c * 4 + gi * 2 + y) * 4 + gj * 2 + xx];
                            Assert.Equal(want, got);
                        }
    }

    [Fact]
    public void MulBroadcastGate_ScalesEveryChannelByTheSameMap()
    {
        Tensor x = Ramp(new TensorShape(1, 2, 2, 2));
        Tensor gate = new(new TensorShape(1, 1, 2, 2), DType.F32);
        float[] g = [0f, 0.5f, 2f, -1f];
        g.AsSpan().CopyTo(gate.AsSpan<float>());

        using Tensor product = BiRefNetOps.MulBroadcastGate(x, gate);

        float* p = (float*)product.DataPointer, xp = (float*)x.DataPointer;
        for (int c = 0; c < 2; c++)
            for (int i = 0; i < 4; i++)
                Assert.Equal(xp[c * 4 + i] * g[i], p[c * 4 + i]);
    }

    [Fact]
    public void RelativePositionIndex_IndexesTheTwoWindowTableByOffset()
    {
        int[] index = SwinOps.RelativePositionIndex(2);

        // Positions 0:(0,0) 1:(0,1) 2:(1,0) 3:(1,1); entry = (dy + 1) * 3 + (dx + 1).
        Assert.Equal(16, index.Length);
        Assert.Equal(4, index[0 * 4 + 0]);
        Assert.Equal(3, index[0 * 4 + 1]);
        Assert.Equal(0, index[0 * 4 + 3]);
        Assert.Equal(5, index[1 * 4 + 0]);
        Assert.Equal(8, index[3 * 4 + 0]);
    }

    private static Dictionary<string, Tensor> DeformWeights(float offsetBias, int regularSeed)
    {
        const int taps = Kernel * Kernel;
        Dictionary<string, Tensor> w = new()
        {
            ["layer.atrous_conv.offset_conv.weight"] = Zeros(new TensorShape(2 * taps, Channels, Kernel, Kernel)),
            ["layer.atrous_conv.offset_conv.bias"] = Filled(new TensorShape(2 * taps), offsetBias),
            ["layer.atrous_conv.modulator_conv.weight"] = Zeros(new TensorShape(taps, Channels, Kernel, Kernel)),
            ["layer.atrous_conv.modulator_conv.bias"] = Zeros(new TensorShape(taps)),
            ["layer.atrous_conv.regular_conv.weight"] = Pseudo(new TensorShape(OutChannels, Channels, Kernel, Kernel), regularSeed),
            ["layer.bn.weight"] = Filled(new TensorShape(OutChannels), 1f),
            ["layer.bn.bias"] = Zeros(new TensorShape(OutChannels)),
            ["layer.bn.running_mean"] = Zeros(new TensorShape(OutChannels)),
            ["layer.bn.running_var"] = Filled(new TensorShape(OutChannels), 1f),
        };
        return w;
    }

    private static Tensor PlainConv(IBackend backend, Tensor x, Tensor weight, int pad)
    {
        Tensor output = new(new TensorShape(1, OutChannels, Size, Size), DType.F32);
        backend.Conv2D(output, x, weight, null, 1, 1, pad, pad);
        return output;
    }

    /// <summary>Direct 3×3 cross-correlation where every tap samples one column to the right of its grid position
    /// (offset dx = +1, dy = 0). Out-of-image samples are zero, so the left border reads column 0, unlike a plain conv
    /// over a pre-shifted input, whose zero padding would sit at column −1.</summary>
    private static Tensor NaiveColumnOffsetConv(Tensor x, Tensor weight)
    {
        Tensor output = new(new TensorShape(1, OutChannels, Size, Size), DType.F32);
        ReadOnlySpan<float> xs = x.AsReadOnlySpan<float>(), ws = weight.AsReadOnlySpan<float>();
        Span<float> os = output.AsSpan<float>();
        for (int oc = 0; oc < OutChannels; oc++)
            for (int y = 0; y < Size; y++)
                for (int xx = 0; xx < Size; xx++)
                {
                    float acc = 0f;
                    for (int c = 0; c < Channels; c++)
                        for (int ki = 0; ki < Kernel; ki++)
                            for (int kj = 0; kj < Kernel; kj++)
                            {
                                int sy = y - Pad + ki, sx = xx - Pad + kj + 1;
                                if (sy < 0 || sy >= Size || sx < 0 || sx >= Size) continue;
                                acc += ws[((oc * Channels + c) * Kernel + ki) * Kernel + kj] * xs[(c * Size + sy) * Size + sx];
                            }
                    os[(oc * Size + y) * Size + xx] = acc;
                }
        return output;
    }

    private static void AssertClose(Tensor expected, Tensor actual, float tolerance)
    {
        Assert.Equal(expected.ElementCount, actual.ElementCount);
        ReadOnlySpan<float> e = expected.AsReadOnlySpan<float>(), a = actual.AsReadOnlySpan<float>();
        for (int i = 0; i < e.Length; i++)
            Assert.True(MathF.Abs(e[i] - a[i]) <= tolerance, $"element {i}: expected {e[i]}, got {a[i]}");
    }

    private static Tensor Ramp(TensorShape shape)
    {
        Tensor t = new(shape, DType.F32);
        Span<float> s = t.AsSpan<float>();
        for (int i = 0; i < s.Length; i++) s[i] = (i % 7) * 0.25f - 0.5f;
        return t;
    }

    private static Tensor Pseudo(TensorShape shape, int seed)
    {
        Tensor t = new(shape, DType.F32);
        Span<float> s = t.AsSpan<float>();
        Random rng = new(seed);
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() * 0.4 - 0.2);
        return t;
    }

    private static Tensor Zeros(TensorShape shape) => Filled(shape, 0f);

    private static Tensor Filled(TensorShape shape, float value)
    {
        Tensor t = new(shape, DType.F32);
        t.AsSpan<float>().Fill(value);
        return t;
    }

    private static long[] Dims(Tensor t)
    {
        long[] dims = new long[t.Shape.Rank];
        for (int i = 0; i < dims.Length; i++) dims[i] = t.Shape[i];
        return dims;
    }
}
