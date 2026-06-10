using Xunit;
using SharpInference.Core.Tensors;
using SharpInference.Cpu;
using SharpInference.Diffusion.Models.Vae;

namespace SharpInference.Diffusion.Tests;

/// <summary>Structural tests for the LTX-Video VAE blocks (<see cref="LtxVaeResnetBlock3d"/>, <see cref="LtxVaeUpsampler3d"/>) on CPU. Shape + finiteness; numerics vs the real checkpoint are validation-pending.</summary>
public unsafe class LtxVaeBlockTests
{
    [Fact]
    public void Resnet_TimestepConditioned_PreservesShape()
    {
        CpuBackend backend = new();
        int c = 8;
        Dictionary<string, Tensor> w = new()
        {
            ["rb.conv1.conv.weight"] = R([c, c, 3, 3, 3]), ["rb.conv1.conv.bias"] = R([c]),
            ["rb.conv2.conv.weight"] = R([c, c, 3, 3, 3]), ["rb.conv2.conv.bias"] = R([c]),
            ["rb.scale_shift_table"] = R([4, c]),
        };
        LtxVaeResnetBlock3d block = new(c, c, timestepCond: true);
        block.LoadWeights(w, "rb");
        Tensor x = R([1, c, 2, 3, 3]);
        Tensor temb = R([4 * c]);
        Tensor outT = block.Forward(backend, x, temb);
        AssertShape(outT, [1, c, 2, 3, 3]);
        AssertFinite(outT);
    }

    [Fact]
    public void Resnet_ChannelChange_UsesShortcut()
    {
        CpuBackend backend = new();
        int inC = 4, outC = 8;
        Dictionary<string, Tensor> w = new()
        {
            ["rb.conv1.conv.weight"] = R([outC, inC, 3, 3, 3]), ["rb.conv1.conv.bias"] = R([outC]),
            ["rb.conv2.conv.weight"] = R([outC, outC, 3, 3, 3]), ["rb.conv2.conv.bias"] = R([outC]),
            ["rb.norm3.weight"] = R([inC]), ["rb.norm3.bias"] = R([inC]),
            ["rb.conv_shortcut.conv.weight"] = R([outC, inC, 1, 1, 1]), ["rb.conv_shortcut.conv.bias"] = R([outC]),
        };
        LtxVaeResnetBlock3d block = new(inC, outC, timestepCond: false);
        block.LoadWeights(w, "rb");
        Tensor x = R([1, inC, 2, 3, 3]);
        Tensor outT = block.Forward(backend, x, null);
        AssertShape(outT, [1, outC, 2, 3, 3]);
        AssertFinite(outT);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Upsampler_SpatialDoublesResolution(bool residual)
    {
        CpuBackend backend = new();
        int inC = 8, stProd = 1 * 2 * 2, upscale = 1;
        int convOut = inC * stProd / upscale;   // 32
        Dictionary<string, Tensor> w = new()
        {
            ["up.conv.conv.weight"] = R([convOut, inC, 3, 3, 3]), ["up.conv.conv.bias"] = R([convOut]),
        };
        LtxVaeUpsampler3d up = new(inC, (1, 2, 2), upscaleFactor: upscale, residual: residual);
        up.LoadWeights(w, "up");
        Tensor x = R([1, inC, 2, 3, 3]);
        Tensor outT = up.Forward(backend, x);
        AssertShape(outT, [1, inC / upscale, 2, 6, 6]);   // F unchanged (st0=1), H/W ×2
        AssertFinite(outT);
    }

    [Fact]
    public void Upsampler_TemporalExpands()
    {
        CpuBackend backend = new();
        int inC = 16, st0 = 2, st1 = 2, st2 = 2, upscale = 2;
        int stProd = st0 * st1 * st2;            // 8
        int convOut = inC * stProd / upscale;    // 64
        Dictionary<string, Tensor> w = new()
        {
            ["up.conv.conv.weight"] = R([convOut, inC, 3, 3, 3]), ["up.conv.conv.bias"] = R([convOut]),
        };
        LtxVaeUpsampler3d up = new(inC, (st0, st1, st2), upscaleFactor: upscale, residual: false);
        up.LoadWeights(w, "up");
        Tensor x = R([1, inC, 2, 2, 2]);
        Tensor outT = up.Forward(backend, x);
        // out channels = inC/upscale = 8; F = 2*2-(2-1) = 3; H/W = 4.
        AssertShape(outT, [1, inC / upscale, 3, 4, 4]);
        AssertFinite(outT);
    }

    private static int s_seed = 50;
    private static Tensor R(int[] dims)
    {
        long[] d = Array.ConvertAll(dims, x => (long)x);
        Tensor t = new Tensor(new TensorShape(d), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(s_seed++);
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 0.2 - 0.1);
        return t;
    }

    private static void AssertShape(Tensor t, int[] expected)
    {
        Assert.Equal(expected.Length, t.Shape.Rank);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], (int)t.Shape[i]);
    }

    private static void AssertFinite(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.Shape.ElementCount; i++) Assert.True(float.IsFinite(p[i]), $"non-finite at {i}");
    }
}
