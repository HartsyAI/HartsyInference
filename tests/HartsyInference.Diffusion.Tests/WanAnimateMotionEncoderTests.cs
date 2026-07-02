using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Tiny-config tests for the Wan-Animate motion encoder (<see cref="WanAnimateMotionEncoder"/>, ComfyUI
/// <c>Generator</c>): the StyleGAN2 appearance stack (<c>enc.net_app.convs</c>: scaled convs + FIR blur + fused
/// leaky-ReLU) → <c>enc.fc</c> → <c>dec.direction</c> QR Linear Motion Decomposition produces a finite
/// <c>[B, styleDim]</c> motion vector. Numerics validation-pending vs the real checkpoint.</summary>
public unsafe class WanAnimateMotionEncoderTests
{
    [Fact]
    public void Forward_TinyConfig_ProducesMotionVector()
    {
        CpuBackend backend = new();
        const int size = 16, ch = 8, style = 8, motion = 4;
        WanAnimateMotionEncoder enc = new();
        enc.LoadWeights(BuildWeights("motion_encoder", size, ch, style, motion, fcLayers: 2), "motion_encoder");
        Assert.Equal(size, enc.ExpectedInputSize());

        int frames = 2;
        Tensor face = Rand4d(frames, 3, size, size, seed: 31);
        Tensor motionVec = enc.Forward(backend, face);

        Assert.Equal(2, motionVec.Shape.Rank);
        Assert.Equal(frames, (int)motionVec.Shape[0]);
        Assert.Equal(style, (int)motionVec.Shape[1]);
        float* p = (float*)motionVec.DataPointer;
        for (long i = 0; i < motionVec.Shape.ElementCount; i++) Assert.True(float.IsFinite(p[i]), $"non-finite at {i}");
    }

    [Fact]
    public void Forward_WrongInputSize_Throws()
    {
        CpuBackend backend = new();
        WanAnimateMotionEncoder enc = new();
        enc.LoadWeights(BuildWeights("motion_encoder", size: 16, ch: 8, style: 8, motion: 4, fcLayers: 2), "motion_encoder");
        Tensor face = Rand4d(1, 3, 32, 32, seed: 32);
        Assert.Throws<ArgumentException>(() => enc.Forward(backend, face));
    }

    /// <summary>ComfyUI Generator naming: <c>enc.net_app.convs.{i}</c> (ConvLayer / ResBlock / final EqualConv2d
    /// Sequential indices), <c>enc.fc.{i}</c>, <c>dec.direction.weight</c>.</summary>
    private static Dictionary<string, Tensor> BuildWeights(string p, int size, int ch, int style, int motion, int fcLayers)
    {
        Dictionary<string, Tensor> w = new()
        {
            [$"{p}.enc.net_app.convs.0.0.weight"] = R([ch, 3, 1, 1]),
            [$"{p}.enc.net_app.convs.0.1.bias"] = R([1, ch, 1, 1]),
            [$"{p}.dec.direction.weight"] = R([style, motion]),
        };
        int numResBlocks = (int)Math.Round(Math.Log2(size)) - 2;   // size 16 → 2 down-blocks (16→8→4)
        for (int i = 1; i <= numResBlocks; i++)
        {
            string b = $"{p}.enc.net_app.convs.{i}";
            w[$"{b}.conv1.0.weight"] = R([ch, ch, 3, 3]); w[$"{b}.conv1.1.bias"] = R([1, ch, 1, 1]);
            w[$"{b}.conv2.0.kernel"] = R([4, 4]);
            w[$"{b}.conv2.1.weight"] = R([ch, ch, 3, 3]); w[$"{b}.conv2.2.bias"] = R([1, ch, 1, 1]);
            w[$"{b}.skip.0.kernel"] = R([4, 4]);
            w[$"{b}.skip.1.weight"] = R([ch, ch, 1, 1]);
        }
        w[$"{p}.enc.net_app.convs.{numResBlocks + 1}.weight"] = R([style, ch, 4, 4]);   // bare EqualConv2d, no bias
        for (int i = 0; i < fcLayers - 1; i++)
        {
            w[$"{p}.enc.fc.{i}.weight"] = R([style, style]); w[$"{p}.enc.fc.{i}.bias"] = R([style]);
        }
        w[$"{p}.enc.fc.{fcLayers - 1}.weight"] = R([motion, style]); w[$"{p}.enc.fc.{fcLayers - 1}.bias"] = R([motion]);
        return w;
    }

    private static int _seed = 700;
    private static Tensor R(int[] dims)
    {
        long[] d = Array.ConvertAll(dims, x => (long)x);
        Tensor t = new Tensor(new TensorShape(d), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(_seed++);
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 0.1 - 0.05);
        return t;
    }

    private static Tensor Rand4d(int b, int c, int h, int w, int seed)
    {
        Tensor x = new Tensor(new TensorShape(b, c, h, w), DType.F32);
        float* p = (float*)x.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < x.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return x;
    }
}
