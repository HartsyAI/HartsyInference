using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Tiny-config smoke test for the Wan-Animate motion encoder (<see cref="WanAnimateMotionEncoder"/>): the
/// StyleGAN appearance stack (scaled convs + FIR blur + fused leaky-ReLU) → motion MLP → QR-based Linear Motion
/// Decomposition produces a finite <c>[B, out_dim]</c> motion vector. Numerics validation-pending.</summary>
public unsafe class WanAnimateMotionEncoderTests
{
    [Fact]
    public void Forward_TinyConfig_ProducesMotionVector()
    {
        CpuBackend backend = new();
        Dictionary<int, int> channels = new() { [4] = 8, [8] = 8, [16] = 8 };
        WanAnimateMotionEncoder enc = new(size: 16, styleDim: 8, motionDim: 4, outDim: 8, motionBlocks: 2, channels: channels);
        enc.LoadWeights(BuildWeights("motion_encoder", size: 16, ch: 8, style: 8, motion: 4, outDim: 8, motionBlocks: 2), "motion_encoder");

        int frames = 2;   // B*T face frames
        Tensor face = Rand4d(frames, 3, 16, 16, seed: 31);
        Tensor motion = enc.Forward(backend, face);

        Assert.Equal(2, motion.Shape.Rank);
        Assert.Equal(frames, (int)motion.Shape[0]);
        Assert.Equal(8, (int)motion.Shape[1]);   // out_dim
        float* p = (float*)motion.DataPointer;
        for (long i = 0; i < motion.Shape.ElementCount; i++) Assert.True(float.IsFinite(p[i]), $"non-finite at {i}");
    }

    private static Dictionary<string, Tensor> BuildWeights(string p, int size, int ch, int style, int motion, int outDim, int motionBlocks)
    {
        Dictionary<string, Tensor> w = new()
        {
            [$"{p}.conv_in.weight"] = R([ch, 3, 1, 1]), [$"{p}.conv_in.act_fn.bias"] = R([ch]),
            [$"{p}.conv_out.weight"] = R([style, ch, 4, 4]),   // no bias (use_activation=false, bias=false)
            [$"{p}.motion_synthesis_weight"] = R([outDim, motion]),
        };
        // res blocks: size 16 → log2=4 → i in {4,3} → 2 blocks.
        int logSize = (int)Math.Round(Math.Log2(size));
        int idx = 0;
        for (int i = logSize; i > 2; i--)
        {
            string b = $"{p}.res_blocks.{idx}";
            w[$"{b}.conv1.weight"] = R([ch, ch, 3, 3]); w[$"{b}.conv1.act_fn.bias"] = R([ch]);
            w[$"{b}.conv2.weight"] = R([ch, ch, 3, 3]); w[$"{b}.conv2.act_fn.bias"] = R([ch]);
            w[$"{b}.conv_skip.weight"] = R([ch, ch, 1, 1]);   // no bias
            idx++;
        }
        // motion network: (motionBlocks-1) × (style→style) + 1 × (style→motion), each with bias (no activation).
        for (int i = 0; i < motionBlocks - 1; i++) { w[$"{p}.motion_network.{i}.weight"] = R([style, style]); w[$"{p}.motion_network.{i}.bias"] = R([style]); }
        w[$"{p}.motion_network.{motionBlocks - 1}.weight"] = R([motion, style]); w[$"{p}.motion_network.{motionBlocks - 1}.bias"] = R([motion]);
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
