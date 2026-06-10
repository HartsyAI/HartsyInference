using Xunit;
using SharpInference.Core.Tensors;
using SharpInference.Cpu;
using SharpInference.Diffusion.Models.Vae;

namespace SharpInference.Diffusion.Tests;

/// <summary>End-to-end structural test for the base LTX-Video VAE decoder (<see cref="LtxVideoVaeDecoder"/>) on CPU: conv_in → mid → up blocks (channel-change + pixel-shuffle upsamplers) → norm_out → conv_out → pixel-unshuffle. Tiny config; numerics vs the real checkpoint are validation-pending.</summary>
public unsafe class LtxVaeDecoderTests
{
    // Tiny config: latent 8ch, block_out [8,16,16,16], scaling [T,T,T,F], 1 layer/block, patch 2.
    private static readonly int[] BlockOut = [8, 16, 16, 16];
    private static readonly bool[] Scaling = [true, true, true, false];
    private static readonly int[] Layers = [1, 1, 1, 1, 1];
    private const int Latent = 8, OutCh = 3, Patch = 2;

    [Fact]
    public void Decode_BaseConfig_ProducesExpandedRgbAndIsFinite()
    {
        CpuBackend backend = new();
        LtxVideoVaeDecoder decoder = new(Latent, OutCh, BlockOut, Scaling, Layers, Patch, isCausal: false);
        decoder.LoadWeights(BuildVae());

        int tLat = 2;
        Tensor latent = Rand([1, Latent, tLat, 2, 2]);
        Tensor rgb = decoder.Decode(backend, latent);

        Assert.Equal(3, (int)rgb.Shape[1]);
        Assert.Equal(tLat * 8 - 7, (int)rgb.Shape[2]);   // 3 temporal upsamplers ×2 → (T_lat-1)*8+1 = 9
        Assert.Equal(decoder.OutputFrames(tLat), (int)rgb.Shape[2]);
        // spatial: 3 upsamplers ×2 (=8) × patch 2 = 16.
        Assert.Equal(2 * 16, (int)rgb.Shape[3]);
        Assert.Equal(2 * 16, (int)rgb.Shape[4]);
        float* p = (float*)rgb.DataPointer;
        for (long i = 0; i < rgb.Shape.ElementCount; i++) Assert.True(float.IsFinite(p[i]), $"non-finite at {i}");
    }

    private static Dictionary<string, Tensor> BuildVae()
    {
        int[] bo = Rev(BlockOut);      // [16,16,16,8]
        bool[] sc = Rev(Scaling);      // [F,T,T,T]
        int[] ly = Rev(Layers);
        Dictionary<string, Tensor> w = new();
        int output = bo[0];

        AddConv(w, "decoder.conv_in", output, Latent, 3);
        for (int j = 0; j < ly[0]; j++) AddResnet(w, $"decoder.mid_block.resnets.{j}", output, output);

        for (int i = 0; i < bo.Length; i++)
        {
            int inC = output, outC = bo[i];
            string p = $"decoder.up_blocks.{i}";
            if (inC != outC) AddResnet(w, $"{p}.conv_in", inC, outC);
            if (sc[i]) AddConv(w, $"{p}.upsamplers.0.conv", outC * 8, outC, 3);
            for (int j = 0; j < ly[i + 1]; j++) AddResnet(w, $"{p}.resnets.{j}", outC, outC);
            output = outC;
        }
        AddConv(w, "decoder.conv_out", OutCh * Patch * Patch, output, 3);
        return w;
    }

    private static void AddResnet(Dictionary<string, Tensor> w, string p, int inC, int outC)
    {
        AddConv(w, $"{p}.conv1", outC, inC, 3);
        AddConv(w, $"{p}.conv2", outC, outC, 3);
        if (inC != outC)
        {
            w[$"{p}.norm3.weight"] = Rand([inC]); w[$"{p}.norm3.bias"] = Rand([inC]);
            AddConv(w, $"{p}.conv_shortcut", outC, inC, 1);
        }
    }

    private static void AddConv(Dictionary<string, Tensor> w, string p, int outC, int inC, int k)
    {
        w[$"{p}.conv.weight"] = Rand([outC, inC, k, k, k]);
        w[$"{p}.conv.bias"] = Rand([outC]);
    }

    private static int[] Rev(int[] a) { int[] r = (int[])a.Clone(); Array.Reverse(r); return r; }
    private static bool[] Rev(bool[] a) { bool[] r = (bool[])a.Clone(); Array.Reverse(r); return r; }

    private static int s_seed = 200;
    private static Tensor Rand(int[] dims)
    {
        long[] d = Array.ConvertAll(dims, x => (long)x);
        Tensor t = new Tensor(new TensorShape(d), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(s_seed++);
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 0.2 - 0.1);
        return t;
    }
}
