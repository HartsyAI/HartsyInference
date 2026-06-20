using HartsyInference.Audio.Models.Rvc;
using HartsyInference.Audio.Models.Vits;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Tests for RVC: the mel coarse-pitch quantization (exact) and a synthetic-weights forward of the
/// full pipeline (content + F0 → content+pitch encoder → reused VITS flow → NSF-HiFi-GAN with source
/// injection) producing finite audio.</summary>
public sealed unsafe class RvcTests
{
    private static uint _rng = 0x12C0FFu;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.2f; }
    private static Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor F1(int a) => Fill(new Tensor(new TensorShape(a), DType.F32));
    private static Tensor F2(int a, int b) => Fill(new Tensor(new TensorShape(a, b), DType.F32));
    private static Tensor F3(int a, int b, int c) => Fill(new Tensor(new TensorShape(a, b, c), DType.F32));

    [Fact]
    public void CoarseQuantize_MapsHzToBins1To255()
    {
        int[] bins = RvcPitch.CoarseQuantize([0f, 50f, 1100f, 220f], 50f, 1100f);
        Assert.Equal(1, bins[0]);                 // f0=0 → bin 1
        Assert.Equal(1, bins[1]);                 // f0=f0_min → bin 1
        Assert.Equal(255, bins[2]);               // f0=f0_max → bin 255
        Assert.InRange(bins[3], 2, 254);          // mid pitch in range
        // Pitch shift up an octave doubles f0.
        float[] shifted = RvcPitch.Shift([220f], 12f);
        Assert.Equal(440f, shifted[0], 3);
    }

    [Fact]
    public void Pipeline_SyntheticForward_ContentPlusF0ToFiniteAudio()
    {
        int gin = 4, content = 6, spk = 3;
        RvcConfig cfg = new()
        {
            Core = new VitsConfig
            {
                InterChannels = 8, HiddenChannels = 8, FilterChannels = 16, NumHeads = 2, NumEncoderLayers = 1, WindowSize = 4,
                GinChannels = gin, FlowFlows = 2, FlowLayers = 2, FlowKernelSize = 3,
                ResBlock = "1", ResBlockKernelSizes = [3], ResBlockDilations = [[1]],
                UpsampleRates = [4, 4], UpsampleInitialChannel = 16, UpsampleKernelSizes = [8, 8], SampleRate = 16_000,
            },
            ContentDim = content, SpkEmbedDim = spk,
        };
        using CpuBackend backend = new();
        using RvcPipeline pipe = new(cfg);
        pipe.LoadWeights(Weights(cfg, gin, content, spk));

        int t = 3;
        using Tensor feats = F3(1, content, t);
        float[] f0 = [150f, 200f, 0f];     // last frame unvoiced
        float[] audio = pipe.Convert(backend, feats, f0, sid: 1, seed: 5);
        Assert.True(audio.Length > 0);
        foreach (float s in audio) Assert.True(float.IsFinite(s));
    }

    private static Dictionary<string, Tensor> Weights(RvcConfig cfg, int gin, int content, int spk)
    {
        VitsConfig c = cfg.Core;
        int h = c.HiddenChannels, inter = c.InterChannels, kc = h / c.NumHeads, half = inter / 2;
        Dictionary<string, Tensor> w = new()
        {
            ["enc_p.emb_phone.weight"] = F2(h, content), ["enc_p.emb_phone.bias"] = F1(h),
            ["enc_p.emb_pitch.weight"] = F2(256, h),
            ["enc_p.proj.weight"] = F3(2 * inter, h, 1), ["enc_p.proj.bias"] = F1(2 * inter),
            ["emb_g.weight"] = F2(spk, gin),
            ["dec.m_source.l_linear.weight"] = F2(1, 1), ["dec.m_source.l_linear.bias"] = F1(1),
            ["dec.conv_pre.weight"] = F3(c.UpsampleInitialChannel, inter, 7), ["dec.conv_pre.bias"] = F1(c.UpsampleInitialChannel),
            ["dec.cond.weight"] = F3(c.UpsampleInitialChannel, gin, 1), ["dec.cond.bias"] = F1(c.UpsampleInitialChannel),
            ["dec.conv_post.weight"] = F3(1, c.UpsampleInitialChannel >> c.UpsampleRates.Count, 7), ["dec.conv_post.bias"] = F1(1),
        };
        // enc_p encoder layer 0.
        string e = "enc_p.encoder";
        foreach (string p in new[] { "conv_q", "conv_k", "conv_v", "conv_o" })
        { w[$"{e}.attn_layers.0.{p}.weight"] = F3(h, h, 1); w[$"{e}.attn_layers.0.{p}.bias"] = F1(h); }
        w[$"{e}.attn_layers.0.emb_rel_k"] = F3(1, 2 * c.WindowSize + 1, kc);
        w[$"{e}.attn_layers.0.emb_rel_v"] = F3(1, 2 * c.WindowSize + 1, kc);
        w[$"{e}.norm_layers_1.0.gamma"] = F1(h); w[$"{e}.norm_layers_1.0.beta"] = F1(h);
        w[$"{e}.norm_layers_2.0.gamma"] = F1(h); w[$"{e}.norm_layers_2.0.beta"] = F1(h);
        w[$"{e}.ffn_layers.0.conv_1.weight"] = F3(c.FilterChannels, h, 3); w[$"{e}.ffn_layers.0.conv_1.bias"] = F1(c.FilterChannels);
        w[$"{e}.ffn_layers.0.conv_2.weight"] = F3(h, c.FilterChannels, 3); w[$"{e}.ffn_layers.0.conv_2.bias"] = F1(h);
        // Flow.
        for (int i = 0; i < c.FlowFlows; i++)
        {
            string f = $"flow.flows.{2 * i}";
            w[$"{f}.pre.weight"] = F3(h, half, 1); w[$"{f}.pre.bias"] = F1(h);
            w[$"{f}.post.weight"] = F3(half, h, 1); w[$"{f}.post.bias"] = F1(half);
            w[$"{f}.enc.cond_layer.weight"] = F3(2 * h * c.FlowLayers, gin, 1); w[$"{f}.enc.cond_layer.bias"] = F1(2 * h * c.FlowLayers);
            for (int j = 0; j < c.FlowLayers; j++)
            {
                w[$"{f}.enc.in_layers.{j}.weight"] = F3(2 * h, h, c.FlowKernelSize); w[$"{f}.enc.in_layers.{j}.bias"] = F1(2 * h);
                int rs = j < c.FlowLayers - 1 ? 2 * h : h;
                w[$"{f}.enc.res_skip_layers.{j}.weight"] = F3(rs, h, 1); w[$"{f}.enc.res_skip_layers.{j}.bias"] = F1(rs);
            }
        }
        // Decoder upsamplers + noise_convs + type-1 resblocks.
        for (int i = 0; i < c.UpsampleRates.Count; i++)
        {
            int inCh = c.UpsampleInitialChannel >> i, outCh = inCh >> 1;
            w[$"dec.ups.{i}.weight"] = F3(inCh, outCh, c.UpsampleKernelSizes[i]); w[$"dec.ups.{i}.bias"] = F1(outCh);
            bool last = i + 1 == c.UpsampleRates.Count;
            int strideF0 = 1; for (int r = i + 1; r < c.UpsampleRates.Count; r++) strideF0 *= c.UpsampleRates[r];
            int nk = last ? 1 : strideF0 * 2;
            w[$"dec.noise_convs.{i}.weight"] = F3(outCh, 1, nk); w[$"dec.noise_convs.{i}.bias"] = F1(outCh);
            w[$"dec.resblocks.{i}.convs1.0.weight"] = F3(outCh, outCh, 3); w[$"dec.resblocks.{i}.convs1.0.bias"] = F1(outCh);
            w[$"dec.resblocks.{i}.convs2.0.weight"] = F3(outCh, outCh, 3); w[$"dec.resblocks.{i}.convs2.0.bias"] = F1(outCh);
        }
        return w;
    }
}
