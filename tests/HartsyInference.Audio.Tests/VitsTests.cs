using HartsyInference.Audio.Models.Vits;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Tests for the shared VITS SynthesizerTrn (Piper's core, reused by MeloTTS/GPT-SoVITS/OpenVoice):
/// the medium config, exact phoneme interspersing + length regulation, and a tiny synthetic-weights forward
/// of the FULL graph (rel-pos text encoder → duration predictor → length regulation → WaveNet flow →
/// HiFi-GAN) producing finite audio.</summary>
public sealed unsafe class VitsTests
{
    private static uint _rng = 0xC0FFEEu;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.2f; }
    private static Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor F1(int a) => Fill(new Tensor(new TensorShape(a), DType.F32));
    private static Tensor F2(int a, int b) => Fill(new Tensor(new TensorShape(a, b), DType.F32));
    private static Tensor F3(int a, int b, int c) => Fill(new Tensor(new TensorShape(a, b, c), DType.F32));

    [Fact]
    public void PiperMedium_Config()
    {
        VitsConfig c = VitsConfig.PiperMedium;
        Assert.Equal(192, c.InterChannels);
        Assert.Equal(6, c.NumEncoderLayers);
        Assert.Equal(2, c.NumHeads);
        Assert.Equal(256, c.HopLength);              // ∏[8,8,4]
        Assert.Equal(22_050, c.SampleRate);
        Assert.Equal("2", c.ResBlock);
        Assert.Equal(256, VitsConfig.PiperHigh.HopLength);  // ∏[8,8,2,2]
    }

    [Fact]
    public void Intersperse_InsertsBlankBetweenPhonemes_WithBosEos()
    {
        int[] outArr = VitsLengthRegulator.Intersperse([45, 23, 56], blank: 0, bos: 1, eos: 2);
        Assert.Equal(new[] { 1, 45, 0, 23, 0, 56, 2 }, outArr);
    }

    [Fact]
    public void Durations_CeilExpLengthScale_AndExpandRepeats()
    {
        // logw = [ln2, ln1] → exp = [2,1]; lengthScale 1 → ceil [2,1] → total 3.
        int[] d = VitsLengthRegulator.Durations([MathF.Log(2f), 0f], 1.0f);
        Assert.Equal(new[] { 2, 1 }, d);
        Assert.Equal(3, VitsLengthRegulator.TotalFrames(d));

        // Expand a [1ch, 2] sequence [10, 20] by [2,1] → [10,10,20].
        Tensor src = new(new TensorShape(1, 1, 2), DType.F32);
        ((float*)src.DataPointer)[0] = 10; ((float*)src.DataPointer)[1] = 20;
        Tensor dst = new(new TensorShape(1, 1, 3), DType.F32);
        VitsLengthRegulator.Expand((float*)src.DataPointer, (float*)dst.DataPointer, 1, 2, d, 3);
        float* dp = (float*)dst.DataPointer;
        Assert.Equal(new[] { 10f, 10f, 20f }, new[] { dp[0], dp[1], dp[2] });
        src.Dispose(); dst.Dispose();
    }

    [Fact]
    public void Synthesizer_SyntheticForward_ProducesFiniteAudio()
    {
        VitsConfig c = new()
        {
            NumVocab = 16, InterChannels = 8, HiddenChannels = 8, FilterChannels = 16, NumHeads = 2,
            NumEncoderLayers = 2, WindowSize = 4, UseSdp = false, DpFilterChannels = 8, DpKernelSize = 3,
            FlowLayers = 2, FlowFlows = 2, FlowKernelSize = 3,
            ResBlock = "2", ResBlockKernelSizes = [3], ResBlockDilations = [[1, 2]],
            UpsampleRates = [4, 4], UpsampleInitialChannel = 16, UpsampleKernelSizes = [8, 8],
        };
        using CpuBackend backend = new();
        using VitsSynthesizer synth = new(c);
        synth.LoadWeights(SyntheticWeights(c));

        int[] tokens = [1, 5, 0, 7, 0, 3, 2];     // interspersed phoneme ids
        float[] audio = synth.Infer(backend, tokens, lengthScale: 1.0f, noiseScale: 0.5f, seed: 11);
        Assert.True(audio.Length > 0);
        foreach (float s in audio) Assert.True(float.IsFinite(s));
    }

    private static Dictionary<string, Tensor> SyntheticWeights(VitsConfig c)
    {
        int h = c.HiddenChannels, inter = c.InterChannels, kc = h / c.NumHeads, half = inter / 2;
        Dictionary<string, Tensor> w = new()
        {
            ["enc_p.emb.weight"] = F2(c.NumVocab, h),
            ["enc_p.proj.weight"] = F3(2 * inter, h, 1),
            ["enc_p.proj.bias"] = F1(2 * inter),
            ["dp.conv_1.weight"] = F3(c.DpFilterChannels, h, 3), ["dp.conv_1.bias"] = F1(c.DpFilterChannels),
            ["dp.norm_1.gamma"] = F1(c.DpFilterChannels), ["dp.norm_1.beta"] = F1(c.DpFilterChannels),
            ["dp.conv_2.weight"] = F3(c.DpFilterChannels, c.DpFilterChannels, 3), ["dp.conv_2.bias"] = F1(c.DpFilterChannels),
            ["dp.norm_2.gamma"] = F1(c.DpFilterChannels), ["dp.norm_2.beta"] = F1(c.DpFilterChannels),
            ["dp.proj.weight"] = F3(1, c.DpFilterChannels, 1), ["dp.proj.bias"] = F1(1),
            ["dec.conv_pre.weight"] = F3(c.UpsampleInitialChannel, inter, 7), ["dec.conv_pre.bias"] = F1(c.UpsampleInitialChannel),
            ["dec.conv_post.weight"] = F3(1, c.UpsampleInitialChannel >> c.UpsampleRates.Count, 7), ["dec.conv_post.bias"] = F1(1),
        };
        // Encoder layers.
        for (int i = 0; i < c.NumEncoderLayers; i++)
        {
            string e = $"enc_p.encoder";
            foreach (string p in new[] { "conv_q", "conv_k", "conv_v", "conv_o" })
            { w[$"{e}.attn_layers.{i}.{p}.weight"] = F3(h, h, 1); w[$"{e}.attn_layers.{i}.{p}.bias"] = F1(h); }
            w[$"{e}.attn_layers.{i}.emb_rel_k"] = F3(1, 2 * c.WindowSize + 1, kc);
            w[$"{e}.attn_layers.{i}.emb_rel_v"] = F3(1, 2 * c.WindowSize + 1, kc);
            w[$"{e}.norm_layers_1.{i}.gamma"] = F1(h); w[$"{e}.norm_layers_1.{i}.beta"] = F1(h);
            w[$"{e}.norm_layers_2.{i}.gamma"] = F1(h); w[$"{e}.norm_layers_2.{i}.beta"] = F1(h);
            w[$"{e}.ffn_layers.{i}.conv_1.weight"] = F3(c.FilterChannels, h, 3); w[$"{e}.ffn_layers.{i}.conv_1.bias"] = F1(c.FilterChannels);
            w[$"{e}.ffn_layers.{i}.conv_2.weight"] = F3(h, c.FilterChannels, 3); w[$"{e}.ffn_layers.{i}.conv_2.bias"] = F1(h);
        }
        // Flow couplings (at flows index 2*i).
        for (int i = 0; i < c.FlowFlows; i++)
        {
            string f = $"flow.flows.{2 * i}";
            w[$"{f}.pre.weight"] = F3(h, half, 1); w[$"{f}.pre.bias"] = F1(h);
            w[$"{f}.post.weight"] = F3(half, h, 1); w[$"{f}.post.bias"] = F1(half);
            for (int j = 0; j < c.FlowLayers; j++)
            {
                w[$"{f}.enc.in_layers.{j}.weight"] = F3(2 * h, h, c.FlowKernelSize); w[$"{f}.enc.in_layers.{j}.bias"] = F1(2 * h);
                int rsCh = j < c.FlowLayers - 1 ? 2 * h : h;
                w[$"{f}.enc.res_skip_layers.{j}.weight"] = F3(rsCh, h, 1); w[$"{f}.enc.res_skip_layers.{j}.bias"] = F1(rsCh);
            }
        }
        // Decoder upsamplers + resblocks.
        for (int i = 0; i < c.UpsampleRates.Count; i++)
        {
            int inCh = c.UpsampleInitialChannel >> i, outCh = inCh >> 1;
            w[$"dec.ups.{i}.weight"] = F3(inCh, outCh, c.UpsampleKernelSizes[i]); w[$"dec.ups.{i}.bias"] = F1(outCh);
            for (int k = 0; k < c.ResBlockKernelSizes.Count; k++)
            {
                string rb = $"dec.resblocks.{i * c.ResBlockKernelSizes.Count + k}";
                for (int dn = 0; dn < c.ResBlockDilations[k].Count; dn++)
                { w[$"{rb}.convs1.{dn}.weight"] = F3(outCh, outCh, c.ResBlockKernelSizes[k]); w[$"{rb}.convs1.{dn}.bias"] = F1(outCh); }
            }
        }
        return w;
    }
}
