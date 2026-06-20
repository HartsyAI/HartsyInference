using HartsyInference.Audio.Models.Vits;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Tests for OpenVoice's Tone Color Converter — a synthetic-weights forward through posterior
/// encoder → speaker-conditioned flow (forward under source g, reverse under target g) → speaker-conditioned
/// HiFi-GAN, producing finite audio. Exercises the new VITS <c>g</c>-conditioning path end-to-end.</summary>
public sealed unsafe class OpenVoiceTests
{
    private static uint _rng = 0x5EED5EEDu;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.2f; }
    private static Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor F1(int a) => Fill(new Tensor(new TensorShape(a), DType.F32));
    private static Tensor F3(int a, int b, int c) => Fill(new Tensor(new TensorShape(a, b, c), DType.F32));

    [Fact]
    public void ToneColorConverter_SyntheticForward_ProducesFiniteAudio()
    {
        int gin = 4, spec = 6, post = 2;
        VitsConfig c = new()
        {
            InterChannels = 8, HiddenChannels = 8, GinChannels = gin, FlowLayers = 2, FlowFlows = 2, FlowKernelSize = 3,
            ResBlock = "2", ResBlockKernelSizes = [3], ResBlockDilations = [[1, 2]],
            UpsampleRates = [4, 4], UpsampleInitialChannel = 16, UpsampleKernelSizes = [8, 8],
        };
        using CpuBackend backend = new();
        using OpenVoicePipeline pipe = new(c, specChannels: spec, posteriorLayers: post);
        pipe.LoadWeights(Weights(c, gin, spec, post));

        int t = 4;
        using Tensor specT = F3(1, spec, t);
        using Tensor gSrc = F3(1, gin, 1);
        using Tensor gTgt = F3(1, gin, 1);
        float[] audio = pipe.Convert(backend, specT, t, gSrc, gTgt, seed: 3);
        Assert.True(audio.Length > 0);
        foreach (float s in audio) Assert.True(float.IsFinite(s));
    }

    private static Dictionary<string, Tensor> Weights(VitsConfig c, int gin, int spec, int post)
    {
        int h = c.HiddenChannels, inter = c.InterChannels, half = inter / 2;
        Dictionary<string, Tensor> w = new()
        {
            ["enc_q.pre.weight"] = F3(h, spec, 1), ["enc_q.pre.bias"] = F1(h),
            ["enc_q.proj.weight"] = F3(2 * inter, h, 1), ["enc_q.proj.bias"] = F1(2 * inter),
            ["enc_q.enc.cond_layer.weight"] = F3(2 * h * post, gin, 1), ["enc_q.enc.cond_layer.bias"] = F1(2 * h * post),
            ["dec.conv_pre.weight"] = F3(c.UpsampleInitialChannel, inter, 7), ["dec.conv_pre.bias"] = F1(c.UpsampleInitialChannel),
            ["dec.cond.weight"] = F3(c.UpsampleInitialChannel, gin, 1), ["dec.cond.bias"] = F1(c.UpsampleInitialChannel),
            ["dec.conv_post.weight"] = F3(1, c.UpsampleInitialChannel >> c.UpsampleRates.Count, 7), ["dec.conv_post.bias"] = F1(1),
        };
        for (int j = 0; j < post; j++)
        {
            w[$"enc_q.enc.in_layers.{j}.weight"] = F3(2 * h, h, c.FlowKernelSize); w[$"enc_q.enc.in_layers.{j}.bias"] = F1(2 * h);
            int rs = j < post - 1 ? 2 * h : h;
            w[$"enc_q.enc.res_skip_layers.{j}.weight"] = F3(rs, h, 1); w[$"enc_q.enc.res_skip_layers.{j}.bias"] = F1(rs);
        }
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
        for (int i = 0; i < c.UpsampleRates.Count; i++)
        {
            int inCh = c.UpsampleInitialChannel >> i, outCh = inCh >> 1;
            w[$"dec.ups.{i}.weight"] = F3(inCh, outCh, c.UpsampleKernelSizes[i]); w[$"dec.ups.{i}.bias"] = F1(outCh);
            for (int dn = 0; dn < c.ResBlockDilations[0].Count; dn++)
            { w[$"dec.resblocks.{i}.convs1.{dn}.weight"] = F3(outCh, outCh, 3); w[$"dec.resblocks.{i}.convs1.{dn}.bias"] = F1(outCh); }
        }
        return w;
    }
}
