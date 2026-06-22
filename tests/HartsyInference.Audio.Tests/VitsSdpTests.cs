using HartsyInference.Audio.Models.Vits;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Tests for the VITS stochastic duration predictor (Piper's <c>use_sdp=True</c> default): the
/// reverse flow path (DDSConv + ConvFlow rational-quadratic spline + Flip + ElementwiseAffine) producing
/// finite per-text-frame log-durations, and the unconstrained rational-quadratic spline transform itself,
/// verified finite + monotone on a small synthetic input.</summary>
public sealed unsafe class VitsSdpTests
{
    private static uint _rng = 0xBADC0DEu;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.2f; }
    private static Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor F1(int a) => Fill(new Tensor(new TensorShape(a), DType.F32));
    private static Tensor F2(int a, int b) => Fill(new Tensor(new TensorShape(a, b), DType.F32));
    private static Tensor F3(int a, int b, int c) => Fill(new Tensor(new TensorShape(a, b, c), DType.F32));

    [Fact]
    public void Sdp_SyntheticForward_ProducesFiniteLogDurations()
    {
        VitsConfig c = new()
        {
            NumVocab = 16, InterChannels = 8, HiddenChannels = 8, FilterChannels = 16, NumHeads = 2,
            NumEncoderLayers = 1, WindowSize = 4, UseSdp = true, SdpFlows = 4,
        };
        using CpuBackend backend = new();
        VitsStochasticDurationPredictor sdp = new(c);
        sdp.LoadWeights(SdpWeights(c));

        int tx = 5;
        Tensor x = F3(1, c.HiddenChannels, tx);
        float[] logw = sdp.Forward(backend, x, tx, noiseScaleW: 0.8f, seed: 7);
        x.Dispose();

        Assert.Equal(tx, logw.Length);
        foreach (float v in logw) Assert.True(float.IsFinite(v));
    }

    [Fact]
    public void Sdp_Deterministic_SameSeedSameOutput()
    {
        VitsConfig c = new()
        {
            NumVocab = 16, InterChannels = 8, HiddenChannels = 8, FilterChannels = 16, UseSdp = true, SdpFlows = 4,
        };
        using CpuBackend backend = new();
        VitsStochasticDurationPredictor sdp = new(c);
        sdp.LoadWeights(SdpWeights(c));

        int tx = 4;
        Tensor x = F3(1, c.HiddenChannels, tx);
        float[] a = sdp.Forward(backend, x, tx, 0.8f, seed: 3);
        float[] b = sdp.Forward(backend, x, tx, 0.8f, seed: 3);
        x.Dispose();
        for (int i = 0; i < tx; i++) Assert.Equal(a[i], b[i], 5);
    }

    [Fact]
    public void Synthesizer_SdpBranch_ProducesFiniteAudio()
    {
        VitsConfig c = new()
        {
            NumVocab = 16, InterChannels = 8, HiddenChannels = 8, FilterChannels = 16, NumHeads = 2,
            NumEncoderLayers = 1, WindowSize = 4, UseSdp = true, SdpFlows = 4,
            FlowLayers = 2, FlowFlows = 2, FlowKernelSize = 3,
            ResBlock = "2", ResBlockKernelSizes = [3], ResBlockDilations = [[1, 2]],
            UpsampleRates = [4, 4], UpsampleInitialChannel = 16, UpsampleKernelSizes = [8, 8],
        };
        using CpuBackend backend = new();
        using VitsSynthesizer synth = new(c);
        synth.LoadWeights(FullSdpWeights(c));

        int[] tokens = [1, 5, 0, 7, 0, 3, 2];
        float[] audio = synth.Infer(backend, tokens, lengthScale: 1.0f, noiseScale: 0.5f, seed: 11);
        Assert.True(audio.Length > 0);
        foreach (float s in audio) Assert.True(float.IsFinite(s));
    }

    [Fact]
    public void Spline_Inverse_IsFiniteAndMonotone()
    {
        const int numBins = 10;
        float[] uw = new float[numBins];
        float[] uh = new float[numBins];
        float[] ud = new float[numBins + 1];
        uint r = 0x1234u;
        for (int i = 0; i < numBins; i++) { uw[i] = NextSmall(ref r); uh[i] = NextSmall(ref r); }
        for (int i = 0; i <= numBins; i++) ud[i] = NextSmall(ref r);

        // The inverse spline must be finite and strictly increasing across its input domain.
        float prevOut = float.NegativeInfinity;
        for (float y = -8f; y <= 8f; y += 0.25f)
        {
            float xInv = VitsStochasticDurationPredictor.UnconstrainedRationalQuadraticSpline(
                y, uw, uh, ud, inverse: true);
            Assert.True(float.IsFinite(xInv));
            Assert.True(xInv > prevOut, $"non-monotone at y={y}: {xInv} <= {prevOut}");
            prevOut = xInv;
        }

        // Round-trip: forward(inverse(y)) ≈ y inside the spline support [-5, 5].
        for (float y = -4f; y <= 4f; y += 0.5f)
        {
            float xInv = VitsStochasticDurationPredictor.UnconstrainedRationalQuadraticSpline(y, uw, uh, ud, inverse: true);
            float yFwd = VitsStochasticDurationPredictor.UnconstrainedRationalQuadraticSpline(xInv, uw, uh, ud, inverse: false);
            Assert.Equal(y, yFwd, 2);
        }
    }

    private static float NextSmall(ref uint s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return ((s & 0xFFFF) / 65535f - 0.5f) * 2f; }

    private static Dictionary<string, Tensor> SdpWeights(VitsConfig c, string prefix = "sdp")
    {
        int h = c.HiddenChannels, kernel = 3, ddsLayers = 3, flowLayers = 3, numBins = 10;
        Dictionary<string, Tensor> w = new()
        {
            [$"{prefix}.pre.weight"] = F3(h, h, 1), [$"{prefix}.pre.bias"] = F1(h),
            [$"{prefix}.proj.weight"] = F3(h, h, 1), [$"{prefix}.proj.bias"] = F1(h),
            [$"{prefix}.flows.0.m"] = F2(2, 1), [$"{prefix}.flows.0.logs"] = F2(2, 1),
        };
        AddDdsConv(w, $"{prefix}.convs", h, kernel, ddsLayers);
        for (int j = 0; j < c.SdpFlows; j++)
        {
            string f = $"{prefix}.flows.{2 * j + 1}";
            w[$"{f}.pre.weight"] = F3(h, 1, 1); w[$"{f}.pre.bias"] = F1(h);
            w[$"{f}.proj.weight"] = F3(numBins * 3 - 1, h, 1); w[$"{f}.proj.bias"] = F1(numBins * 3 - 1);
            AddDdsConv(w, $"{f}.convs", h, kernel, flowLayers);
        }
        return w;
    }

    private static void AddDdsConv(Dictionary<string, Tensor> w, string prefix, int ch, int kernel, int layers)
    {
        for (int i = 0; i < layers; i++)
        {
            w[$"{prefix}.convs_sep.{i}.weight"] = F3(ch, 1, kernel); w[$"{prefix}.convs_sep.{i}.bias"] = F1(ch);
            w[$"{prefix}.convs_1x1.{i}.weight"] = F3(ch, ch, 1); w[$"{prefix}.convs_1x1.{i}.bias"] = F1(ch);
            w[$"{prefix}.norms_1.{i}.gamma"] = F1(ch); w[$"{prefix}.norms_1.{i}.beta"] = F1(ch);
            w[$"{prefix}.norms_2.{i}.gamma"] = F1(ch); w[$"{prefix}.norms_2.{i}.beta"] = F1(ch);
        }
    }

    private static Dictionary<string, Tensor> FullSdpWeights(VitsConfig c)
    {
        int h = c.HiddenChannels, inter = c.InterChannels, kc = h / c.NumHeads, half = inter / 2;
        Dictionary<string, Tensor> w = new()
        {
            ["enc_p.emb.weight"] = F2(c.NumVocab, h),
            ["enc_p.proj.weight"] = F3(2 * inter, h, 1),
            ["enc_p.proj.bias"] = F1(2 * inter),
            ["dec.conv_pre.weight"] = F3(c.UpsampleInitialChannel, inter, 7), ["dec.conv_pre.bias"] = F1(c.UpsampleInitialChannel),
            ["dec.conv_post.weight"] = F3(1, c.UpsampleInitialChannel >> c.UpsampleRates.Count, 7), ["dec.conv_post.bias"] = F1(1),
        };
        foreach (KeyValuePair<string, Tensor> kv in SdpWeights(c)) w[kv.Key] = kv.Value;
        for (int i = 0; i < c.NumEncoderLayers; i++)
        {
            string e = "enc_p.encoder";
            foreach (string p in new[] { "conv_q", "conv_k", "conv_v", "conv_o" })
            { w[$"{e}.attn_layers.{i}.{p}.weight"] = F3(h, h, 1); w[$"{e}.attn_layers.{i}.{p}.bias"] = F1(h); }
            w[$"{e}.attn_layers.{i}.emb_rel_k"] = F3(1, 2 * c.WindowSize + 1, kc);
            w[$"{e}.attn_layers.{i}.emb_rel_v"] = F3(1, 2 * c.WindowSize + 1, kc);
            w[$"{e}.norm_layers_1.{i}.gamma"] = F1(h); w[$"{e}.norm_layers_1.{i}.beta"] = F1(h);
            w[$"{e}.norm_layers_2.{i}.gamma"] = F1(h); w[$"{e}.norm_layers_2.{i}.beta"] = F1(h);
            w[$"{e}.ffn_layers.{i}.conv_1.weight"] = F3(c.FilterChannels, h, 3); w[$"{e}.ffn_layers.{i}.conv_1.bias"] = F1(c.FilterChannels);
            w[$"{e}.ffn_layers.{i}.conv_2.weight"] = F3(h, c.FilterChannels, 3); w[$"{e}.ffn_layers.{i}.conv_2.bias"] = F1(h);
        }
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
