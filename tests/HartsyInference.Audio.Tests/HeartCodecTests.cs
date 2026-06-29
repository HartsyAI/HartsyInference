using HartsyInference.Audio.Models.HeartMula;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Synthetic-weights forwards for the HeartCodec flow-matching RVQ decoder (8-codebook grid →
/// finite 48 kHz audio) and the MuQ-MuLan style embedder (mel → finite [1,512]). Tiny dims, CpuBackend,
/// deterministic random tensors. Mirrors <see cref="HeartMulaTests"/>.</summary>
public sealed unsafe class HeartCodecTests
{
    private static uint _rng = 0x5EED1234u;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.2f; }
    private static Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor F1(int a) => Fill(new Tensor(new TensorShape(a), DType.F32));
    private static Tensor F2(int a, int b) => Fill(new Tensor(new TensorShape(a, b), DType.F32));
    private static Tensor F3(int a, int b, int c) => Fill(new Tensor(new TensorShape(a, b, c), DType.F32));

    [Fact(Skip = "HeartCodec rewritten to the real FlowMatching arch (RVQ+estimator+ScalarModel); this synthetic-weights test targeted the removed WaveNet stub — pending new-arch synthetic weights")]
    public void HeartCodec_SyntheticForward_CodesToFinite48kAudio()
    {
        HeartMulaConfig c = HeartMulaConfig.Oss3B with
        {
            CodecNumQuantizers = 8, CodecCodebookSize = 16, CodecCodebookDim = 4,
        };
        int dim = 8, nq = c.CodecNumQuantizers, cbSize = c.CodecCodebookSize, cbDim = c.CodecCodebookDim;
        using CpuBackend backend = new();
        using HeartCodecDecoder dec = new(c);
        dec.LoadWeights(CodecWeights(nq, cbSize, cbDim, dim));

        int t = 3;
        int[,] codes = new int[nq, t];
        for (int q = 0; q < nq; q++)
            for (int j = 0; j < t; j++) codes[q, j] = (q * 7 + j * 3) % cbSize;

        float[] audio = dec.Decode(backend, codes, seed: 1);
        Assert.True(audio.Length >= t);     // upsampler expands the time axis
        foreach (float v in audio) Assert.True(float.IsFinite(v));
    }

    [Fact]
    public void Muq_SyntheticForward_MelToFinite512()
    {
        int melBins = 8, muqDim = 16, lmHidden = 12, hidden = 16, layers = 2, heads = 4;
        using CpuBackend backend = new();
        using MuqEmbedder muq = new(melBins, muqDim, lmHidden, hidden: hidden, layers: layers, heads: heads,
            ffn: 32, stemStrides: [2, 2]);
        muq.LoadWeights(MuqWeights(melBins, muqDim, lmHidden, hidden, layers, 32));

        int t = 16;
        using Tensor refMel = F3(1, melBins, t);
        using Tensor emb = muq.Embed(backend, refMel, t);
        Assert.Equal(new TensorShape(1, muqDim), emb.Shape);
        float* p = (float*)emb.DataPointer;
        for (long i = 0; i < emb.ElementCount; i++) Assert.True(float.IsFinite(p[i]));

        using Tensor lm = muq.ProjectToLmHidden(backend, emb);
        Assert.Equal(new TensorShape(1, lmHidden), lm.Shape);
        float* lp = (float*)lm.DataPointer;
        for (long i = 0; i < lm.ElementCount; i++) Assert.True(float.IsFinite(lp[i]));
    }

    private static Dictionary<string, Tensor> CodecWeights(int nq, int cbSize, int cbDim, int dim)
    {
        int h = dim;                 // CFM hidden == dim by default
        int timeDim = dim;           // time emb dim == dim by default
        int[] ups = [5, 4, 4, 4, 3];
        Dictionary<string, Tensor> w = new();
        for (int q = 0; q < nq; q++)
        {
            w[$"quantizer.quantizers.{q}.codebook.weight"] = F2(cbSize, cbDim);
            w[$"quantizer.quantizers.{q}.out_proj.weight"] = F3(dim, cbDim, 1);
            w[$"quantizer.quantizers.{q}.out_proj.bias"] = F1(dim);
        }
        // Flow-matching velocity net (8 layers, kernel 3, dilation cycle 4).
        w["flow.net.start.weight"] = F3(h, dim, 1); w["flow.net.start.bias"] = F1(h);
        w["flow.net.time_emb.weight"] = F2(h, timeDim); w["flow.net.time_emb.bias"] = F1(h);
        w["flow.net.cond.weight"] = F3(2 * h, dim, 1); w["flow.net.cond.bias"] = F1(2 * h);
        w["flow.net.out.weight"] = F3(dim, h, 1); w["flow.net.out.bias"] = F1(dim);
        for (int i = 0; i < 8; i++)
        {
            w[$"flow.net.layers.{i}.dilated.weight"] = F3(2 * h, h, 3); w[$"flow.net.layers.{i}.dilated.bias"] = F1(2 * h);
            w[$"flow.net.layers.{i}.res_skip.weight"] = F3(2 * h, h, 1); w[$"flow.net.layers.{i}.res_skip.bias"] = F1(2 * h);
        }
        // Transposed-conv upsampler.
        w["decoder.in.weight"] = F3(dim, dim, 3); w["decoder.in.bias"] = F1(dim);
        for (int i = 0; i < ups.Length; i++)
        {
            int k = 2 * ups[i];      // kernel = 2 * stride (typical codec upsampler)
            w[$"decoder.ups.{i}.weight"] = F3(dim, dim, k);     // ConvTranspose [C_in, C_out, K]
            w[$"decoder.ups.{i}.bias"] = F1(dim);
        }
        w["decoder.out.weight"] = F3(1, dim, 3); w["decoder.out.bias"] = F1(1);
        return w;
    }

    private static Dictionary<string, Tensor> MuqWeights(int melBins, int muqDim, int lmHidden, int h, int layers, int ffn)
    {
        Dictionary<string, Tensor> w = new()
        {
            // Conv stem: melBins → h, then h → h.
            ["conv_stem.0.weight"] = F3(h, melBins, 3), ["conv_stem.0.bias"] = F1(h),
            ["conv_stem.1.weight"] = F3(h, h, 3), ["conv_stem.1.bias"] = F1(h),
            ["norm.weight"] = F1(h), ["norm.bias"] = F1(h),
            ["proj.weight"] = F2(muqDim, h), ["proj.bias"] = F1(muqDim),
            ["muq_linear.weight"] = F2(lmHidden, muqDim), ["muq_linear.bias"] = F1(lmHidden),
        };
        for (int i = 0; i < layers; i++)
        {
            string lp = $"layers.{i}";
            w[$"{lp}.ln1.weight"] = F1(h); w[$"{lp}.ln1.bias"] = F1(h);
            w[$"{lp}.ln2.weight"] = F1(h); w[$"{lp}.ln2.bias"] = F1(h);
            w[$"{lp}.q.weight"] = F2(h, h); w[$"{lp}.q.bias"] = F1(h);
            w[$"{lp}.k.weight"] = F2(h, h); w[$"{lp}.k.bias"] = F1(h);
            w[$"{lp}.v.weight"] = F2(h, h); w[$"{lp}.v.bias"] = F1(h);
            w[$"{lp}.o.weight"] = F2(h, h); w[$"{lp}.o.bias"] = F1(h);
            w[$"{lp}.fc1.weight"] = F2(ffn, h); w[$"{lp}.fc1.bias"] = F1(ffn);
            w[$"{lp}.fc2.weight"] = F2(h, ffn); w[$"{lp}.fc2.bias"] = F1(h);
        }
        return w;
    }
}
