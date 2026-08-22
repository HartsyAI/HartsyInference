using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Vits;

/// <summary>VITS deterministic duration predictor (<c>DurationPredictor</c>, used when <c>use_sdp=False</c>): <c>conv_1 → channelLN → ReLU → conv_2 → channelLN → ReLU → proj</c> → log-duration <c>[T]</c>, channels-first <c>[1, hidden, T]</c>; the stochastic spline variant is staged separately.</summary>
public sealed unsafe class VitsDurationPredictor
{
    private readonly int _in, _filter, _kernel;
    private Tensor? _c1W, _c1B, _n1G, _n1B, _c2W, _c2B, _n2G, _n2B, _projW, _projB, _condW, _condB;

    public VitsDurationPredictor(VitsConfig cfg)
    {
        _in = cfg.HiddenChannels; _filter = cfg.DpFilterChannels; _kernel = cfg.DpKernelSize;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "dp")
    {
        _c1W = VitsWeights.Conv(w, $"{prefix}.conv_1"); _c1B = VitsWeights.Bias(w, $"{prefix}.conv_1");
        _n1G = w[$"{prefix}.norm_1.gamma"]; _n1B = w[$"{prefix}.norm_1.beta"];
        _c2W = VitsWeights.Conv(w, $"{prefix}.conv_2"); _c2B = VitsWeights.Bias(w, $"{prefix}.conv_2");
        _n2G = w[$"{prefix}.norm_2.gamma"]; _n2B = w[$"{prefix}.norm_2.beta"];
        _projW = VitsWeights.Conv(w, $"{prefix}.proj"); _projB = VitsWeights.Bias(w, $"{prefix}.proj");
        if (w.ContainsKey($"{prefix}.cond.weight") || w.ContainsKey($"{prefix}.cond.weight_g"))
        { _condW = VitsWeights.Conv(w, $"{prefix}.cond"); _condB = VitsWeights.Bias(w, $"{prefix}.cond"); }
    }

    /// <summary>Predicts log-durations from the encoder hidden <c>[1, hidden, T]</c>; when a speaker embedding <paramref name="g"/> <c>[1, gin, 1]</c> and a <c>cond</c> layer are present, <c>x = x + cond(g)</c> is added before the conv stack (MeloTTS/Bert-VITS2).</summary>
    public float[] Forward(IBackend backend, Tensor x, int t, Tensor? g = null)
    {
        int pad = (_kernel - 1) / 2;
        Tensor xc = x;
        if (g is not null && _condW is not null)
        {
            Tensor cg = new(new TensorShape(1, _in, 1), DType.F32);
            backend.Conv1d(cg, g, _condW, _condB, 1, 0, 0, 1, 1);
            xc = new Tensor(new TensorShape(1, _in, t), DType.F32);
            float* dst = (float*)xc.DataPointer, src = (float*)x.DataPointer, c = (float*)cg.DataPointer;
            for (int ch = 0; ch < _in; ch++)
                for (int j = 0; j < t; j++) dst[(long)ch * t + j] = src[(long)ch * t + j] + c[ch];
            cg.Dispose();
        }
        Tensor h1 = new(new TensorShape(1, _filter, t), DType.F32);
        backend.Conv1d(h1, xc, _c1W!, _c1B, 1, pad, pad, 1, 1);
        if (!ReferenceEquals(xc, x)) xc.Dispose();
        ChannelLnReluInPlace(h1, _n1G!, _n1B!, _filter, t);
        Tensor h2 = new(new TensorShape(1, _filter, t), DType.F32);
        backend.Conv1d(h2, h1, _c2W!, _c2B, 1, pad, pad, 1, 1); h1.Dispose();
        ChannelLnReluInPlace(h2, _n2G!, _n2B!, _filter, t);
        Tensor logw = new(new TensorShape(1, 1, t), DType.F32);
        backend.Conv1d(logw, h2, _projW!, _projB, 1, 0, 0, 1, 1); h2.Dispose();
        float[] outArr = new float[t];
        new Span<float>((void*)logw.DataPointer, t).CopyTo(outArr);
        logw.Dispose();
        return outArr;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_c1W, _c1B, _n1G, _n1B, _c2W, _c2B, _n2G, _n2B, _projW, _projB, _condW, _condB];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }

    // VITS DurationPredictor applies ReLU BEFORE the channel LayerNorm (conv -> relu -> norm), not after.
    private static void ChannelLnReluInPlace(Tensor x, Tensor gamma, Tensor beta, int ch, int t)
    {
        float* xp = (float*)x.DataPointer, g = (float*)gamma.DataPointer, be = (float*)beta.DataPointer;
        for (long n = 0; n < (long)ch * t; n++) if (xp[n] < 0) xp[n] = 0;     // ReLU first
        for (int j = 0; j < t; j++)
        {
            double mean = 0;
            for (int c = 0; c < ch; c++) mean += xp[(long)c * t + j];
            mean /= ch;
            double var = 0;
            for (int c = 0; c < ch; c++) { double d = xp[(long)c * t + j] - mean; var += d * d; }
            var /= ch;
            float inv = (float)(1.0 / Math.Sqrt(var + 1e-5));
            for (int c = 0; c < ch; c++)
                xp[(long)c * t + j] = ((xp[(long)c * t + j] - (float)mean) * inv) * g[c] + be[c];   // then LayerNorm
        }
    }
}
