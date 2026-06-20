using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.ResembleEnhance;

/// <summary>Resemble-enhance CFM velocity network (`WN`): a DiffWave/WaveNet-style stack of dilated gated 1D
/// convolutions. The current latent <c>x [1, latent, T]</c> is the input; a sinusoidal time embedding is the
/// global conditioning; the (InstanceNorm'd) conditioning mel is the local conditioning added per layer; skips
/// sum and project to the latent velocity. Implements <see cref="ICfmEstimator"/> so the existing OT-CFM solver
/// drives it (CFG off — the <c>mu</c>/<c>spk</c> args are unused; <c>cond</c> carries the conditioning mel).</summary>
public sealed unsafe class ResembleWnEstimator : ICfmEstimator
{
    private readonly ResembleEnhanceConfig _cfg;
    private readonly int _hidden, _layers, _kernel, _cycle, _latent, _melDim, _timeDim;
    private Tensor? _startW, _startB, _timeW, _timeB, _localW, _localB, _outW, _outB;
    private readonly Tensor?[] _inW, _inB, _rsW, _rsB;

    public ResembleWnEstimator(ResembleEnhanceConfig cfg, int melDim)
    {
        _cfg = cfg; _hidden = cfg.WnHidden; _layers = cfg.WnLayers; _kernel = cfg.WnKernel;
        _cycle = cfg.WnDilationCycle; _latent = cfg.LatentDim; _melDim = melDim; _timeDim = cfg.TimeEmbDim;
        _inW = new Tensor?[_layers]; _inB = new Tensor?[_layers]; _rsW = new Tensor?[_layers]; _rsB = new Tensor?[_layers];
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "lcfm.cfm.net")
    {
        _startW = WhisperOps.EnsureF32(w[$"{prefix}.start.weight"]); _startB = Bias(w, $"{prefix}.start.bias");
        _timeW = WhisperOps.EnsureF32(w[$"{prefix}.global_cond.weight"]); _timeB = Bias(w, $"{prefix}.global_cond.bias");
        _localW = WhisperOps.EnsureF32(w[$"{prefix}.local_cond.weight"]); _localB = Bias(w, $"{prefix}.local_cond.bias");
        _outW = WhisperOps.EnsureF32(w[$"{prefix}.out.weight"]); _outB = Bias(w, $"{prefix}.out.bias");
        for (int i = 0; i < _layers; i++)
        {
            _inW[i] = WhisperOps.EnsureF32(w[$"{prefix}.layers.{i}.dilated.weight"]); _inB[i] = Bias(w, $"{prefix}.layers.{i}.dilated.bias");
            _rsW[i] = WhisperOps.EnsureF32(w[$"{prefix}.layers.{i}.res_skip.weight"]); _rsB[i] = Bias(w, $"{prefix}.layers.{i}.res_skip.bias");
        }
    }

    /// <summary>Velocity estimate. <paramref name="x"/> is the latent <c>[1, latent, T]</c>; <paramref name="cond"/>
    /// the conditioning mel <c>[1, melDim, T]</c>; <paramref name="t"/> the flow time.</summary>
    public Tensor Estimate(IBackend backend, Tensor x, Tensor mu, float t, Tensor spk, Tensor cond)
    {
        int h = _hidden, tl = (int)x.Shape[2];
        Tensor hid = new(new TensorShape(1, h, tl), DType.F32);
        backend.Conv1d(hid, x, _startW!, _startB, 1, 0, 0, 1, 1);

        // Global cond: sinusoidal time emb → Linear → [h], broadcast-added.
        float[] te = SinTimeEmb(t, _timeDim);
        Tensor teT = new(new TensorShape(1, 1, _timeDim), DType.F32);
        te.AsSpan().CopyTo(new Span<float>((void*)teT.DataPointer, _timeDim));
        Tensor teProj = WhisperOps.ProjectLinear(backend, teT, _timeW!, _timeB, 1, 1, _timeDim, h); teT.Dispose();
        float* hp = (float*)hid.DataPointer; float* gp = (float*)teProj.DataPointer;
        for (int c = 0; c < h; c++) for (int j = 0; j < tl; j++) hp[(long)c * tl + j] += gp[c];
        teProj.Dispose();

        // Local cond: InstanceNorm(mel) → Conv1d → [2h, T].
        Tensor cn = InstanceNorm(cond, _melDim, tl);
        Tensor local = new(new TensorShape(1, 2 * h, tl), DType.F32);
        backend.Conv1d(local, cn, _localW!, _localB, 1, 0, 0, 1, 1); cn.Dispose();
        float* lp = (float*)local.DataPointer;

        Tensor skipAcc = new(new TensorShape(1, h, tl), DType.F32);
        float* sa = (float*)skipAcc.DataPointer;
        for (long n = 0; n < (long)h * tl; n++) sa[n] = 0;

        for (int i = 0; i < _layers; i++)
        {
            int dilation = 1 << (i % _cycle);
            int pad = dilation * (_kernel - 1) / 2;
            Tensor inl = new(new TensorShape(1, 2 * h, tl), DType.F32);
            backend.Conv1d(inl, hid, _inW[i]!, _inB[i], 1, pad, pad, dilation, 1);
            float* ip = (float*)inl.DataPointer;
            // Add local cond, then gated tanh*sigmoid → acts [h, T].
            Tensor acts = new(new TensorShape(1, h, tl), DType.F32);
            float* ap = (float*)acts.DataPointer;
            for (int c = 0; c < h; c++)
                for (int j = 0; j < tl; j++)
                {
                    float a = ip[(long)c * tl + j] + lp[(long)c * tl + j];
                    float b = ip[(long)(h + c) * tl + j] + lp[(long)(h + c) * tl + j];
                    ap[(long)c * tl + j] = MathF.Tanh(a) * (1f / (1f + MathF.Exp(-b)));
                }
            inl.Dispose();
            Tensor rs = new(new TensorShape(1, 2 * h, tl), DType.F32);
            backend.Conv1d(rs, acts, _rsW[i]!, _rsB[i], 1, 0, 0, 1, 1); acts.Dispose();
            float* rp = (float*)rs.DataPointer;
            for (int c = 0; c < h; c++)
                for (int j = 0; j < tl; j++)
                {
                    hp[(long)c * tl + j] += rp[(long)c * tl + j];
                    sa[(long)c * tl + j] += rp[(long)(h + c) * tl + j];
                }
            rs.Dispose();
        }
        local.Dispose(); hid.Dispose();

        float inv = 1f / MathF.Sqrt(_layers);
        for (long n = 0; n < (long)h * tl; n++) sa[n] *= inv;
        Tensor outT = new(new TensorShape(1, _latent, tl), DType.F32);
        backend.Conv1d(outT, skipAcc, _outW!, _outB, 1, 0, 0, 1, 1); skipAcc.Dispose();
        return outT;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] own = [_startW, _startB, _timeW, _timeB, _localW, _localB, _outW, _outB];
        foreach (Tensor? t in own) if (t is not null) yield return t;
        Tensor?[][] g = [_inW, _inB, _rsW, _rsB];
        foreach (Tensor?[] arr in g) foreach (Tensor? t in arr) if (t is not null) yield return t;
    }

    private static float[] SinTimeEmb(float t, int dim)
    {
        float[] e = new float[dim];
        int half = dim / 2;
        for (int i = 0; i < half; i++)
        {
            float freq = MathF.Exp(-(MathF.Log(10000f) * i / half));
            e[i] = MathF.Sin(t * freq * 1000f);
            e[half + i] = MathF.Cos(t * freq * 1000f);
        }
        return e;
    }

    private static Tensor InstanceNorm(Tensor x, int ch, int t)
    {
        Tensor o = new(new TensorShape(1, ch, t), DType.F32);
        float* xp = (float*)x.DataPointer; float* op = (float*)o.DataPointer;
        for (int c = 0; c < ch; c++)
        {
            double mean = 0; for (int j = 0; j < t; j++) mean += xp[(long)c * t + j]; mean /= t;
            double var = 0; for (int j = 0; j < t; j++) { double d = xp[(long)c * t + j] - mean; var += d * d; } var /= t;
            float invs = (float)(1.0 / Math.Sqrt(var + 1e-5));
            for (int j = 0; j < t; j++) op[(long)c * t + j] = (xp[(long)c * t + j] - (float)mean) * invs;
        }
        return o;
    }

    private static Tensor? Bias(IReadOnlyDictionary<string, Tensor> w, string key) => w.TryGetValue(key, out Tensor? b) ? WhisperOps.EnsureF32(b) : null;
}
