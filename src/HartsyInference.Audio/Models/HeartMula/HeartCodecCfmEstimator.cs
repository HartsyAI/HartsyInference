using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.HeartMula;

/// <summary>HeartCodec flow-matching velocity network (SQ-Codec style reconstruction head). A WaveNet-style
/// stack of dilated gated 1D convolutions over the current sample <c>x [1, dim, T]</c>; the RVQ codec latent
/// (<paramref name="mu"/>) is the local conditioning added per layer, and a sinusoidal flow-time embedding is
/// the global conditioning (the <c>ada_norm_single</c> modulation). Implements <see cref="ICfmEstimator"/> so
/// the reused OT-CFM Euler solver integrates it from noise to the codec feature. Keys: <c>{prefix}.start</c>,
/// <c>{prefix}.time_emb</c>, <c>{prefix}.cond</c>, <c>{prefix}.out</c>, <c>{prefix}.layers.{i}.{dilated,res_skip}</c>.</summary>
public sealed unsafe class HeartCodecCfmEstimator : ICfmEstimator
{
    private readonly int _dim, _hidden, _layers, _kernel, _cycle, _timeDim;
    private Tensor? _startW, _startB, _timeW, _timeB, _condW, _condB, _outW, _outB;
    private readonly Tensor?[] _inW, _inB, _rsW, _rsB;

    public HeartCodecCfmEstimator(int dim, int hidden = 0, int layers = 8, int kernel = 3, int dilationCycle = 4, int timeEmbDim = 0)
    {
        _dim = dim;
        _hidden = hidden > 0 ? hidden : dim;
        _layers = layers;
        _kernel = kernel;
        _cycle = dilationCycle;
        _timeDim = timeEmbDim > 0 ? timeEmbDim : dim;
        _inW = new Tensor?[_layers]; _inB = new Tensor?[_layers];
        _rsW = new Tensor?[_layers]; _rsB = new Tensor?[_layers];
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _startW = WhisperOps.EnsureF32(w[$"{prefix}.start.weight"]); _startB = Bias(w, $"{prefix}.start.bias");
        _timeW = WhisperOps.EnsureF32(w[$"{prefix}.time_emb.weight"]); _timeB = Bias(w, $"{prefix}.time_emb.bias");
        _condW = WhisperOps.EnsureF32(w[$"{prefix}.cond.weight"]); _condB = Bias(w, $"{prefix}.cond.bias");
        _outW = WhisperOps.EnsureF32(w[$"{prefix}.out.weight"]); _outB = Bias(w, $"{prefix}.out.bias");
        for (int i = 0; i < _layers; i++)
        {
            _inW[i] = WhisperOps.EnsureF32(w[$"{prefix}.layers.{i}.dilated.weight"]); _inB[i] = Bias(w, $"{prefix}.layers.{i}.dilated.bias");
            _rsW[i] = WhisperOps.EnsureF32(w[$"{prefix}.layers.{i}.res_skip.weight"]); _rsB[i] = Bias(w, $"{prefix}.layers.{i}.res_skip.bias");
        }
    }

    /// <summary>Velocity estimate. <paramref name="x"/> is the current sample <c>[1, dim, T]</c>;
    /// <paramref name="mu"/> the RVQ codec latent conditioning <c>[1, dim, T]</c>; <paramref name="t"/> the
    /// flow time. <paramref name="spk"/>/<paramref name="cond"/> are unused (the solver zeroes them for CFG).</summary>
    public Tensor Estimate(IBackend backend, Tensor x, Tensor mu, float t, Tensor spk, Tensor cond)
    {
        if (_startW is null) throw new InvalidOperationException("HeartCodecCfmEstimator weights not loaded.");
        int h = _hidden, tl = (int)x.Shape[2];
        Tensor hid = new(new TensorShape(1, h, tl), DType.F32);
        backend.Conv1d(hid, x, _startW!, _startB, 1, 0, 0, 1, 1);

        // Global cond (ada_norm_single): sinusoidal time emb → Linear → [h], broadcast-added.
        float[] te = SinTimeEmb(t, _timeDim);
        Tensor teT = new(new TensorShape(1, 1, _timeDim), DType.F32);
        te.AsSpan().CopyTo(new Span<float>((void*)teT.DataPointer, _timeDim));
        Tensor teProj = WhisperOps.ProjectLinear(backend, teT, _timeW!, _timeB, 1, 1, _timeDim, h); teT.Dispose();
        float* hp = (float*)hid.DataPointer; float* gp = (float*)teProj.DataPointer;
        for (int c = 0; c < h; c++) for (int j = 0; j < tl; j++) hp[(long)c * tl + j] += gp[c];
        teProj.Dispose();

        // Local cond: codec latent mu → 1×1 conv → [2h, T].
        Tensor local = new(new TensorShape(1, 2 * h, tl), DType.F32);
        backend.Conv1d(local, mu, _condW!, _condB, 1, 0, 0, 1, 1);
        float* lp = (float*)local.DataPointer;

        Tensor skipAcc = new(new TensorShape(1, h, tl), DType.F32);
        float* sa = (float*)skipAcc.DataPointer;
        for (long m = 0; m < (long)h * tl; m++) sa[m] = 0f;

        for (int i = 0; i < _layers; i++)
        {
            int dilation = 1 << (i % _cycle);
            int pad = dilation * (_kernel - 1);     // causal: left-only pad
            Tensor inl = new(new TensorShape(1, 2 * h, tl), DType.F32);
            backend.Conv1d(inl, hid, _inW[i]!, _inB[i], 1, pad, 0, dilation, 1);
            float* ip = (float*)inl.DataPointer;
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
        for (long m = 0; m < (long)h * tl; m++) sa[m] *= inv;
        Tensor outT = new(new TensorShape(1, _dim, tl), DType.F32);
        backend.Conv1d(outT, skipAcc, _outW!, _outB, 1, 0, 0, 1, 1); skipAcc.Dispose();
        return outT;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] own = [_startW, _startB, _timeW, _timeB, _condW, _condB, _outW, _outB];
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

    private static Tensor? Bias(IReadOnlyDictionary<string, Tensor> w, string key) =>
        w.TryGetValue(key, out Tensor? b) ? WhisperOps.EnsureF32(b) : null;
}
