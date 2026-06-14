using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.StyleTts2;

/// <summary>The KDiffusion (EDM) denoiser <c>D(x, σ)</c> for StyleTTS 2's style sampler. Applies the
/// Karras preconditioning around the <c>StyleTransformer1d</c> network and classifier-free guidance:
///
/// <code>
///   c_in   = 1/√(σ² + σ_data²)
///   c_out  = σ·σ_data/√(σ² + σ_data²)
///   c_skip = σ_data²/(σ² + σ_data²)
///   c_noise= ln(σ)/4
///   F      = (1−scale)·net(c_in·x, c_noise, ∅)   +   scale·net(c_in·x, c_noise, textEmb, refStyle)
///   D(x,σ) = c_skip·x + c_out·F
/// </code>
///
/// <para>The preconditioning + CFG combine are exact (unit-tested via a stub network); the
/// <c>StyleTransformer1d</c> network forward is a checkpoint-gated scaffold (3-layer 1-D transformer over
/// the length-1 256-d latent with timestep embedding + cross-attention to the PLBERT features).</para></summary>
public sealed unsafe class StyleDenoiser : IStyleDenoiser
{
    private readonly StyleTts2Config _cfg;
    private readonly int _dim;
    private Tensor? _textEmb;          // [1, seq, 768] PLBERT features (conditioning)
    private Tensor? _refStyle;         // [1, 256] reference style (optional)

    // StyleTransformer1d scaffold weights.
    private Tensor? _timeW1, _timeB1, _timeW2, _timeB2;
    private Tensor? _inProjW, _inProjB, _outProjW, _outProjB;
    private Tensor? _crossKvW, _crossKvB;     // text(768) → 2·dim for a single cross-attn read

    public StyleDenoiser(StyleTts2Config cfg)
    {
        _cfg = cfg;
        _dim = cfg.StyleDim;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "diffusion")
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        _timeW1 = TryGet(w, $"{p}to_time.0.weight"); _timeB1 = TryGet(w, $"{p}to_time.0.bias");
        _timeW2 = TryGet(w, $"{p}to_time.2.weight"); _timeB2 = TryGet(w, $"{p}to_time.2.bias");
        _inProjW = TryGet(w, $"{p}to_in.weight"); _inProjB = TryGet(w, $"{p}to_in.bias");
        _outProjW = TryGet(w, $"{p}to_out.weight"); _outProjB = TryGet(w, $"{p}to_out.bias");
        _crossKvW = TryGet(w, $"{p}cross.to_kv.weight"); _crossKvB = TryGet(w, $"{p}cross.to_kv.bias");
    }

    private static Tensor? TryGet(IReadOnlyDictionary<string, Tensor> w, string key)
        => w.TryGetValue(key, out Tensor? t) ? WhisperOps.EnsureF32(t) : null;

    /// <summary>Sets the per-call conditioning (PLBERT text features + optional reference style).</summary>
    public void Prepare(Tensor textEmb, Tensor? refStyle)
    {
        _textEmb = textEmb;
        _refStyle = refStyle;
    }

    public Tensor Denoise(IBackend backend, Tensor x, float sigma)
    {
        float sd = _cfg.SigmaData;
        float denom = MathF.Sqrt(sigma * sigma + sd * sd);
        float cIn = 1f / denom;
        float cOut = sigma * sd / denom;
        float cSkip = sd * sd / (sigma * sigma + sd * sd);
        float cNoise = sigma > 0f ? MathF.Log(sigma) * 0.25f : 0f;

        // scaled input.
        Tensor scaled = new(x.Shape, DType.F32);
        float* xp = (float*)x.DataPointer;
        float* scp = (float*)scaled.DataPointer;
        for (long i = 0; i < x.ElementCount; i++) scp[i] = cIn * xp[i];

        Tensor condF = Network(backend, scaled, cNoise, conditioned: true);
        Tensor f;
        if (MathF.Abs(_cfg.EmbeddingScale - 1f) > 1e-6f)
        {
            Tensor uncondF = Network(backend, scaled, cNoise, conditioned: false);
            f = new(x.Shape, DType.F32);
            float* cp = (float*)condF.DataPointer;
            float* up = (float*)uncondF.DataPointer;
            float* fp = (float*)f.DataPointer;
            float s = _cfg.EmbeddingScale;
            for (long i = 0; i < f.ElementCount; i++) fp[i] = (1f - s) * up[i] + s * cp[i];
            uncondF.Dispose();
            condF.Dispose();
        }
        else
        {
            f = condF;
        }
        scaled.Dispose();

        // D = c_skip·x + c_out·F.
        Tensor outT = new(x.Shape, DType.F32);
        float* op = (float*)outT.DataPointer;
        float* ffp = (float*)f.DataPointer;
        for (long i = 0; i < outT.ElementCount; i++) op[i] = cSkip * xp[i] + cOut * ffp[i];
        f.Dispose();
        return outT;
    }

    /// <summary>StyleTransformer1d network (scaffold): timestep embedding → input projection →
    /// (when conditioned) a mean-pooled cross-attention read of the PLBERT text features → output
    /// projection. The full 3-layer self/cross-attention stack is checkpoint-gated; this preserves the
    /// I/O contract and the conditioning dependency so the EDM/CFG wrapper + sampler are exercisable.</summary>
    private Tensor Network(IBackend backend, Tensor scaledX, float cNoise, bool conditioned)
    {
        // If weights aren't present (pre-checkpoint), fall back to an identity-shaped zero velocity so
        // the sampler still runs deterministically.
        if (_inProjW is null)
            return new Tensor(scaledX.Shape, DType.F32);

        Tensor x3 = scaledX.Reshape(new TensorShape(1, 1, _dim));
        Tensor h = WhisperOps.ProjectLinear(backend, x3, _inProjW!, _inProjB, 1, 1, _dim, _dim);

        // Add the timestep embedding.
        Tensor tEmb = TimeEmbedding(backend, cNoise);
        float* hp = (float*)h.DataPointer;
        float* tp = (float*)tEmb.DataPointer;
        for (int i = 0; i < _dim; i++) hp[i] += tp[i];
        tEmb.Dispose();

        // Conditioning: add a mean-pooled cross read of the text features (and reference style).
        if (conditioned && _textEmb is not null && _crossKvW is not null)
        {
            float[] ctx = MeanPoolText(backend, _textEmb);
            for (int i = 0; i < _dim; i++) hp[i] += ctx[i];
        }

        Tensor outT = WhisperOps.ProjectLinear(backend, h, _outProjW!, _outProjB, 1, 1, _dim, _dim);
        h.Dispose();
        return outT.Reshape(scaledX.Shape);
    }

    private Tensor TimeEmbedding(IBackend backend, float cNoise)
    {
        int half = _dim / 2;
        Tensor sin = new(new TensorShape(1, 1, _dim), DType.F32);
        float* sp = (float*)sin.DataPointer;
        double logBase = Math.Log(10000.0) / Math.Max(1, half - 1);
        for (int i = 0; i < half; i++)
        {
            double freq = Math.Exp(-logBase * i);
            sp[i] = (float)Math.Sin(cNoise * freq);
            sp[half + i] = (float)Math.Cos(cNoise * freq);
        }
        if (_timeW1 is null) return sin;
        Tensor h1 = WhisperOps.ProjectLinear(backend, sin, _timeW1!, _timeB1, 1, 1, _dim, _dim);
        sin.Dispose();
        backend.Silu(h1, h1);
        Tensor h2 = WhisperOps.ProjectLinear(backend, h1, _timeW2!, _timeB2, 1, 1, _dim, _dim);
        h1.Dispose();
        return h2;
    }

    private float[] MeanPoolText(IBackend backend, Tensor textEmb)
    {
        int seq = (int)textEmb.Shape[1];
        int dim = (int)textEmb.Shape[2];
        float[] pooled = new float[dim];
        float* tp = (float*)textEmb.DataPointer;
        for (int s = 0; s < seq; s++)
            for (int d = 0; d < dim; d++)
                pooled[d] += tp[(long)s * dim + d];
        for (int d = 0; d < dim; d++) pooled[d] /= seq;

        // Project text dim → style dim via the cross to_kv weight (use the first `dim` rows as a read).
        Tensor pooledT = new(new TensorShape(1, 1, dim), DType.F32);
        float* pp = (float*)pooledT.DataPointer;
        for (int d = 0; d < dim; d++) pp[d] = pooled[d];
        int kvOut = (int)_crossKvW!.Shape[0];
        Tensor proj = WhisperOps.ProjectLinear(backend, pooledT, _crossKvW!, _crossKvB, 1, 1, dim, kvOut);
        pooledT.Dispose();
        float[] result = new float[_dim];
        float* prp = (float*)proj.DataPointer;
        for (int d = 0; d < _dim && d < kvOut; d++) result[d] = prp[d];
        proj.Dispose();
        return result;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_timeW1, _timeB1, _timeW2, _timeB2, _inProjW, _inProjB, _outProjW, _outProjB, _crossKvW, _crossKvB];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }
}
