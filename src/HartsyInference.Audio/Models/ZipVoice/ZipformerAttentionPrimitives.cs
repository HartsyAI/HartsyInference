using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.ZipVoice;

/// <summary>"Compact" relative positional encoding (<c>CompactRelPositionalEncoding</c>): projects the interval
/// (-inf, inf) through a log-compression then atan() to a bounded range before a Fourier expansion, so that
/// large offsets differ only subtly. Builds a <c>[1, 2T-1, pos_dim]</c> table per stage per forward call (row
/// <c>i</c> = relative offset <c>i - (T-1)</c>), shared by every layer in that stage.</summary>
internal static unsafe class ZipformerRelPosEncoding
{
    public static Tensor Build(int t, int posDim)
    {
        int len = 2 * t - 1;
        int half = posDim / 2;
        double compressionLength = Math.Sqrt(posDim);
        double lengthScale = posDim / (2.0 * Math.PI);   // length_factor = 1.0

        Tensor pe = new(new TensorShape(1, len, posDim), DType.F32);
        float* p = (float*)pe.DataPointer;
        for (int i = 0; i < len; i++)
        {
            int offset = i - (t - 1);
            double sign = offset == 0 ? 0.0 : (offset > 0 ? 1.0 : -1.0);
            double xCompressed = compressionLength * sign * (Math.Log(Math.Abs(offset) + compressionLength) - Math.Log(compressionLength));
            double xAtan = Math.Atan(xCompressed / lengthScale);

            float* row = p + (long)i * posDim;
            for (int k = 0; k < half; k++)
            {
                double angle = xAtan * (k + 1);
                row[2 * k] = (float)Math.Cos(angle);
                row[2 * k + 1] = (float)Math.Sin(angle);
            }
            row[posDim - 1] = 1.0f;   // bias column, overwritten after the sin/cos fill
        }
        return pe;
    }
}

/// <summary>Computes shared relative-position multi-head attention WEIGHTS (softmax scores only, no value
/// mixing) — <c>RelPositionMultiheadAttentionWeights</c>. Reused verbatim by two independent
/// <see cref="ZipformerSelfAttentionValue"/> value-paths and (head 0 only) by
/// <see cref="ZipformerNonlinAttention"/> within one encoder layer. <c>in_proj</c> splits into
/// <c>[q | k | p]</c> along the last dim; content score is <c>q·k</c>, position score is
/// <c>p·linear_pos(pos_emb)</c> passed through the Transformer-XL rel_shift gather — the two are summed with
/// NO additional <c>1/sqrt(d)</c> division (that scaling was only ever baked into initial weight values, now
/// just whatever the checkpoint's <c>in_proj</c>/<c>linear_pos</c> weights already contain).</summary>
internal sealed unsafe class ZipformerAttentionWeights
{
    private readonly int _dim, _numHeads, _queryHeadDim, _posHeadDim, _posDim;
    private Tensor? _inProjW, _inProjB, _linearPosW;

    public ZipformerAttentionWeights(int dim, int numHeads, int queryHeadDim, int posHeadDim, int posDim)
    {
        _dim = dim; _numHeads = numHeads; _queryHeadDim = queryHeadDim; _posHeadDim = posHeadDim; _posDim = posDim;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _inProjW = WhisperOps.EnsureF32(w[$"{prefix}.in_proj.weight"]);
        _inProjB = WhisperOps.EnsureF32(w[$"{prefix}.in_proj.bias"]);
        _linearPosW = WhisperOps.EnsureF32(w[$"{prefix}.linear_pos.weight"]);
    }

    /// <summary><paramref name="x"/> is <c>[1, T, dim]</c>, <paramref name="posEmb"/> is <c>[1, 2T-1, pos_dim]</c>.
    /// Returns attention weights <c>[H, T, T]</c> (softmaxed over the last axis).</summary>
    public Tensor Forward(IBackend backend, Tensor x, Tensor posEmb, int t)
    {
        int h = _numHeads, qhd = _queryHeadDim, phd = _posHeadDim;
        int queryDim = qhd * h, posLen = 2 * t - 1;
        int projDim = queryDim * 2 + phd * h;

        Tensor proj = WhisperOps.ProjectLinear(backend, x, _inProjW!, _inProjB, 1, t, _dim, projDim);
        Tensor posProj = WhisperOps.ProjectLinear(backend, posEmb, _linearPosW!, null, 1, posLen, _posDim, phd * h);

        float* pp = (float*)proj.DataPointer;      // [T, projDim] (batch=1)
        float* ppos = (float*)posProj.DataPointer; // [posLen, phd*h]

        Tensor scores = new(new TensorShape(h, t, t), DType.F32);
        float* sp = (float*)scores.DataPointer;

        for (int head = 0; head < h; head++)
        {
            int qOff = head * qhd, kOff = queryDim + head * qhd, posEmbOff = head * phd;
            for (int i = 0; i < t; i++)
            {
                float* qi = pp + (long)i * projDim + qOff;
                float* pi = pp + (long)i * projDim + (2 * queryDim + head * phd);
                float* rowOut = sp + ((long)head * t + i) * t;
                for (int j = 0; j < t; j++)
                {
                    float* kj = pp + (long)j * projDim + kOff;
                    float ac = 0f;
                    for (int e = 0; e < qhd; e++) ac += qi[e] * kj[e];

                    int relIdx = (t - 1) - i + j;
                    float* posj = ppos + (long)relIdx * (phd * h) + posEmbOff;
                    float bd = 0f;
                    for (int e = 0; e < phd; e++) bd += pi[e] * posj[e];

                    rowOut[j] = ac + bd;
                }
            }
        }
        proj.Dispose();
        posProj.Dispose();

        // Softmax over the last axis, per (head, i) row.
        for (int head = 0; head < h; head++)
        {
            for (int i = 0; i < t; i++)
            {
                float* row = sp + ((long)head * t + i) * t;
                float maxS = float.NegativeInfinity;
                for (int j = 0; j < t; j++) if (row[j] > maxS) maxS = row[j];
                float sum = 0f;
                for (int j = 0; j < t; j++) { float e = MathF.Exp(row[j] - maxS); row[j] = e; sum += e; }
                float invSum = 1f / sum;
                for (int j = 0; j < t; j++) row[j] *= invSum;
            }
        }
        return scores;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_inProjW, _inProjB, _linearPosW];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }
}

/// <summary>Value-mixing attention (<c>SelfAttention</c> in the upstream source — renamed here to avoid
/// confusion with a "compute attention weights" module): projects <c>x</c> to per-head values, applies the
/// ALREADY-COMPUTED (shared) attention weights, and projects back. Two independent instances per Zipformer
/// layer (<c>self_attn1</c>, <c>self_attn2</c>) share one <see cref="ZipformerAttentionWeights"/> pattern but
/// each own a separate value projection.</summary>
internal sealed unsafe class ZipformerSelfAttentionValue
{
    private readonly int _dim, _numHeads, _valueHeadDim;
    private Tensor? _inProjW, _inProjB, _outProjW, _outProjB;

    public ZipformerSelfAttentionValue(int dim, int numHeads, int valueHeadDim)
    {
        _dim = dim; _numHeads = numHeads; _valueHeadDim = valueHeadDim;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _inProjW = WhisperOps.EnsureF32(w[$"{prefix}.in_proj.weight"]);
        _inProjB = WhisperOps.EnsureF32(w[$"{prefix}.in_proj.bias"]);
        _outProjW = WhisperOps.EnsureF32(w[$"{prefix}.out_proj.weight"]);
        _outProjB = WhisperOps.EnsureF32(w[$"{prefix}.out_proj.bias"]);
    }

    /// <summary><paramref name="x"/> is <c>[1, T, dim]</c>, <paramref name="attnWeights"/> is <c>[H, T, T]</c>.
    /// Returns <c>[1, T, dim]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor x, Tensor attnWeights, int t)
    {
        int h = _numHeads, vhd = _valueHeadDim, vDim = h * vhd;
        Tensor v = WhisperOps.ProjectLinear(backend, x, _inProjW!, _inProjB, 1, t, _dim, vDim);
        float* vp = (float*)v.DataPointer;
        float* awp = (float*)attnWeights.DataPointer;

        Tensor mixed = new(new TensorShape(1, t, vDim), DType.F32);
        float* mp = (float*)mixed.DataPointer;
        for (int head = 0; head < h; head++)
        {
            int vOff = head * vhd;
            for (int i = 0; i < t; i++)
            {
                float* rowW = awp + ((long)head * t + i) * t;
                float* outRow = mp + (long)i * vDim + vOff;
                for (int e = 0; e < vhd; e++) outRow[e] = 0f;
                for (int j = 0; j < t; j++)
                {
                    float wgt = rowW[j];
                    float* vj = vp + (long)j * vDim + vOff;
                    for (int e = 0; e < vhd; e++) outRow[e] += wgt * vj[e];
                }
            }
        }
        v.Dispose();

        Tensor output = WhisperOps.ProjectLinear(backend, mixed, _outProjW!, _outProjB, 1, t, vDim, _dim);
        mixed.Dispose();
        return output;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_inProjW, _inProjB, _outProjW, _outProjB];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }
}

/// <summary>NonlinAttention: a ConvolutionModule-like module that mixes across time using the shared attention
/// pattern (head 0 only) instead of a real convolution. <c>in_proj</c> splits 3 ways into <c>[s | x | y]</c>;
/// <c>s</c> is gated through tanh and multiplies <c>x</c> PER-POSITION first, the gated result is then mixed
/// across time via <c>attn_weights[0] @ (x*tanh(s))</c> (single head spanning the full <c>hidden_channels</c>
/// width), and finally gated again by <c>y</c> (per output position, not mixed) before the output
/// projection.</summary>
internal sealed unsafe class ZipformerNonlinAttention
{
    private readonly int _dim, _hidden;
    private Tensor? _inProjW, _inProjB, _outProjW, _outProjB;

    public ZipformerNonlinAttention(int dim)
    {
        _dim = dim;
        _hidden = 3 * dim / 4;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _inProjW = WhisperOps.EnsureF32(w[$"{prefix}.in_proj.weight"]);
        _inProjB = WhisperOps.EnsureF32(w[$"{prefix}.in_proj.bias"]);
        _outProjW = WhisperOps.EnsureF32(w[$"{prefix}.out_proj.weight"]);
        _outProjB = WhisperOps.EnsureF32(w[$"{prefix}.out_proj.bias"]);
    }

    /// <summary><paramref name="x"/> is <c>[1, T, dim]</c>; <paramref name="headZeroWeights"/> is head-0-only
    /// attention weights, <c>[1, T, T]</c> (<c>attn_weights[0:1]</c> from the shared
    /// <see cref="ZipformerAttentionWeights"/> output). Returns <c>[1, T, dim]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor x, Tensor headZeroWeights, int t)
    {
        int hidden = _hidden;
        Tensor proj = WhisperOps.ProjectLinear(backend, x, _inProjW!, _inProjB, 1, t, _dim, 3 * hidden);
        float* pp = (float*)proj.DataPointer;   // [T, 3*hidden] = [s | x | y] per row
        float* awp = (float*)headZeroWeights.DataPointer;   // [T, T] (single head)

        // Precompute the gated content branch once per position: gated[j,:] = x[j,:] * tanh(s[j,:]).
        Tensor gated = new(new TensorShape(1, t, hidden), DType.F32);
        float* gp = (float*)gated.DataPointer;
        for (int j = 0; j < t; j++)
        {
            float* row = pp + (long)j * 3 * hidden;
            float* gRow = gp + (long)j * hidden;
            for (int e = 0; e < hidden; e++) gRow[e] = row[hidden + e] * MathF.Tanh(row[e]);
        }

        Tensor mixed = new(new TensorShape(1, t, hidden), DType.F32);
        float* mp = (float*)mixed.DataPointer;
        for (int i = 0; i < t; i++)
        {
            float* rowW = awp + (long)i * t;
            float* outRow = mp + (long)i * hidden;
            for (int e = 0; e < hidden; e++) outRow[e] = 0f;
            for (int j = 0; j < t; j++)
            {
                float wgt = rowW[j];
                float* gj = gp + (long)j * hidden;
                for (int e = 0; e < hidden; e++) outRow[e] += wgt * gj[e];
            }
        }
        gated.Dispose();

        // Gate by y (third chunk) at position i (per-position, not mixed across time).
        for (int i = 0; i < t; i++)
        {
            float* yRow = pp + (long)i * 3 * hidden + 2 * hidden;
            float* outRow = mp + (long)i * hidden;
            for (int e = 0; e < hidden; e++) outRow[e] *= yRow[e];
        }
        proj.Dispose();

        Tensor output = WhisperOps.ProjectLinear(backend, mixed, _outProjW!, _outProjB, 1, t, hidden, _dim);
        mixed.Dispose();
        return output;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_inProjW, _inProjB, _outProjW, _outProjB];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }
}
