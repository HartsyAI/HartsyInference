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
    /// Returns attention weights <c>[H, T, T]</c> (softmaxed over the last axis).
    ///
    /// <para>Content score (<c>q·k</c>, the dominant <c>O(H·T²·qhd)</c> cost) runs on-device via
    /// <see cref="IBackend.BatchedMatMul"/> — previously a quadruple-nested host scalar loop, the single
    /// biggest cost in the whole Zipformer backbone (this module's output is shared by 3 consumers per layer:
    /// two <see cref="ZipformerSelfAttentionValue"/> instances plus <see cref="ZipformerNonlinAttention"/>).
    /// The position-score dot products (<c>p·linear_pos(pos_emb)</c>) also run on-device against all
    /// <c>2T-1</c> relative offsets at once; only the Transformer-XL rel_shift band remap (a per-row memcpy,
    /// no arithmetic) and the softmax (operates on the already-small <c>[H,T,T]</c> result) stay host —
    /// rel-shift-gather/softmax GPU primitives are documented follow-ups, not the priority lever here (see
    /// docs/Checklists/PHASE_5_AUDIO.md ZipVoice perf-pass notes).</para></summary>
    public Tensor Forward(IBackend backend, Tensor x, Tensor posEmb, int t)
    {
        int h = _numHeads, qhd = _queryHeadDim, phd = _posHeadDim;
        int queryDim = qhd * h, posLen = 2 * t - 1;
        int projDim = queryDim * 2 + phd * h;

        Tensor proj = WhisperOps.ProjectLinear(backend, x, _inProjW!, _inProjB, 1, t, _dim, projDim);
        Tensor posProj = WhisperOps.ProjectLinear(backend, posEmb, _linearPosW!, null, 1, posLen, _posDim, phd * h);

        // Content score: q·k over all heads at once via a batched GEMM instead of a T×T×qhd scalar loop.
        Tensor qFlat = new(new TensorShape(1, t, queryDim), DType.F32);
        Tensor kFlat = new(new TensorShape(1, t, queryDim), DType.F32);
        backend.SliceLastDim(qFlat, proj, 0);
        backend.SliceLastDim(kFlat, proj, queryDim);
        Tensor qHeads = new(new TensorShape(h, t, qhd), DType.F32);
        Tensor kHeads = new(new TensorShape(h, t, qhd), DType.F32);
        backend.Permute0213(qHeads, qFlat, t, h, qhd);
        backend.Permute0213(kHeads, kFlat, t, h, qhd);
        qFlat.Dispose(); kFlat.Dispose();
        Tensor kHeadsT = new(new TensorShape(h, qhd, t), DType.F32);
        backend.Transpose2D(kHeadsT, kHeads, t, qhd);
        kHeads.Dispose();
        Tensor contentScores = new(new TensorShape(h, t, t), DType.F32);
        backend.BatchedMatMul(contentScores, qHeads, kHeadsT);
        qHeads.Dispose(); kHeadsT.Dispose();

        // Position score: q_pos·linear_pos(pos_emb), the Transformer-XL rel_shift pattern. The dot products
        // (O(H·T·posLen·phd)) run on-device via BatchedMatMul against ALL posLen relative offsets at once;
        // only the diagonal-band index remap (relIdx = (t-1)-i+j is a contiguous run of length t starting at
        // (t-1-i) for each row i — no multiply-adds, just a per-row memcpy) stays host.
        Tensor pFlat = new(new TensorShape(1, t, phd * h), DType.F32);
        backend.SliceLastDim(pFlat, proj, 2 * queryDim);
        Tensor pHeads = new(new TensorShape(h, t, phd), DType.F32);
        backend.Permute0213(pHeads, pFlat, t, h, phd);
        pFlat.Dispose();
        Tensor posProjHeads = new(new TensorShape(h, posLen, phd), DType.F32);
        backend.Permute0213(posProjHeads, posProj, posLen, h, phd);
        Tensor posProjHeadsT = new(new TensorShape(h, phd, posLen), DType.F32);
        backend.Transpose2D(posProjHeadsT, posProjHeads, posLen, phd);
        posProjHeads.Dispose();
        Tensor rawPosScores = new(new TensorShape(h, t, posLen), DType.F32);
        backend.BatchedMatMul(rawPosScores, pHeads, posProjHeadsT);
        pHeads.Dispose(); posProjHeadsT.Dispose();
        proj.Dispose();
        posProj.Dispose();

        Tensor posScores = new(new TensorShape(h, t, t), DType.F32);
        float* rawp = (float*)rawPosScores.DataPointer;
        float* psp = (float*)posScores.DataPointer;
        for (int head = 0; head < h; head++)
        {
            for (int i = 0; i < t; i++)
            {
                float* srcRow = rawp + ((long)head * t + i) * posLen + (t - 1 - i);
                float* dstRow = psp + ((long)head * t + i) * t;
                Buffer.MemoryCopy(srcRow, dstRow, t * sizeof(float), t * sizeof(float));
            }
        }
        rawPosScores.Dispose();

        Tensor scores = new(new TensorShape(h, t, t), DType.F32);
        backend.Add(scores, contentScores, posScores);
        contentScores.Dispose();
        posScores.Dispose();

        // Softmax over the last axis, per (head, i) row.
        float* sp = (float*)scores.DataPointer;
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
    /// Returns <c>[1, T, dim]</c>. Value-mixing (<c>attnWeights @ V</c>, the second-biggest <c>O(H·T²·vhd)</c>
    /// cost after the attention-weights content score) runs via <see cref="IBackend.BatchedMatMul"/> instead of
    /// a host scalar loop.</summary>
    public Tensor Forward(IBackend backend, Tensor x, Tensor attnWeights, int t)
    {
        int h = _numHeads, vhd = _valueHeadDim, vDim = h * vhd;
        Tensor vFlat = WhisperOps.ProjectLinear(backend, x, _inProjW!, _inProjB, 1, t, _dim, vDim);

        Tensor vHeads = new(new TensorShape(h, t, vhd), DType.F32);
        backend.Permute0213(vHeads, vFlat, t, h, vhd);
        vFlat.Dispose();

        Tensor mixedHeads = new(new TensorShape(h, t, vhd), DType.F32);
        backend.BatchedMatMul(mixedHeads, attnWeights, vHeads);
        vHeads.Dispose();

        Tensor mixed = new(new TensorShape(1, t, vDim), DType.F32);
        backend.Permute0213(mixed, mixedHeads, h, t, vhd);
        mixedHeads.Dispose();

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
    /// <see cref="ZipformerAttentionWeights"/> output). Returns <c>[1, T, dim]</c>. Fully GPU-resident: the
    /// <c>[s | x | y]</c> chunk split is <see cref="IBackend.SliceLastDim"/>, the content gate is
    /// <see cref="IBackend.Tanh"/> + <see cref="IBackend.Mul"/>, the time-mixing is a batch-1
    /// <see cref="IBackend.BatchedMatMul"/> (<c>[1,T,T] @ [1,T,hidden]</c>), and the output gate is another
    /// <see cref="IBackend.Mul"/> — replaces three host <c>float*</c> loops (gate, O(T²·hidden) mix, y-gate).</summary>
    public Tensor Forward(IBackend backend, Tensor x, Tensor headZeroWeights, int t)
    {
        int hidden = _hidden;
        Tensor proj = WhisperOps.ProjectLinear(backend, x, _inProjW!, _inProjB, 1, t, _dim, 3 * hidden);

        Tensor sChunk = new(new TensorShape(1, t, hidden), DType.F32);
        backend.SliceLastDim(sChunk, proj, 0);
        Tensor xChunk = new(new TensorShape(1, t, hidden), DType.F32);
        backend.SliceLastDim(xChunk, proj, hidden);
        Tensor yChunk = new(new TensorShape(1, t, hidden), DType.F32);
        backend.SliceLastDim(yChunk, proj, 2 * hidden);
        proj.Dispose();

        backend.Tanh(sChunk, sChunk);
        Tensor gated = new(new TensorShape(1, t, hidden), DType.F32);
        backend.Mul(gated, xChunk, sChunk);
        sChunk.Dispose();
        xChunk.Dispose();

        Tensor mixed = new(new TensorShape(1, t, hidden), DType.F32);
        backend.BatchedMatMul(mixed, headZeroWeights, gated);
        gated.Dispose();

        backend.Mul(mixed, mixed, yChunk);
        yChunk.Dispose();

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
