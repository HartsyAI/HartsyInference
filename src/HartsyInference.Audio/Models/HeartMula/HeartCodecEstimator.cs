using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.HeartMula;

/// <summary>HeartCodec flow-matching velocity estimator — the upstream <c>LlamaTransformer</c>
/// (heartcodec/models/transformer.py). A two-stage adaLN-single DiT that predicts the CFM velocity over the
/// codec latent:
/// <list type="number">
///   <item><b>proj_in</b> (<see cref="ProjectLayer"/>): in_channels(1024) → inner(1536).</item>
///   <item><b>24 blocks @1536</b> (24 heads, head_dim 64), each adaLN-single modulated by the stage-1
///   timestep embedding.</item>
///   <item><b>norm_out</b> (affine-free LN) + <c>scale_shift_table[2,1536]</c> + embedded_timestep.</item>
///   <item><b>connection_proj</b>: cat(original input, stage-1 out) [1024+1536=2560] → inner_2(3072).</item>
///   <item><b>6 blocks @3072</b> (24 heads, head_dim 128), adaLN-single from a second timestep embedding.</item>
///   <item><b>norm_out_2</b> + <c>scale_shift_table_2[2,3072]</c> + proj_out → out_channels(256).</item>
/// </list>
/// RMSNorm (eps 1e-6), interleaved (GPT-J) RoPE on the full head_dim, SwiGLU MLP, full bidirectional
/// attention (no causal mask), scale 1/√head_dim.</summary>
public sealed unsafe class HeartCodecEstimator
{
    private readonly int _inCh, _outCh, _inner, _inner2, _heads, _headDim, _headDim2, _nLayers, _nLayers2;

    private readonly ProjectLayer _projIn, _connProj, _projOut;
    private readonly AdaLnSingleFlow _adaln, _adaln2;
    private readonly Block[] _blocks, _blocks2;
    private Tensor? _ssTable, _ssTable2;   // [2, inner] / [2, inner2]

    public HeartCodecEstimator(int inChannels = 1024, int outChannels = 256, int numHeads = 24,
        int headDim = 64, int numLayers = 24, int numLayers2 = 6)
    {
        _inCh = inChannels;
        _outCh = outChannels;
        _heads = numHeads;
        _headDim = headDim;
        _headDim2 = headDim * 2;
        _inner = numHeads * headDim;          // 1536
        _inner2 = _inner * 2;                 // 3072
        _nLayers = numLayers;
        _nLayers2 = numLayers2;

        _projIn = new ProjectLayer(_inCh, _inner);
        _connProj = new ProjectLayer(_inCh + _inner, _inner2);
        _projOut = new ProjectLayer(_inner2, _outCh);
        _adaln = new AdaLnSingleFlow(_inner);
        _adaln2 = new AdaLnSingleFlow(_inner2);
        _blocks = new Block[_nLayers];
        _blocks2 = new Block[_nLayers2];
        for (int i = 0; i < _nLayers; i++) _blocks[i] = new Block(_inner, _heads, _headDim);
        for (int i = 0; i < _nLayers2; i++) _blocks2[i] = new Block(_inner2, _heads, _headDim2);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _projIn.Load(w, $"{prefix}.proj_in");
        _connProj.Load(w, $"{prefix}.connection_proj");
        _projOut.Load(w, $"{prefix}.proj_out");
        _adaln.Load(w, $"{prefix}.adaln_single");
        _adaln2.Load(w, $"{prefix}.adaln_single_2");
        _ssTable = WhisperOps.EnsureF32(w[$"{prefix}.scale_shift_table"]);
        _ssTable2 = WhisperOps.EnsureF32(w[$"{prefix}.scale_shift_table_2"]);
        for (int i = 0; i < _nLayers; i++) _blocks[i].Load(w, $"{prefix}.transformer_blocks.{i}");
        for (int i = 0; i < _nLayers2; i++) _blocks2[i].Load(w, $"{prefix}.transformer_blocks_2.{i}");
    }

    /// <summary>Velocity estimate. <paramref name="input"/> is the concatenated estimator input
    /// <c>[B, T, in_channels]</c> = cat(x, incontext, mu). <paramref name="timestep"/> is the flow time per
    /// batch (length B). Returns <c>[B, T, out_channels]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor input, float[] timestep)
    {
        int b = (int)input.Shape[0];
        int t = (int)input.Shape[1];

        // Stage 1.
        Tensor s = _projIn.Forward(backend, input, b, t);                            // [B,T,inner]
        (Tensor mod1, Tensor emb1) = _adaln.Forward(backend, timestep, b);           // mod1 [B,6*inner], emb1 [B,inner]
        // Block.Forward mutates its input in place and RETURNS THE SAME tensor — do NOT dispose between
        // iterations (that frees the live buffer and the next block reads freed memory → heap corruption).
        for (int i = 0; i < _nLayers; i++)
            s = _blocks[i].Forward(backend, s, mod1, b, t);
        mod1.Dispose();

        // norm_out + scale_shift_table + emb1.
        ApplyFinalNorm(s, _ssTable!, emb1, b, t, _inner);
        emb1.Dispose();
        // connection_proj over cat(input, s).
        Tensor cat = new(new TensorShape(b, t, _inCh + _inner), DType.F32);
        float* ip = (float*)input.DataPointer; float* sp = (float*)s.DataPointer; float* cp = (float*)cat.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (int ti = 0; ti < t; ti++)
            {
                long dst = ((long)bi * t + ti) * (_inCh + _inner);
                long srcI = ((long)bi * t + ti) * _inCh;
                long srcS = ((long)bi * t + ti) * _inner;
                for (int c = 0; c < _inCh; c++) cp[dst + c] = ip[srcI + c];
                for (int c = 0; c < _inner; c++) cp[dst + _inCh + c] = sp[srcS + c];
            }
        s.Dispose();
        Tensor x = _connProj.Forward(backend, cat, b, t);                            // [B,T,inner2]
        cat.Dispose();

        // Stage 2.
        (Tensor mod2, Tensor emb2) = _adaln2.Forward(backend, timestep, b);
        for (int i = 0; i < _nLayers2; i++)
            x = _blocks2[i].Forward(backend, x, mod2, b, t);
        mod2.Dispose();        ApplyFinalNorm(x, _ssTable2!, emb2, b, t, _inner2);
        emb2.Dispose();

        Tensor outp = _projOut.Forward(backend, x, b, t);                            // [B,T,out_channels]
        x.Dispose();
        return outp;
    }

    // norm_out (affine-free LN, eps 1e-6) then x * (1 + scale) + shift, where
    // [shift, scale] = scale_shift_table[None] + embedded_timestep[:,None] (chunk 2 over dim 1).
    private static void ApplyFinalNorm(Tensor x, Tensor ssTable, Tensor emb, int b, int t, int dim)
    {
        float* xp = (float*)x.DataPointer;
        float* tab = (float*)ssTable.DataPointer;   // [2, dim]: row0=shift, row1=scale
        float* ep = (float*)emb.DataPointer;        // [b, dim]
        const float eps = 1e-6f;
        for (int bi = 0; bi < b; bi++)
        {
            for (int ti = 0; ti < t; ti++)
            {
                long off = ((long)bi * t + ti) * dim;
                double mean = 0;
                for (int c = 0; c < dim; c++) mean += xp[off + c];
                mean /= dim;
                double var = 0;
                for (int c = 0; c < dim; c++) { double d = xp[off + c] - mean; var += d * d; }
                var /= dim;
                float inv = 1f / MathF.Sqrt((float)var + eps);
                for (int c = 0; c < dim; c++)
                {
                    float n = ((float)(xp[off + c] - mean)) * inv;
                    float shift = tab[c] + ep[(long)bi * dim + c];
                    float scale = tab[dim + c] + ep[(long)bi * dim + c];
                    xp[off + c] = n * (1f + scale) + shift;
                }
            }
        }
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor tt in _projIn.Weights()) yield return tt;
        foreach (Tensor tt in _connProj.Weights()) yield return tt;
        foreach (Tensor tt in _projOut.Weights()) yield return tt;
        foreach (Tensor tt in _adaln.Weights()) yield return tt;
        foreach (Tensor tt in _adaln2.Weights()) yield return tt;
        if (_ssTable is not null) yield return _ssTable;
        if (_ssTable2 is not null) yield return _ssTable2;
        foreach (Block bl in _blocks) foreach (Tensor tt in bl.Weights()) yield return tt;
        foreach (Block bl in _blocks2) foreach (Tensor tt in bl.Weights()) yield return tt;
    }

    // ── ProjectLayer: Conv1d(k=3, pad=1) over [B,C,T] then * 3^-0.5 then Linear ────────────────────────
    private sealed class ProjectLayer
    {
        private readonly int _in, _filter;
        private Tensor? _convW, _convB, _linW, _linB;
        private static readonly float KScale = 1f / MathF.Sqrt(3f);

        public ProjectLayer(int inDim, int filterDim) { _in = inDim; _filter = filterDim; }

        public void Load(IReadOnlyDictionary<string, Tensor> w, string p)
        {
            _convW = WhisperOps.EnsureF32(w[$"{p}.ffn_1.weight"]);   // [filter, in, 3]
            _convB = WhisperOps.EnsureF32(w[$"{p}.ffn_1.bias"]);
            _linW = WhisperOps.EnsureF32(w[$"{p}.ffn_2.weight"]);    // [filter, filter]
            _linB = WhisperOps.EnsureF32(w[$"{p}.ffn_2.bias"]);
        }

        public Tensor Forward(IBackend backend, Tensor x, int b, int t)
        {
            // x [B,T,in] → transpose to [B,in,T] → conv1d k3 pad1 → [B,filter,T] → transpose → scale → Linear.
            Tensor xc = new(new TensorShape(b, _in, t), DType.F32);
            TransposeBTCtoBCT(x, xc, b, t, _in);
            Tensor conv = new(new TensorShape(b, _filter, t), DType.F32);
            backend.Conv1d(conv, xc, _convW!, _convB, 1, 1, 1, 1, 1);
            xc.Dispose();
            Tensor convBT = new(new TensorShape(b, t, _filter), DType.F32);
            Transpose_BCT_to_BTC(conv, convBT, b, t, _filter);
            conv.Dispose();
            backend.Scale(convBT, convBT, KScale);
            Tensor outp = WhisperOps.ProjectLinear(backend, convBT, _linW!, _linB, b, t, _filter, _filter);
            convBT.Dispose();
            return outp;
        }

        public IEnumerable<Tensor> Weights()
        {
            if (_convW is not null) yield return _convW;
            if (_convB is not null) yield return _convB;
            if (_linW is not null) yield return _linW;
            if (_linB is not null) yield return _linB;
        }

        private static unsafe void TransposeBTCtoBCT(Tensor src, Tensor dst, int b, int t, int c)
        {
            float* s = (float*)src.DataPointer; float* d = (float*)dst.DataPointer;
            for (int bi = 0; bi < b; bi++)
                for (int ti = 0; ti < t; ti++)
                    for (int ci = 0; ci < c; ci++)
                        d[((long)bi * c + ci) * t + ti] = s[((long)bi * t + ti) * c + ci];
        }

        private static unsafe void Transpose_BCT_to_BTC(Tensor src, Tensor dst, int b, int t, int c)
        {
            float* s = (float*)src.DataPointer; float* d = (float*)dst.DataPointer;
            for (int bi = 0; bi < b; bi++)
                for (int ci = 0; ci < c; ci++)
                    for (int ti = 0; ti < t; ti++)
                        d[((long)bi * t + ti) * c + ci] = s[((long)bi * c + ci) * t + ti];
        }
    }

    // ── AdaLayerNormSingleFlow: sinusoidal flow-time emb (512, scale 1000, cos-then-sin) → TimestepEmbedding
    //    (Linear 512→D, SiLU, Linear D→D) = embedded_timestep; then linear(silu(embedded)) → [B,6D]. ────
    private sealed class AdaLnSingleFlow
    {
        private const int FlowTSize = 512;
        private readonly int _dim;
        private Tensor? _te1W, _te1B, _te2W, _te2B, _linW, _linB;

        public AdaLnSingleFlow(int dim) { _dim = dim; }

        public void Load(IReadOnlyDictionary<string, Tensor> w, string p)
        {
            _te1W = WhisperOps.EnsureF32(w[$"{p}.emb.timestep_embedder.linear_1.weight"]); // [dim, 512]
            _te1B = WhisperOps.EnsureF32(w[$"{p}.emb.timestep_embedder.linear_1.bias"]);
            _te2W = WhisperOps.EnsureF32(w[$"{p}.emb.timestep_embedder.linear_2.weight"]); // [dim, dim]
            _te2B = WhisperOps.EnsureF32(w[$"{p}.emb.timestep_embedder.linear_2.bias"]);
            _linW = WhisperOps.EnsureF32(w[$"{p}.linear.weight"]);                          // [6*dim, dim]
            _linB = WhisperOps.EnsureF32(w[$"{p}.linear.bias"]);
        }

        public (Tensor mod, Tensor embedded) Forward(IBackend backend, float[] timestep, int b)
        {
            // sinusoidal flow-time projection [B, 512].
            Tensor proj = new(new TensorShape(b, 1, FlowTSize), DType.F32);
            float* pp = (float*)proj.DataPointer;
            int half = FlowTSize / 2;
            for (int bi = 0; bi < b; bi++)
            {
                float ts = timestep[bi];
                for (int i = 0; i < half; i++)
                {
                    float freq = MathF.Exp(-MathF.Log(10000f) * i / half);
                    float arg = ts * freq * 1000f;
                    pp[(long)bi * FlowTSize + i] = MathF.Cos(arg);
                    pp[(long)bi * FlowTSize + half + i] = MathF.Sin(arg);
                }
            }
            // TimestepEmbedding: Linear → SiLU → Linear → embedded_timestep [B, dim].
            Tensor h1 = WhisperOps.ProjectLinear(backend, proj, _te1W!, _te1B, b, 1, FlowTSize, _dim);
            proj.Dispose();
            backend.Silu(h1, h1);
            Tensor embedded = WhisperOps.ProjectLinear(backend, h1, _te2W!, _te2B, b, 1, _dim, _dim);
            h1.Dispose();
            // reshape embedded [B,1,dim] → [B,dim].
            Tensor emb2d = new(new TensorShape(b, _dim), DType.F32);
            Buffer.MemoryCopy((void*)embedded.DataPointer, (void*)emb2d.DataPointer, (long)b * _dim * 4, (long)b * _dim * 4);
            // linear(silu(embedded)) → [B,1,6*dim]. Keep the SiLU output rank-3 ([b,1,dim]) — ProjectLinear
            // heap-corrupts on a rank-2 input (missing dim=0). Silu from the still-rank-3 `embedded`.
            Tensor sil = new(new TensorShape(b, 1, _dim), DType.F32);
            backend.Silu(sil, embedded);
            embedded.Dispose();
            Tensor mod = WhisperOps.ProjectLinear(backend, sil, _linW!, _linB, b, 1, _dim, 6 * _dim);
            sil.Dispose();
            // mod is [B,1,6*dim]; flatten to [B,6*dim].
            Tensor mod2d = new(new TensorShape(b, 6 * _dim), DType.F32);
            Buffer.MemoryCopy((void*)mod.DataPointer, (void*)mod2d.DataPointer, (long)b * 6 * _dim * 4, (long)b * 6 * _dim * 4);
            mod.Dispose();
            return (mod2d, emb2d);
        }

        public IEnumerable<Tensor> Weights()
        {
            Tensor?[] ws = [_te1W, _te1B, _te2W, _te2B, _linW, _linB];
            foreach (Tensor? t in ws) if (t is not null) yield return t;
        }
    }

    // ── Transformer block (adaLN-single) ───────────────────────────────────────────────────────────────
    private sealed class Block
    {
        private readonly int _dim, _heads, _headDim;
        private Tensor? _attnNorm, _mlpNorm, _ssTable;
        private Tensor? _qW, _kW, _vW, _oW;
        private Tensor? _gateW, _upW, _downW;

        public Block(int dim, int heads, int headDim) { _dim = dim; _heads = heads; _headDim = headDim; }

        public void Load(IReadOnlyDictionary<string, Tensor> w, string p)
        {
            _attnNorm = WhisperOps.EnsureF32(w[$"{p}.attn_norm.weight"]);
            _mlpNorm = WhisperOps.EnsureF32(w[$"{p}.mlp_norm.weight"]);
            _ssTable = WhisperOps.EnsureF32(w[$"{p}.scale_shift_table"]);   // [6, dim]
            _qW = WhisperOps.EnsureF32(w[$"{p}.attn.q_proj.weight"]);
            _kW = WhisperOps.EnsureF32(w[$"{p}.attn.k_proj.weight"]);
            _vW = WhisperOps.EnsureF32(w[$"{p}.attn.v_proj.weight"]);
            _oW = WhisperOps.EnsureF32(w[$"{p}.attn.o_proj.weight"]);
            _gateW = WhisperOps.EnsureF32(w[$"{p}.mlp.gate.weight"]);
            _upW = WhisperOps.EnsureF32(w[$"{p}.mlp.up.weight"]);
            _downW = WhisperOps.EnsureF32(w[$"{p}.mlp.down.weight"]);
        }

        public Tensor Forward(IBackend backend, Tensor x, Tensor mod, int b, int t)
        {
            int d = _dim;
            // chunk: scale_shift_table[None] + mod.reshape(B,6,D) → 6 × [B,D].
            float* tab = (float*)_ssTable!.DataPointer;     // [6, D]
            float* mp = (float*)mod.DataPointer;            // [B, 6*D]
            float[] shiftMsa = new float[(long)b * d], scaleMsa = new float[(long)b * d], gateMsa = new float[(long)b * d];
            float[] shiftMlp = new float[(long)b * d], scaleMlp = new float[(long)b * d], gateMlp = new float[(long)b * d];
            for (int bi = 0; bi < b; bi++)
                for (int c = 0; c < d; c++)
                {
                    long mo = (long)bi * 6 * d;
                    long o = (long)bi * d + c;
                    shiftMsa[o] = tab[0 * d + c] + mp[mo + 0 * d + c];
                    scaleMsa[o] = tab[1 * d + c] + mp[mo + 1 * d + c];
                    gateMsa[o] = tab[2 * d + c] + mp[mo + 2 * d + c];
                    shiftMlp[o] = tab[3 * d + c] + mp[mo + 3 * d + c];
                    scaleMlp[o] = tab[4 * d + c] + mp[mo + 4 * d + c];
                    gateMlp[o] = tab[5 * d + c] + mp[mo + 5 * d + c];
                }

            // Self-attention path.
            Tensor an = new(new TensorShape(b, t, d), DType.F32);
            backend.RmsNorm(an, x, _attnNorm!, 1e-6f);
            Modulate(an, scaleMsa, shiftMsa, b, t, d);
            Tensor attnOut = Attention(backend, an, b, t);
            an.Dispose();
            // x = x + gate_msa * attn.
            GatedResidual(x, attnOut, gateMsa, b, t, d);
            attnOut.Dispose();

            // MLP path.
            Tensor mn = new(new TensorShape(b, t, d), DType.F32);
            backend.RmsNorm(mn, x, _mlpNorm!, 1e-6f);
            Modulate(mn, scaleMlp, shiftMlp, b, t, d);
            Tensor mlpOut = Mlp(backend, mn, b, t);
            mn.Dispose();
            GatedResidual(x, mlpOut, gateMlp, b, t, d);
            mlpOut.Dispose();
            return x;   // mutated in place and returned (caller owns)
        }

        private static unsafe void Modulate(Tensor x, float[] scale, float[] shift, int b, int t, int d)
        {
            float* xp = (float*)x.DataPointer;
            for (int bi = 0; bi < b; bi++)
                for (int ti = 0; ti < t; ti++)
                {
                    long off = ((long)bi * t + ti) * d;
                    long so = (long)bi * d;
                    for (int c = 0; c < d; c++) xp[off + c] = xp[off + c] * (1f + scale[so + c]) + shift[so + c];
                }
        }

        private static unsafe void GatedResidual(Tensor x, Tensor h, float[] gate, int b, int t, int d)
        {
            float* xp = (float*)x.DataPointer; float* hp = (float*)h.DataPointer;
            for (int bi = 0; bi < b; bi++)
                for (int ti = 0; ti < t; ti++)
                {
                    long off = ((long)bi * t + ti) * d;
                    long go = (long)bi * d;
                    for (int c = 0; c < d; c++) xp[off + c] += gate[go + c] * hp[off + c];
                }
        }

        private Tensor Attention(IBackend backend, Tensor x, int b, int t)
        {
            int d = _dim, h = _heads, hd = _headDim;
            Tensor q = WhisperOps.ProjectLinear(backend, x, _qW!, null, b, t, d, d);
            Tensor k = WhisperOps.ProjectLinear(backend, x, _kW!, null, b, t, d, d);
            Tensor v = WhisperOps.ProjectLinear(backend, x, _vW!, null, b, t, d, d);
            // Apply interleaved RoPE on q,k.
            RopeInterleaved(q, b, t, h, hd);
            RopeInterleaved(k, b, t, h, hd);
            // Bidirectional attention (no mask), scale 1/sqrt(head_dim).
            Tensor outp = new(new TensorShape(b, t, d), DType.F32);
            float* qp = (float*)q.DataPointer; float* kp = (float*)k.DataPointer;
            float* vp = (float*)v.DataPointer; float* op = (float*)outp.DataPointer;
            float scale = 1f / MathF.Sqrt(hd);
            float[] scores = new float[t];
            for (int bi = 0; bi < b; bi++)
                for (int hi = 0; hi < h; hi++)
                {
                    for (int qi = 0; qi < t; qi++)
                    {
                        long qbase = ((long)bi * t + qi) * d + (long)hi * hd;
                        float mx = float.NegativeInfinity;
                        for (int ki = 0; ki < t; ki++)
                        {
                            long kbase = ((long)bi * t + ki) * d + (long)hi * hd;
                            float dot = 0;
                            for (int e = 0; e < hd; e++) dot += qp[qbase + e] * kp[kbase + e];
                            dot *= scale;
                            scores[ki] = dot;
                            if (dot > mx) mx = dot;
                        }
                        float sum = 0;
                        for (int ki = 0; ki < t; ki++) { scores[ki] = MathF.Exp(scores[ki] - mx); sum += scores[ki]; }
                        float inv = 1f / sum;
                        long obase = ((long)bi * t + qi) * d + (long)hi * hd;
                        for (int e = 0; e < hd; e++)
                        {
                            float acc = 0;
                            for (int ki = 0; ki < t; ki++)
                                acc += scores[ki] * vp[((long)bi * t + ki) * d + (long)hi * hd + e];
                            op[obase + e] = acc * inv;
                        }
                    }
                }
            q.Dispose(); k.Dispose(); v.Dispose();
            Tensor proj = WhisperOps.ProjectLinear(backend, outp, _oW!, null, b, t, d, d);
            outp.Dispose();
            return proj;
        }

        // GPT-J interleaved RoPE: head viewed as [hd/2, 2] pairs (x1=even, x2=odd);
        // rot pair = [x1*cos - x2*sin, x1*sin + x2*cos]. inv_freq base 10000 over hd.
        private static unsafe void RopeInterleaved(Tensor t_, int b, int t, int h, int hd)
        {
            float* p = (float*)t_.DataPointer;
            int half = hd / 2;
            int d = h * hd;
            for (int ti = 0; ti < t; ti++)
                for (int i = 0; i < half; i++)
                {
                    float invFreq = 1f / MathF.Pow(10000f, (2f * i) / hd);
                    float ang = ti * invFreq;
                    float cs = MathF.Cos(ang), sn = MathF.Sin(ang);
                    for (int bi = 0; bi < b; bi++)
                        for (int hi = 0; hi < h; hi++)
                        {
                            long baseOff = ((long)bi * t + ti) * d + (long)hi * hd + 2 * i;
                            float x1 = p[baseOff], x2 = p[baseOff + 1];
                            p[baseOff] = x1 * cs - x2 * sn;
                            p[baseOff + 1] = x1 * sn + x2 * cs;
                        }
                }
        }

        private Tensor Mlp(IBackend backend, Tensor x, int b, int t)
        {
            int hidden = (int)_gateW!.Shape[0];
            Tensor g = WhisperOps.ProjectLinear(backend, x, _gateW!, null, b, t, _dim, hidden);
            backend.Silu(g, g);
            Tensor u = WhisperOps.ProjectLinear(backend, x, _upW!, null, b, t, _dim, hidden);
            backend.Mul(g, g, u);
            u.Dispose();
            Tensor outp = WhisperOps.ProjectLinear(backend, g, _downW!, null, b, t, hidden, _dim);
            g.Dispose();
            return outp;
        }

        public IEnumerable<Tensor> Weights()
        {
            Tensor?[] ws = [_attnNorm, _mlpNorm, _ssTable, _qW, _kW, _vW, _oW, _gateW, _upW, _downW];
            foreach (Tensor? t in ws) if (t is not null) yield return t;
        }
    }
}
