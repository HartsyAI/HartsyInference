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
        (Tensor cos1, Tensor sin1) = BuildRope(b, t, _headDim);
        // Block.Forward consumes s and returns a fresh tensor each iteration.
        for (int i = 0; i < _nLayers; i++)
            s = _blocks[i].Forward(backend, s, mod1, b, t, cos1, sin1);
        mod1.Dispose(); cos1.Dispose(); sin1.Dispose();

        // norm_out + scale_shift_table + emb1.
        s = ApplyFinalNorm(backend, s, _ssTable!, emb1, b, t, _inner);
        emb1.Dispose();
        // connection_proj over cat(input, s) along the feature dim.
        Tensor cat = new(new TensorShape(b, t, _inCh + _inner), DType.F32);
        backend.Concat(cat, [input, s], 2);
        s.Dispose();
        Tensor x = _connProj.Forward(backend, cat, b, t);                            // [B,T,inner2]
        cat.Dispose();

        // Stage 2.
        (Tensor mod2, Tensor emb2) = _adaln2.Forward(backend, timestep, b);
        (Tensor cos2, Tensor sin2) = BuildRope(b, t, _headDim2);
        for (int i = 0; i < _nLayers2; i++)
            x = _blocks2[i].Forward(backend, x, mod2, b, t, cos2, sin2);
        mod2.Dispose(); cos2.Dispose(); sin2.Dispose();
        x = ApplyFinalNorm(backend, x, _ssTable2!, emb2, b, t, _inner2);
        emb2.Dispose();

        Tensor outp = _projOut.Forward(backend, x, b, t);                            // [B,T,out_channels]
        x.Dispose();
        return outp;
    }

    // norm_out (affine-free LN, eps 1e-6) then x * (1 + scale) + shift, where
    // [shift, scale] = scale_shift_table[None] + embedded_timestep[:,None] (chunk 2 over dim 1). GPU-resident.
    // Consumes x and returns the normed+modulated result (caller reassigns).
    private static Tensor ApplyFinalNorm(IBackend backend, Tensor x, Tensor ssTable, Tensor emb, int b, int t, int dim)
    {
        // shift[b,dim] = ssTable[0] + emb ; (1+scale)[b,dim] = 1 + ssTable[1] + emb (both rows share emb).
        Tensor ss0 = RowRep(ssTable, 0, b, dim), ss1 = RowRep(ssTable, 1, b, dim);
        Tensor shift = new(new TensorShape(b, dim), DType.F32);
        Tensor scale = new(new TensorShape(b, dim), DType.F32);
        backend.Add(shift, emb, ss0);
        backend.Add(scale, emb, ss1);
        ss0.Dispose(); ss1.Dispose();
        backend.AddScalar(scale, scale, 1f);
        Tensor normed = new(new TensorShape(b, t, dim), DType.F32);
        backend.LayerNormNoAffine(normed, x, 1e-6f);
        x.Dispose();
        backend.AffineBroadcastLastDim(normed, normed, scale, shift);   // in place: normed = normed*(1+scale)+shift
        shift.Dispose(); scale.Dispose();
        return normed;
    }

    // [b,dim] broadcast of ssTable row (host copy; ssTable is a host weight so no D2H sync).
    private static Tensor RowRep(Tensor ssTable, int row, int b, int dim)
    {
        Tensor rep = new(new TensorShape(b, dim), DType.F32);
        float* src = (float*)ssTable.DataPointer + (long)row * dim; float* rp = (float*)rep.DataPointer;
        for (int bi = 0; bi < b; bi++) Buffer.MemoryCopy(src, rp + (long)bi * dim, (long)dim * 4, (long)dim * 4);
        return rep;
    }

    // Interleaved (GPT-J) RoPE tables [b*t, headDim]: freq_i (angle ti·10000^(-2i/hd), ti = pos % t) at
    // index 2i (and 2i+1) so the on-device WanRopeInterleaved kernel reads its angle at 2i. Matches the host
    // RopeInterleaved math exactly (float cos/sin), tiled over the batch so seqLen = b*t is a flat pass.
    private static (Tensor Cos, Tensor Sin) BuildRope(int b, int t, int hd)
    {
        int half = hd / 2;
        Tensor cos = new(new TensorShape(b * t, hd), DType.F32);
        Tensor sin = new(new TensorShape(b * t, hd), DType.F32);
        float* cp = (float*)cos.DataPointer; float* sp = (float*)sin.DataPointer;
        for (int s = 0; s < b * t; s++)
        {
            int ti = s % t;
            for (int i = 0; i < half; i++)
            {
                float invFreq = 1f / MathF.Pow(10000f, (2f * i) / hd);
                float ang = ti * invFreq;
                float cs = MathF.Cos(ang), sn = MathF.Sin(ang);
                int i0 = 2 * i;
                cp[(long)s * hd + i0] = cp[(long)s * hd + i0 + 1] = cs;
                sp[(long)s * hd + i0] = sp[(long)s * hd + i0 + 1] = sn;
            }
        }
        return (cos, sin);
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
            backend.Transpose2D(xc, x, t, _in);
            Tensor conv = new(new TensorShape(b, _filter, t), DType.F32);
            backend.Conv1d(conv, xc, _convW!, _convB, 1, 1, 1, 1, 1);
            xc.Dispose();
            Tensor convBT = new(new TensorShape(b, t, _filter), DType.F32);
            backend.Transpose2D(convBT, conv, _filter, t);
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

        // Fully GPU-resident. Consumes (disposes) x and returns a new tensor; caller reassigns.
        // ropeCos/ropeSin are [b*t, headDim] tables (freq_i at index 2i), shared across a stage's blocks.
        public Tensor Forward(IBackend backend, Tensor x, Tensor mod, int b, int t, Tensor ropeCos, Tensor ropeSin)
        {
            int d = _dim;
            // combined = mod.reshape(B,6,D) + scale_shift_table[None] → [B,6D], then chunk into six [B,D].
            Tensor ssRep = BuildSsRep(b, d);   // host-built [B,6D] broadcast of _ssTable (sync-free)
            Tensor comb = new(new TensorShape(b, 6 * d), DType.F32);
            backend.Add(comb, mod, ssRep);
            ssRep.Dispose();
            Tensor shiftMsa = Slice(backend, comb, 0, d, b), scaleMsa = Slice(backend, comb, 1, d, b), gateMsa = Slice(backend, comb, 2, d, b);
            Tensor shiftMlp = Slice(backend, comb, 3, d, b), scaleMlp = Slice(backend, comb, 4, d, b), gateMlp = Slice(backend, comb, 5, d, b);
            comb.Dispose();
            backend.AddScalar(scaleMsa, scaleMsa, 1f);   // modulate uses (1 + scale)
            backend.AddScalar(scaleMlp, scaleMlp, 1f);

            // Self-attention path: modulate(RmsNorm(x)) → attn → x + gate_msa * attn.
            Tensor an = new(new TensorShape(b, t, d), DType.F32);
            backend.RmsNorm(an, x, _attnNorm!, 1e-6f);
            Tensor anMod = new(new TensorShape(b, t, d), DType.F32);
            backend.AffineBroadcastLastDim(anMod, an, scaleMsa, shiftMsa);
            an.Dispose();
            Tensor attnOut = Attention(backend, anMod, b, t, ropeCos, ropeSin);
            anMod.Dispose();
            Tensor afterAttn = new(new TensorShape(b, t, d), DType.F32);
            backend.GatedResidualLastDim(afterAttn, x, attnOut, gateMsa);
            x.Dispose(); attnOut.Dispose();

            // MLP path.
            Tensor mn = new(new TensorShape(b, t, d), DType.F32);
            backend.RmsNorm(mn, afterAttn, _mlpNorm!, 1e-6f);
            Tensor mnMod = new(new TensorShape(b, t, d), DType.F32);
            backend.AffineBroadcastLastDim(mnMod, mn, scaleMlp, shiftMlp);
            mn.Dispose();
            Tensor mlpOut = Mlp(backend, mnMod, b, t);
            mnMod.Dispose();
            Tensor outX = new(new TensorShape(b, t, d), DType.F32);
            backend.GatedResidualLastDim(outX, afterAttn, mlpOut, gateMlp);
            afterAttn.Dispose(); mlpOut.Dispose();
            shiftMsa.Dispose(); scaleMsa.Dispose(); gateMsa.Dispose();
            shiftMlp.Dispose(); scaleMlp.Dispose(); gateMlp.Dispose();
            return outX;
        }

        // [B,6D] broadcast of _ssTable[6,D] over the batch (host copy; both operands host so no D2H sync).
        private Tensor BuildSsRep(int b, int d)
        {
            Tensor rep = new(new TensorShape(b, 6 * d), DType.F32);
            float* tab = (float*)_ssTable!.DataPointer; float* rp = (float*)rep.DataPointer;
            for (int bi = 0; bi < b; bi++)
                Buffer.MemoryCopy(tab, rp + (long)bi * 6 * d, (long)6 * d * 4, (long)6 * d * 4);
            return rep;
        }

        private static Tensor Slice(IBackend backend, Tensor comb, int idx, int d, int b)
        {
            Tensor s = new(new TensorShape(b, d), DType.F32);
            backend.SliceLastDim(s, comb, idx * d);
            return s;
        }

        private Tensor Attention(IBackend backend, Tensor x, int b, int t, Tensor ropeCos, Tensor ropeSin)
        {
            int d = _dim, h = _heads, hd = _headDim;
            // Projections write straight into [b,t,heads,headDim] (shape is metadata to Linear).
            TensorShape packed = new(b, t, h, hd);
            Tensor q = new(packed, DType.F32), k = new(packed, DType.F32), v = new(packed, DType.F32);
            backend.Linear(q, x, _qW!, null);
            backend.Linear(k, x, _kW!, null);
            backend.Linear(v, x, _vW!, null);
            // Interleaved (GPT-J) RoPE; flat q is [b*t, heads, headDim] so seqLen = b*t.
            backend.WanRopeInterleaved(q, ropeCos, ropeSin, b * t, h, hd);
            backend.WanRopeInterleaved(k, ropeCos, ropeSin, b * t, h, hd);
            TensorShape mh = new(b, h, t, hd);
            Tensor qMh = new(mh, DType.F32), kMh = new(mh, DType.F32), vMh = new(mh, DType.F32);
            backend.Permute0213(qMh, q, t, h, hd);
            backend.Permute0213(kMh, k, t, h, hd);
            backend.Permute0213(vMh, v, t, h, hd);
            q.Dispose(); k.Dispose(); v.Dispose();
            // Bidirectional (no mask), scale 1/sqrt(head_dim). ScaledDotProductAttention instead of the LM
            // FlashAttention API: same [b,h,t,hd] contract, but it dispatches to the Sage-INT8/cuDNN fused
            // engines for D∈{64,128} — this full-sequence attention was the codec's single largest GPU cost
            // (300 monolithic lm_flash_attn_f32 calls × ~5.7 ms on a 10 s song, 2026-07-25 nsys). allowF16
            // is safe here: Q/K are RMS-normed (bounded scores), the same gate the LM prefill path uses.
            Tensor attn = new(mh, DType.F32);
            backend.ScaledDotProductAttention(attn, qMh, kMh, vMh, null, 1f / MathF.Sqrt(hd), allowF16: true);
            qMh.Dispose(); kMh.Dispose(); vMh.Dispose();
            Tensor merged = new(new TensorShape(b, t, d), DType.F32);
            backend.Permute0213(merged, attn, h, t, hd);
            attn.Dispose();
            Tensor proj = WhisperOps.ProjectLinear(backend, merged, _oW!, null, b, t, d, d);
            merged.Dispose();
            return proj;
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
