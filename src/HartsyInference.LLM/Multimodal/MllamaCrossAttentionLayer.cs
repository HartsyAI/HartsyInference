using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Rotary;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Multimodal;

/// <summary>The gated cross-attention decoder layer of Llama-3.2-Vision (mllama), interleaved into the text decoder at layers <c>[3,8,13,18,23,28,33,38]</c> (11B) instead of splicing image tokens into the sequence like every other VLM here.</summary>
/// <remarks>
/// <code>
///   h = text + tanh(attn_gate) · o_proj( cross_attn( q=q_norm(q_proj(rms(text))), k=k_norm(k_proj(vision)), v=v_proj(vision) ) )
///   out = h    + tanh(mlp_gate)  · mlp( rms(h) )
/// </code>
/// The query comes from the text hidden states; the keys/values from the (fixed) vision features, so the attention is bidirectional over all vision tokens (no causal mask). Q/K are RMSNorm-ed per head. Two learned scalar gates (<c>cross_attn_attn_gate</c>, <c>cross_attn_mlp_gate</c>) are passed through <c>tanh</c>; both are ≈0 at init so a freshly-initialized layer is a no-op (a strong correctness signal). All math runs through <see cref="IBackend"/>; the attention reuses the decoder's GQA FlashAttention with <c>causal:false</c>.
/// </remarks>
public sealed unsafe class MllamaCrossAttentionLayer
{
    private readonly int _hidden, _numHeads, _numKvHeads, _headDim, _inter;
    private readonly float _rmsEps;
    private readonly bool _lowVram;
    private readonly AttentionLayer _attn;

    private Tensor _inNorm = null!, _postAttnNorm = null!;
    private Tensor _qW = null!, _kW = null!, _vW = null!, _oW = null!;
    private Tensor _qNorm = null!, _kNorm = null!;
    private Tensor _gateW = null!, _upW = null!, _downW = null!;
    private float _attnGate, _mlpGate;   // raw (pre-tanh) learned scalar gates

    public MllamaCrossAttentionLayer(int hidden, int numHeads, int numKvHeads, int headDim, int intermediate, float rmsEps, bool lowVram = false)
    {
        _hidden = hidden; _numHeads = numHeads; _numKvHeads = numKvHeads; _headDim = headDim;
        _inter = intermediate; _rmsEps = rmsEps; _lowVram = lowVram;
        _attn = new AttentionLayer(new AttentionLayerConfig
        {
            Heads = numHeads, KvHeads = numKvHeads, HeadDim = headDim, ComputeDType = DType.F32, Scale = 1f / MathF.Sqrt(headDim),
        });
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _inNorm = w[$"{prefix}.input_layernorm.weight"];
        _postAttnNorm = w[$"{prefix}.post_attention_layernorm.weight"];
        _qW = w[$"{prefix}.cross_attn.q_proj.weight"];
        _kW = w[$"{prefix}.cross_attn.k_proj.weight"];
        _vW = w[$"{prefix}.cross_attn.v_proj.weight"];
        _oW = w[$"{prefix}.cross_attn.o_proj.weight"];
        _qNorm = w[$"{prefix}.cross_attn.q_norm.weight"];
        _kNorm = w[$"{prefix}.cross_attn.k_norm.weight"];
        _gateW = w[$"{prefix}.mlp.gate_proj.weight"];
        _upW = w[$"{prefix}.mlp.up_proj.weight"];
        _downW = w[$"{prefix}.mlp.down_proj.weight"];
        _attnGate = Scalar(w[$"{prefix}.cross_attn_attn_gate"]);
        _mlpGate = Scalar(w[$"{prefix}.cross_attn_mlp_gate"]);
    }

    /// <summary>Cross-attention + gated FFN: <paramref name="text"/> is <c>[1, t, hidden]</c> (the decoder query), <paramref name="vision"/> is <c>[1, L, hidden]</c> (the encoded image features); returns <c>[1, t, hidden]</c>.</summary>
    /// <param name="rope">Optional MOSS-VL tables: the query rows are rotated by <see cref="MultimodalRope.TextCos"/>, the key rows by the vision tables, and rows with a 0 <see cref="MultimodalRope.RowMask"/> contribute nothing. Null (the mllama default) applies no rotation and no row mask.</param>
    /// <param name="kvCache">With a cache, <paramref name="vision"/> holds only the vision tokens this forward brings (null when it
    /// brings none): they are projected once and appended, and the attention reads every vision token the cache holds plus them.
    /// Without one (mllama), <paramref name="vision"/> is the whole vision sequence, projected for this call only.</param>
    public Tensor Forward(IBackend backend, Tensor text, int t, Tensor? vision, int visionLen, MultimodalRope? rope = null,
        CrossKvCache? kvCache = null, int layer = 0)
    {
        if (kvCache is not null)
        {
            int newTokens = vision is null ? 0 : visionLen;
            if (newTokens > 0)
            {
                (Tensor k, Tensor v) = ProjectVision(backend, vision!, newTokens, rope);
                try
                {
                    kvCache.Append(backend, layer, k, v);
                }
                finally
                {
                    k.Dispose();
                    v.Dispose();
                }
            }
            (Tensor K, Tensor V) cached = kvCache.Get(layer)
                ?? throw new InvalidOperationException($"Cross-attention layer {layer} has no vision keys: no vision was appended.");
            return Attend(backend, text, t, cached.K, cached.V, kvCache.Length + newTokens, rope);
        }

        if (vision is null) throw new ArgumentNullException(nameof(vision), "Cross-attention without a cache needs the vision sequence.");
        (Tensor kMh, Tensor vMh) = ProjectVision(backend, vision, visionLen, rope);
        try
        {
            return Attend(backend, text, t, kMh, vMh, visionLen, rope);
        }
        finally
        {
            kMh.Dispose();
            vMh.Dispose();
        }
    }

    /// <summary>Projects the vision sequence to head-major keys and values: <c>k_norm(k_proj(vision))</c> rotated by the vision tables, and
    /// <c>v_proj(vision)</c>. The caller owns both results.</summary>
    private (Tensor K, Tensor V) ProjectVision(IBackend backend, Tensor vision, int visionLen, MultimodalRope? rope)
    {
        int hkv = _numKvHeads, d = _headDim;
        Tensor k = new(new TensorShape(1, visionLen, hkv, d), DType.F32);
        GenericTransformer.Project(backend, k, vision, _kW, null, lowVram: _lowVram);
        Tensor kN = new(new TensorShape(1, visionLen, hkv, d), DType.F32);
        backend.RmsNorm(kN, k, _kNorm, _rmsEps);
        k.Dispose();
        if (rope is not null)
        {
            Tensor cos = rope.VisionCos ?? throw new ArgumentException("Projecting vision keys needs the vision rotary tables.", nameof(rope));
            backend.Rotary(RotaryCall.TokenMajor(kN, cos, rope.VisionSin!, RotaryPairing.RotateHalf, RopeTableLayout.RotateHalf));
        }
        Tensor v = new(new TensorShape(1, visionLen, hkv, d), DType.F32);
        GenericTransformer.Project(backend, v, vision, _vW, null, lowVram: _lowVram);

        // Head-major [1, heads, seq, d] for the GQA FlashAttention.
        Tensor kMh = new(new TensorShape(1, hkv, visionLen, d), DType.F32); backend.Permute0213(kMh, kN, visionLen, hkv, d);
        Tensor vMh = new(new TensorShape(1, hkv, visionLen, d), DType.F32); backend.Permute0213(vMh, v, visionLen, hkv, d);
        kN.Dispose(); v.Dispose();
        return (kMh, vMh);
    }

    /// <summary>The query side, attention over borrowed head-major vision keys and values (bidirectional: no causal mask), and the gated
    /// FFN. Does not dispose <paramref name="kMh"/> or <paramref name="vMh"/>.</summary>
    private Tensor Attend(IBackend backend, Tensor text, int t, Tensor kMh, Tensor vMh, int visionLen, MultimodalRope? rope)
    {
        int hq = _numHeads, d = _headDim;
        TensorShape flat = new(1, t, _hidden);

        // --- Cross attention ---
        Tensor pre = new(flat, DType.F32);
        backend.RmsNorm(pre, text, _inNorm, _rmsEps);

        Tensor q = new(new TensorShape(1, t, hq, d), DType.F32);
        GenericTransformer.Project(backend, q, pre, _qW, null, lowVram: _lowVram);
        pre.Dispose();
        Tensor qN = new(new TensorShape(1, t, hq, d), DType.F32);
        backend.RmsNorm(qN, q, _qNorm, _rmsEps);   // q-norm over the head dim
        q.Dispose();
        if (rope is not null) backend.Rotary(RotaryCall.TokenMajor(qN, rope.TextCos, rope.TextSin, RotaryPairing.RotateHalf, RopeTableLayout.RotateHalf));

        // Groups that each name their own contiguous rows attend only those rows (consumes qN).
        if (rope?.ImageGroups is { Count: > 0 } ranged && hq * d == _hidden && ranged.All(g => g.RowStart >= 0))
        {
            Tensor rangedOut = ApplyRowMask(backend, RangedGroupAttention(backend, qN, t, kMh, vMh, ranged), rope);
            return GatedTail(backend, text, t, rangedOut, rope);
        }

        Tensor qMh = new(new TensorShape(1, hq, t, d), DType.F32); backend.Permute0213(qMh, qN, t, hq, d);
        qN.Dispose();

        // The keys and values are borrowed throughout (the kv cache or the caller disposes them). The query is consumed.
        Tensor attnOut = rope?.ImageGroups is { } groups
            ? GroupedAttention(backend, qMh, t, kMh, vMh, groups)
            : ProjectMerged(backend, _attn.AttendOverKvHeadMajor(backend, qMh, kMh, vMh, null, 0, 0,
                new KvAttentionCall { KvLength = visionLen, Bidirectional = true }, OperandOwnership.Query), t);
        attnOut = ApplyRowMask(backend, attnOut, rope);
        return GatedTail(backend, text, t, attnOut, rope);
    }

    /// <summary>The gated residual over the projected, row-masked attention output <paramref name="attnOut"/> (consumed), then the gated
    /// FFN: <c>h = text + tanh(attn_gate) · attnOut</c>, <c>out = h + tanh(mlp_gate) · mask(SwiGLU(norm(h)))</c>.</summary>
    private Tensor GatedTail(IBackend backend, Tensor text, int t, Tensor attnOut, MultimodalRope? rope)
    {
        TensorShape flat = new(1, t, _hidden);
        Tensor h = new(flat, DType.F32);
        backend.Scale(attnOut, attnOut, MathF.Tanh(_attnGate));
        backend.Add(h, text, attnOut);
        attnOut.Dispose();

        Tensor postNorm = new(flat, DType.F32);
        backend.RmsNorm(postNorm, h, _postAttnNorm, _rmsEps);
        Tensor mlp = ApplyRowMask(backend, SwiGlu(backend, postNorm, t), rope);
        postNorm.Dispose();
        backend.Scale(mlp, mlp, MathF.Tanh(_mlpGate));
        Tensor outp = new(flat, DType.F32);
        backend.Add(outp, h, mlp);
        h.Dispose(); mlp.Dispose();
        return outp;
    }

    /// <summary>Multi-image cross-attention when each group's rows are one contiguous range: the group's query rows attend its key
    /// prefix, the group outputs are stitched back in row order (zeros for rows no group covers), and the result is projected once.
    /// Equal to <see cref="GroupedAttention"/>, at the cost of one attention over the rows instead of one per group over all rows.
    /// Consumes the token-major query <paramref name="qN"/> (<c>[1, t, heads, d]</c>); the keys and values are borrowed.</summary>
    private Tensor RangedGroupAttention(IBackend backend, Tensor qN, int t, Tensor kMh, Tensor vMh, IReadOnlyList<CrossImageGroup> groups)
    {
        int hq = _numHeads, d = _headDim;
        List<Tensor> pieces = [];
        try
        {
            int row = 0;
            foreach (CrossImageGroup group in groups.OrderBy(g => g.RowStart))
            {
                if (group.RowStart < row || group.RowStart + group.RowCount > t)
                    throw new ArgumentException($"Cross-attention groups overlap or run past the {t} rows.", nameof(groups));
                if (group.RowStart > row) pieces.Add(Zeros(backend, group.RowStart - row));
                int rows = group.RowCount;
                Tensor qRows = new(new TensorShape(1, rows, hq, d), DType.F32);
                backend.SliceRows(qRows, qN, group.RowStart * hq);   // rows of d values: a row of the sequence is hq of them
                Tensor qMh = new(new TensorShape(1, hq, rows, d), DType.F32);
                backend.Permute0213(qMh, qRows, rows, hq, d);
                qRows.Dispose();
                Tensor attn = _attn.AttendOverKvHeadMajor(backend, qMh, kMh, vMh, null, 0, 0,
                    new KvAttentionCall { KvLength = group.KvLength, Bidirectional = true }, OperandOwnership.Query);
                pieces.Add(_attn.MergeHeads(backend, attn, new TensorShape(1, rows, _hidden)));
                row = group.RowStart + rows;
            }
            if (row < t) pieces.Add(Zeros(backend, t - row));

            Tensor merged;
            if (pieces.Count == 1)
            {
                merged = pieces[0];
                pieces.Clear();
            }
            else
            {
                merged = new Tensor(new TensorShape(1, t, _hidden), DType.F32);
                backend.Concat(merged, [.. pieces], dim: 1);
            }
            using (merged)
            {
                Tensor projected = new(new TensorShape(1, t, _hidden), DType.F32);
                GenericTransformer.Project(backend, projected, merged, _oW, null, lowVram: _lowVram);
                return projected;
            }
        }
        finally
        {
            foreach (Tensor piece in pieces) piece.Dispose();
            qN.Dispose();
        }
    }

    private static ArgumentException MissingSelect() =>
        new("A cross-attention group without a row range needs its row selection mask.", "groups");

    private Tensor Zeros(IBackend backend, int rows)
    {
        Tensor zeros = new(new TensorShape(1, rows, _hidden), DType.F32);
        zeros.AsSpan<float>().Clear();
        return zeros;
    }

    /// <summary>Projects merged-head attention output <paramref name="attn"/> (disposed) to <c>[1, t, hidden]</c>; the caller owns the result.</summary>
    private Tensor ProjectMerged(IBackend backend, Tensor attn, int t)
    {
        Tensor attnFlat = _attn.MergeHeads(backend, attn, new TensorShape(1, t, _numHeads * _headDim));
        Tensor projected = new(new TensorShape(1, t, _hidden), DType.F32);
        GenericTransformer.Project(backend, projected, attnFlat, _oW, null, lowVram: _lowVram);
        attnFlat.Dispose();
        return projected;
    }

    /// <summary>Multi-image cross-attention: each group's rows attend the vision keys of the images they see (a prefix of the sequence),
    /// and the masked group outputs are summed. Consumes <paramref name="qMh"/>; the keys and values are borrowed.</summary>
    /// <remarks>The output projection is linear and the group masks are disjoint row selections, so the masked attention outputs are
    /// summed first and projected once: <c>Σ_g m_g ⊙ (a_g W_o) = (Σ_g m_g ⊙ a_g) W_o</c>. With one group per video frame that is one
    /// <c>o_proj</c> per layer instead of one per frame, which dominated a long video's prefill.</remarks>
    private Tensor GroupedAttention(IBackend backend, Tensor qMh, int t, Tensor kMh, Tensor vMh, IReadOnlyList<CrossImageGroup> groups)
    {
        if (_numHeads * _headDim != _hidden)
            return GroupedAttentionProjectEach(backend, qMh, t, kMh, vMh, groups);
        TensorShape flat = new(1, t, _hidden);
        Tensor? sum = null;
        try
        {
            foreach (CrossImageGroup group in groups)
            {
                // The query is shared by every group, so it is not consumed until the last one (the finally below).
                Tensor attn = _attn.AttendOverKvHeadMajor(backend, qMh, kMh, vMh, null, 0, 0,
                    new KvAttentionCall { KvLength = group.KvLength, Bidirectional = true }, OperandOwnership.None);
                Tensor merged = _attn.MergeHeads(backend, attn, flat);
                Tensor masked = new(flat, DType.F32);
                backend.Mul(masked, merged, group.Select ?? throw MissingSelect());
                merged.Dispose();
                if (sum is null)
                {
                    sum = masked;
                    continue;
                }
                Tensor next = new(flat, DType.F32);
                backend.Add(next, sum, masked);
                sum.Dispose();
                masked.Dispose();
                sum = next;
            }
            if (sum is null) throw new ArgumentException("A multi-image cross-attention needs at least one group.", nameof(groups));
            Tensor projected = new(flat, DType.F32);
            GenericTransformer.Project(backend, projected, sum, _oW, null, lowVram: _lowVram);
            return projected;
        }
        finally
        {
            sum?.Dispose();
            qMh.Dispose();
        }
    }

    /// <summary><see cref="GroupedAttention"/> for a layer whose merged heads are not the hidden width: each group is projected, then
    /// masked and summed.</summary>
    private Tensor GroupedAttentionProjectEach(IBackend backend, Tensor qMh, int t, Tensor kMh, Tensor vMh, IReadOnlyList<CrossImageGroup> groups)
    {
        TensorShape flat = new(1, t, _hidden);
        Tensor? sum = null;
        try
        {
            foreach (CrossImageGroup group in groups)
            {
                Tensor attn = _attn.AttendOverKvHeadMajor(backend, qMh, kMh, vMh, null, 0, 0,
                    new KvAttentionCall { KvLength = group.KvLength, Bidirectional = true }, OperandOwnership.None);
                Tensor part = ProjectMerged(backend, attn, t);
                Tensor masked = new(flat, DType.F32);
                backend.Mul(masked, part, group.Select ?? throw MissingSelect());
                part.Dispose();
                if (sum is null)
                {
                    sum = masked;
                    continue;
                }
                Tensor next = new(flat, DType.F32);
                backend.Add(next, sum, masked);
                sum.Dispose();
                masked.Dispose();
                sum = next;
            }
            return sum ?? throw new ArgumentException("A multi-image cross-attention needs at least one group.", nameof(groups));
        }
        catch
        {
            sum?.Dispose();
            throw;
        }
        finally
        {
            qMh.Dispose();
        }
    }

    /// <summary>Returns <paramref name="x"/> with each row scaled by the rope's row mask, disposing <paramref name="x"/>; returns it unchanged when there is no mask.</summary>
    private static Tensor ApplyRowMask(IBackend backend, Tensor x, MultimodalRope? rope)
    {
        if (rope?.RowMask is null) return x;
        Tensor masked = new(x.Shape, DType.F32);
        backend.Mul(masked, x, rope.RowMask);
        x.Dispose();
        return masked;
    }

    private Tensor SwiGlu(IBackend backend, Tensor x, int t)
    {
        TensorShape ff = new(1, t, _inter);
        Tensor gate = new(ff, DType.F32);
        GenericTransformer.Project(backend, gate, x, _gateW, null, lowVram: _lowVram);
        Tensor act = new(ff, DType.F32);
        backend.Silu(act, gate);
        gate.Dispose();
        Tensor up = new(ff, DType.F32);
        GenericTransformer.Project(backend, up, x, _upW, null, lowVram: _lowVram);
        Tensor comb = new(ff, DType.F32);
        backend.Mul(comb, act, up);
        act.Dispose(); up.Dispose();
        Tensor outp = new(new TensorShape(1, t, _hidden), DType.F32);
        GenericTransformer.Project(backend, outp, comb, _downW, null, lowVram: _lowVram);
        comb.Dispose();
        return outp;
    }

    /// <summary>The loaded weight tensors (for GPU residency / lifetime management by the host transformer).</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        yield return _inNorm; yield return _postAttnNorm;
        yield return _qW; yield return _kW; yield return _vW; yield return _oW;
        yield return _qNorm; yield return _kNorm;
        yield return _gateW; yield return _upW; yield return _downW;
    }

    private static float Scalar(Tensor t)
    {
        if (t.DType != DType.F32) throw new NotSupportedException("mllama gate scalar must be F32.");
        return *(float*)t.DataPointer;
    }
}
