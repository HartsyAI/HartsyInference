using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Moonshine;

/// <summary>Moonshine text decoder. Autoregressive Transformer with three sub-blocks per
/// layer — RoPE causal self-attention, cross-attention against the encoder hidden states,
/// then a SwiGLU MLP. Same KV-cache pattern as our Whisper decoder.
///
/// <para><b>Differences from Whisper decoder:</b>
/// <list type="bullet">
///   <item>All norms are RMSNorm (Llama-style, weight-only).</item>
///   <item>All attention projections are bias-less (<c>attention_bias=false</c>).</item>
///   <item>Self-attention applies RoPE; cross-attention does not (encoder hidden states
///         already carry positional info).</item>
///   <item>MLP is gated SwiGLU: fc1 outputs 2× intermediate; split, gate the first half
///         with SiLU, multiply by the second half, fc2 reduces.</item>
///   <item>Logits use weight-tied <c>embed_tokens</c> (no separate <c>proj_out</c>).</item>
/// </list></para></summary>
public sealed unsafe class MoonshineDecoder : IDisposable
{
    private readonly MoonshineConfig _cfg;
    private readonly MoonshineDecoderLayer[] _layers;

    private Tensor? _embedTokens;  // [vocab, hidden]
    private Tensor? _finalNormWeight;

    private float[]? _cosTable, _sinTable;
    private bool _loaded;
    private int _disposed;

    public MoonshineConfig Config => _cfg;

    public MoonshineDecoder(MoonshineConfig cfg)
    {
        _cfg = cfg;
        _layers = new MoonshineDecoderLayer[cfg.DecoderLayers];
        for (int i = 0; i < cfg.DecoderLayers; i++) _layers[i] = new MoonshineDecoderLayer(cfg);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "model.decoder")
    {
        _embedTokens = WhisperOps.EnsureF32(weights[$"{prefix}.embed_tokens.weight"]);
        for (int i = 0; i < _layers.Length; i++)
            _layers[i].LoadWeights(weights, $"{prefix}.layers.{i}");
        _finalNormWeight = WhisperOps.EnsureF32(weights[$"{prefix}.norm.weight"]);

        // RoPE table sized to MaxTextPositions (194). Self-attention positions never
        // exceed that — we throw at decode time if a request would.
        (float[] cos, float[] sin) = RotaryEmbedding.GetTables(_cfg.RotaryDim, _cfg.RopeTheta, _cfg.MaxTextPositions);
        _cosTable = cos;
        _sinTable = sin;
        _loaded = true;
    }

    /// <summary>Decode state. Holds per-layer cross-attention K/V (precomputed once from
    /// encoder output) and self-attention K/V (appended per step).</summary>
    public sealed class DecodeState : IDisposable
    {
        public Tensor[] CrossK { get; }
        public Tensor[] CrossV { get; }
        public Tensor[] SelfK { get; }
        public Tensor[] SelfV { get; }
        public int CurrentPos { get; set; }

        public DecodeState(MoonshineConfig cfg, int encSeqLen)
        {
            CrossK = new Tensor[cfg.DecoderLayers];
            CrossV = new Tensor[cfg.DecoderLayers];
            SelfK = new Tensor[cfg.DecoderLayers];
            SelfV = new Tensor[cfg.DecoderLayers];
            TensorShape cross = new(1, cfg.NumHeads, encSeqLen, cfg.HeadDim);
            TensorShape self = new(1, cfg.NumHeads, cfg.MaxTextPositions, cfg.HeadDim);
            for (int i = 0; i < cfg.DecoderLayers; i++)
            {
                CrossK[i] = new Tensor(cross, DType.F32);
                CrossV[i] = new Tensor(cross, DType.F32);
                SelfK[i] = new Tensor(self, DType.F32);
                SelfV[i] = new Tensor(self, DType.F32);
            }
        }

        public void Dispose()
        {
            foreach (Tensor t in CrossK) t.Dispose();
            foreach (Tensor t in CrossV) t.Dispose();
            foreach (Tensor t in SelfK) t.Dispose();
            foreach (Tensor t in SelfV) t.Dispose();
        }
    }

    /// <summary>Precomputes cross-attention K/V from the encoder output. Cross-attn does
    /// NOT receive RoPE (verified against the HF transformers Moonshine reference) so the
    /// cached tensors are just the multi-head-reshaped projections.</summary>
    public DecodeState StartDecode(IBackend backend, Tensor encoderHidden)
    {
        ThrowIfDisposed();
        if (!_loaded) throw new InvalidOperationException("Call LoadWeights before StartDecode.");
        int batch = (int)encoderHidden.Shape[0];
        if (batch != 1) throw new NotSupportedException("MoonshineDecoder supports batch=1 only for now.");
        int encSeq = (int)encoderHidden.Shape[1];

        DecodeState state = new(_cfg, encSeq);
        for (int i = 0; i < _layers.Length; i++)
            _layers[i].PrecomputeCrossKv(backend, encoderHidden, state.CrossK[i], state.CrossV[i], encSeq);
        return state;
    }

    /// <summary>Runs the decoder for a sequence of tokens (either the full prompt prefix
    /// at start, or a single new token per step thereafter). Returns logits for the LAST
    /// token in <paramref name="tokenIds"/>.</summary>
    public Tensor DecodeStep(IBackend backend, ReadOnlySpan<int> tokenIds, DecodeState state)
    {
        ThrowIfDisposed();
        int newCount = tokenIds.Length;
        if (newCount == 0) throw new ArgumentException("tokenIds empty", nameof(tokenIds));
        if (state.CurrentPos + newCount > _cfg.MaxTextPositions)
            throw new InvalidOperationException($"Decode would exceed MaxTextPositions ({_cfg.MaxTextPositions})");

        int d = _cfg.HiddenSize;
        int posStart = state.CurrentPos;

        // Token embedding — no positional add (RoPE handles position inside attention).
        Tensor hidden = new(new TensorShape(1, newCount, d), DType.F32);
        EmbedTokens(hidden, tokenIds, d);

        for (int i = 0; i < _layers.Length; i++)
        {
            Tensor next = _layers[i].Forward(backend, hidden,
                state.SelfK[i], state.SelfV[i],
                state.CrossK[i], state.CrossV[i],
                posStart, newCount, _cosTable!, _sinTable!);
            hidden.Dispose();
            hidden = next;
        }

        Tensor normed = new(hidden.Shape, DType.F32);
        MoonshineOps.LayerNormNoBias(normed, hidden, _finalNormWeight!, _cfg.LayerNormEps);
        hidden.Dispose();

        // Logits for the last position only — that's what greedy decode samples.
        Tensor lastHidden = new(new TensorShape(1, 1, d), DType.F32);
        float* nPtr = (float*)normed.DataPointer;
        float* lPtr = (float*)lastHidden.DataPointer;
        int srcOff = (newCount - 1) * d;
        for (int k = 0; k < d; k++) lPtr[k] = nPtr[srcOff + k];
        normed.Dispose();

        // Weight-tied LM head: logits = lastHidden @ embed_tokens^T.
        Tensor logits = new(new TensorShape(1, _cfg.VocabSize), DType.F32);
        ComputeLogits(lastHidden, logits, _cfg.VocabSize, d);
        lastHidden.Dispose();

        state.CurrentPos += newCount;
        return logits;
    }

    private void EmbedTokens(Tensor hidden, ReadOnlySpan<int> tokenIds, int d)
    {
        float* hPtr = (float*)hidden.DataPointer;
        float* ePtr = (float*)_embedTokens!.DataPointer;
        for (int s = 0; s < tokenIds.Length; s++)
        {
            int tok = tokenIds[s];
            int hOff = s * d;
            int eOff = tok * d;
            for (int k = 0; k < d; k++) hPtr[hOff + k] = ePtr[eOff + k];
        }
    }

    private void ComputeLogits(Tensor hidden, Tensor logits, int vocab, int d)
    {
        float* h = (float*)hidden.DataPointer;
        float* w = (float*)_embedTokens!.DataPointer;
        float* l = (float*)logits.DataPointer;
        for (int v = 0; v < vocab; v++)
        {
            float acc = 0f;
            int wRow = v * d;
            for (int k = 0; k < d; k++) acc += h[k] * w[wRow + k];
            l[v] = acc;
        }
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_embedTokens is not null) yield return _embedTokens;
        foreach (MoonshineDecoderLayer layer in _layers)
            foreach (Tensor t in layer.EnumerateWeights()) yield return t;
        if (_finalNormWeight is not null) yield return _finalNormWeight;
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0) throw new ObjectDisposedException(nameof(MoonshineDecoder));
    }

    public void Dispose() { Interlocked.Exchange(ref _disposed, 1); }
}

/// <summary>Single Moonshine decoder block. Pre-norm causal self-attn (RoPE, KV-cached)
/// → pre-norm cross-attn (against precomputed encoder K/V, no RoPE) → pre-norm SwiGLU MLP.</summary>
internal sealed unsafe class MoonshineDecoderLayer
{
    private readonly MoonshineConfig _cfg;

    private Tensor? _inputLnW;
    private Tensor? _selfQW, _selfKW, _selfVW, _selfOW;

    private Tensor? _postAttnLnW;
    private Tensor? _crossQW, _crossKW, _crossVW, _crossOW;

    private Tensor? _finalLnW;
    private Tensor? _fc1W, _fc1B, _fc2W, _fc2B;

    public MoonshineDecoderLayer(MoonshineConfig cfg) { _cfg = cfg; }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _inputLnW = WhisperOps.EnsureF32(weights[$"{prefix}.input_layernorm.weight"]);
        _selfQW = WhisperOps.EnsureF32(weights[$"{prefix}.self_attn.q_proj.weight"]);
        _selfKW = WhisperOps.EnsureF32(weights[$"{prefix}.self_attn.k_proj.weight"]);
        _selfVW = WhisperOps.EnsureF32(weights[$"{prefix}.self_attn.v_proj.weight"]);
        _selfOW = WhisperOps.EnsureF32(weights[$"{prefix}.self_attn.o_proj.weight"]);

        _postAttnLnW = WhisperOps.EnsureF32(weights[$"{prefix}.post_attention_layernorm.weight"]);
        _crossQW = WhisperOps.EnsureF32(weights[$"{prefix}.encoder_attn.q_proj.weight"]);
        _crossKW = WhisperOps.EnsureF32(weights[$"{prefix}.encoder_attn.k_proj.weight"]);
        _crossVW = WhisperOps.EnsureF32(weights[$"{prefix}.encoder_attn.v_proj.weight"]);
        _crossOW = WhisperOps.EnsureF32(weights[$"{prefix}.encoder_attn.o_proj.weight"]);

        _finalLnW = WhisperOps.EnsureF32(weights[$"{prefix}.final_layernorm.weight"]);
        _fc1W = WhisperOps.EnsureF32(weights[$"{prefix}.mlp.fc1.weight"]);  // [2 * inter, hidden]
        _fc1B = WhisperOps.EnsureF32(weights[$"{prefix}.mlp.fc1.bias"]);
        _fc2W = WhisperOps.EnsureF32(weights[$"{prefix}.mlp.fc2.weight"]);  // [hidden, inter]
        _fc2B = WhisperOps.EnsureF32(weights[$"{prefix}.mlp.fc2.bias"]);
    }

    public void PrecomputeCrossKv(IBackend backend, Tensor encHidden, Tensor outK, Tensor outV, int encSeqLen)
    {
        int d = _cfg.HiddenSize;
        Tensor k = WhisperOps.ProjectLinear(backend, encHidden, _crossKW!, bias: null, 1, encSeqLen, d, d);
        Tensor v = WhisperOps.ProjectLinear(backend, encHidden, _crossVW!, bias: null, 1, encSeqLen, d, d);
        WhisperOps.ReshapeToMultiHead4D(outK, k, 1, encSeqLen, _cfg.NumHeads, _cfg.HeadDim);
        WhisperOps.ReshapeToMultiHead4D(outV, v, 1, encSeqLen, _cfg.NumHeads, _cfg.HeadDim);
        k.Dispose(); v.Dispose();
    }

    public Tensor Forward(IBackend backend, Tensor hidden,
        Tensor selfK, Tensor selfV, Tensor crossK, Tensor crossV,
        int posStart, int newCount, float[] cos, float[] sin)
    {
        int d = _cfg.HiddenSize;
        int totalPos = posStart + newCount;
        TensorShape inShape = new(1, newCount, d);

        // === Self-attention (RoPE, causal, KV-cached) ===
        Tensor normed = new(inShape, DType.F32);
        MoonshineOps.LayerNormNoBias(normed, hidden, _inputLnW!, _cfg.LayerNormEps);
        Tensor q = WhisperOps.ProjectLinear(backend, normed, _selfQW!, bias: null, 1, newCount, d, d);
        Tensor k = WhisperOps.ProjectLinear(backend, normed, _selfKW!, bias: null, 1, newCount, d, d);
        Tensor v = WhisperOps.ProjectLinear(backend, normed, _selfVW!, bias: null, 1, newCount, d, d);
        normed.Dispose();

        TensorShape mhNew = new(1, _cfg.NumHeads, newCount, _cfg.HeadDim);
        Tensor qMh = new(mhNew, DType.F32);
        Tensor kMhNew = new(mhNew, DType.F32);
        Tensor vMhNew = new(mhNew, DType.F32);
        WhisperOps.ReshapeToMultiHead4D(qMh, q, 1, newCount, _cfg.NumHeads, _cfg.HeadDim);
        WhisperOps.ReshapeToMultiHead4D(kMhNew, k, 1, newCount, _cfg.NumHeads, _cfg.HeadDim);
        WhisperOps.ReshapeToMultiHead4D(vMhNew, v, 1, newCount, _cfg.NumHeads, _cfg.HeadDim);
        q.Dispose(); k.Dispose(); v.Dispose();

        // RoPE on Q and K using absolute positions [posStart .. posStart+newCount).
        RotaryEmbedding.ApplyInPlace(qMh, _cfg.NumHeads, newCount, _cfg.HeadDim, _cfg.RotaryDim, posStart, cos, sin);
        RotaryEmbedding.ApplyInPlace(kMhNew, _cfg.NumHeads, newCount, _cfg.HeadDim, _cfg.RotaryDim, posStart, cos, sin);

        // Append the rotated K/V into the cache slot [posStart..posStart+newCount).
        AppendToKvCache(selfK, kMhNew, posStart, _cfg.NumHeads, _cfg.MaxTextPositions, _cfg.HeadDim, newCount);
        AppendToKvCache(selfV, vMhNew, posStart, _cfg.NumHeads, _cfg.MaxTextPositions, _cfg.HeadDim, newCount);
        kMhNew.Dispose(); vMhNew.Dispose();

        // Slice the contiguous prefix [0..totalPos) for SDPA. SDPA expects a contiguous
        // S_kv inner block per head and our cache is over-allocated; materialize the slice.
        Tensor kCached = SliceKvPrefix(selfK, _cfg.NumHeads, totalPos, _cfg.MaxTextPositions, _cfg.HeadDim);
        Tensor vCached = SliceKvPrefix(selfV, _cfg.NumHeads, totalPos, _cfg.MaxTextPositions, _cfg.HeadDim);

        // Incremental causal mask: new position s (= posStart+s) sees cache [0..posStart+s].
        Tensor causalMask = BuildIncrementalCausalMask(posStart, newCount, totalPos);

        float scale = 1f / MathF.Sqrt(_cfg.HeadDim);
        Tensor attnOut = new(mhNew, DType.F32);
        backend.ScaledDotProductAttention(attnOut, qMh, kCached, vCached, causalMask, scale);
        qMh.Dispose(); kCached.Dispose(); vCached.Dispose(); causalMask.Dispose();

        Tensor merged = new(inShape, DType.F32);
        WhisperOps.ReshapeFromMultiHead4D(merged, attnOut, 1, newCount, _cfg.NumHeads, _cfg.HeadDim);
        attnOut.Dispose();
        Tensor selfProj = WhisperOps.ProjectLinear(backend, merged, _selfOW!, bias: null, 1, newCount, d, d);
        merged.Dispose();
        Tensor res1 = new(inShape, DType.F32);
        backend.Add(res1, hidden, selfProj);
        selfProj.Dispose();

        // === Cross-attention (no RoPE) ===
        Tensor normed2 = new(inShape, DType.F32);
        MoonshineOps.LayerNormNoBias(normed2, res1, _postAttnLnW!, _cfg.LayerNormEps);
        Tensor crossQ = WhisperOps.ProjectLinear(backend, normed2, _crossQW!, bias: null, 1, newCount, d, d);
        normed2.Dispose();
        Tensor crossQMh = new(mhNew, DType.F32);
        WhisperOps.ReshapeToMultiHead4D(crossQMh, crossQ, 1, newCount, _cfg.NumHeads, _cfg.HeadDim);
        crossQ.Dispose();

        Tensor crossAttn = new(mhNew, DType.F32);
        backend.ScaledDotProductAttention(crossAttn, crossQMh, crossK, crossV, mask: null, scale);
        crossQMh.Dispose();

        Tensor crossMerged = new(inShape, DType.F32);
        WhisperOps.ReshapeFromMultiHead4D(crossMerged, crossAttn, 1, newCount, _cfg.NumHeads, _cfg.HeadDim);
        crossAttn.Dispose();
        Tensor crossProj = WhisperOps.ProjectLinear(backend, crossMerged, _crossOW!, bias: null, 1, newCount, d, d);
        crossMerged.Dispose();
        Tensor res2 = new(inShape, DType.F32);
        backend.Add(res2, res1, crossProj);
        res1.Dispose(); crossProj.Dispose();

        // === SwiGLU MLP ===
        Tensor normed3 = new(inShape, DType.F32);
        MoonshineOps.LayerNormNoBias(normed3, res2, _finalLnW!, _cfg.LayerNormEps);

        // fc1 outputs 2 * intermediate features; split into [gate, up], apply SiLU(gate) * up.
        int interX2 = _cfg.IntermediateSize * 2;
        Tensor fc1Out = WhisperOps.ProjectLinear(backend, normed3, _fc1W!, _fc1B, 1, newCount, d, interX2);
        normed3.Dispose();
        Tensor gated = new(new TensorShape(1, newCount, _cfg.IntermediateSize), DType.F32);
        ApplySwiGlu(gated, fc1Out, newCount, _cfg.IntermediateSize);
        fc1Out.Dispose();
        Tensor fc2 = WhisperOps.ProjectLinear(backend, gated, _fc2W!, _fc2B, 1, newCount, _cfg.IntermediateSize, d);
        gated.Dispose();
        Tensor res3 = new(inShape, DType.F32);
        backend.Add(res3, res2, fc2);
        res2.Dispose(); fc2.Dispose();
        return res3;
    }

    /// <summary>Moonshine's SwiGLU gating: fc1 output is split into [up, gate] along the last
    /// dim — the FIRST half is the "up" projection, the SECOND half is the actual gate.
    /// Matches <c>transformers/models/moonshine/modeling_moonshine.py</c>:
    /// <code>
    ///   hidden_states = self.fc1(hidden_states)
    ///   hidden_states, gate = hidden_states.chunk(2, dim=-1)
    ///   hidden_states = self.activation_fn(gate) * hidden_states
    /// </code>
    /// Easy to get backwards (and we did, on the first run) — most Llama-style models use
    /// the opposite convention. Diff-debugged against the actual HF reference to nail down.</summary>
    private static void ApplySwiGlu(Tensor output, Tensor fc1Out, int seqLen, int inter)
    {
        float* o = (float*)output.DataPointer;
        float* x = (float*)fc1Out.DataPointer;
        int interX2 = inter * 2;
        for (int s = 0; s < seqLen; s++)
        {
            int srcBase = s * interX2;
            int dstBase = s * inter;
            for (int i = 0; i < inter; i++)
            {
                float up = x[srcBase + i];           // FIRST half = up
                float gate = x[srcBase + inter + i]; // SECOND half = gate
                float silu = gate / (1f + MathF.Exp(-gate));  // x * sigmoid(x)
                o[dstBase + i] = silu * up;
            }
        }
    }

    private static void AppendToKvCache(Tensor cache, Tensor src, int posStart, int heads, int maxPos, int headDim, int newCount)
    {
        float* c = (float*)cache.DataPointer;
        float* s = (float*)src.DataPointer;
        for (int h = 0; h < heads; h++)
            for (int p = 0; p < newCount; p++)
            {
                int sOff = (h * newCount + p) * headDim;
                int cOff = (h * maxPos + posStart + p) * headDim;
                for (int d = 0; d < headDim; d++) c[cOff + d] = s[sOff + d];
            }
    }

    private static Tensor SliceKvPrefix(Tensor cache, int heads, int totalPos, int maxPos, int headDim)
    {
        Tensor sliced = new(new TensorShape(1, heads, totalPos, headDim), DType.F32);
        float* c = (float*)cache.DataPointer;
        float* s = (float*)sliced.DataPointer;
        for (int h = 0; h < heads; h++)
        {
            int cBase = h * maxPos * headDim;
            int sBase = h * totalPos * headDim;
            for (int p = 0; p < totalPos; p++)
            {
                int cOff = cBase + p * headDim;
                int sOff = sBase + p * headDim;
                for (int d = 0; d < headDim; d++) s[sOff + d] = c[cOff + d];
            }
        }
        return sliced;
    }

    private static Tensor BuildIncrementalCausalMask(int posStart, int newCount, int totalPos)
    {
        Tensor mask = new(new TensorShape(1, 1, newCount, totalPos), DType.F32);
        float* p = (float*)mask.DataPointer;
        for (int s = 0; s < newCount; s++)
        {
            int allowedEnd = posStart + s;
            int rowOff = s * totalPos;
            for (int kp = 0; kp < totalPos; kp++)
                p[rowOff + kp] = kp <= allowedEnd ? 0f : float.NegativeInfinity;
        }
        return mask;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all =
        [
            _inputLnW, _selfQW, _selfKW, _selfVW, _selfOW,
            _postAttnLnW, _crossQW, _crossKW, _crossVW, _crossOW,
            _finalLnW, _fc1W, _fc1B, _fc2W, _fc2B,
        ];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }
}
