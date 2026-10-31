using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Ssm;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Models.Lfm3;

/// <summary>LFM3 decoder from HuggingFace-keyed weights: gated short-conv, full-attention and sliding-attention layers, each followed by a SwiGLU FFN, with sandwich norms. Carries its own conv windows and KV cache, so callers pass only the new tokens.</summary>
public sealed unsafe class Lfm3Model : IDisposable
{
    private const string Prefix = "model";

    private readonly IReadOnlyDictionary<string, Tensor> _weights;
    private readonly List<Tensor> _owned = [];
    private readonly Tensor?[] _convState;
    private readonly FixedKvCache _kv;
    private readonly AttentionLayer _attention;
    private readonly int _kernel;
    private int _pos;
    private int _disposed;

    public Lfm3Config Config { get; }

    /// <summary>Tokens consumed since the last <see cref="ResetState"/>.</summary>
    public int Position => _pos;

    /// <summary>Wraps <paramref name="weights"/>, which stay owned by the caller and must outlive the model. Norm and conv weights are widened to F32 once; projections are used as stored.</summary>
    /// <param name="maxSequenceLength">Longest sequence (prompt plus generation) the KV cache holds.</param>
    /// <exception cref="KeyNotFoundException">A tensor the config needs is missing.</exception>
    public Lfm3Model(IReadOnlyDictionary<string, Tensor> weights, Lfm3Config config, int maxSequenceLength)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(config);
        if (config.SandwichNorm is false)
        {
            throw new NotSupportedException("LFM3 without sandwich norms is not supported; the released checkpoints use them.");
        }
        Config = config;
        _kernel = config.ConvKernel;
        Dictionary<string, Tensor> staged = new(StringComparer.Ordinal);
        try
        {
            Stage(weights, staged, $"{Prefix}.embed_tokens.weight", asF32: false);
            Stage(weights, staged, $"{Prefix}.embedding_norm.weight", asF32: true);
            if (!config.TieWordEmbeddings)
            {
                Stage(weights, staged, "lm_head.weight", asF32: false);
            }
            for (int i = 0; i < config.NumLayers; i++)
            {
                StageLayer(weights, staged, i);
            }
            _weights = staged;
            _convState = new Tensor?[config.NumLayers];
            _kv = new FixedKvCache(config.NumLayers, 1, config.NumKvHeads, config.HeadDim, maxSequenceLength);
            _attention = new AttentionLayer(new AttentionLayerConfig
            {
                Heads = config.NumHeads, KvHeads = config.NumKvHeads, HeadDim = config.HeadDim, ComputeDType = DType.F32,
            });
            ResetState();
        }
        catch
        {
            foreach (Tensor t in _owned)
            {
                t.Dispose();
            }
            throw;
        }
    }

    /// <summary>Forgets the conv windows and the KV cache; call before an unrelated sequence.</summary>
    public void ResetState()
    {
        for (int i = 0; i < Config.NumLayers; i++)
        {
            _convState[i]?.Dispose();
            _convState[i] = Config.Layers[i] == Lfm3LayerKind.Conv ? new Tensor(new TensorShape(1, 1, _kernel - 1, Config.HiddenSize), DType.F32) : null;
            if (_convState[i] is Tensor window)
            {
                new Span<byte>((void*)window.DataPointer, (int)Tensor.ComputeByteSize(window.Shape, window.DType)).Clear();
            }
        }
        _kv.Reset();
        _pos = 0;
    }

    /// <summary>Runs the new tokens <paramref name="ids"/> through the stack and returns the next-token logits of the last one.</summary>
    public float[] ForwardLastLogits(IBackend backend, IReadOnlyList<int> ids)
    {
        Tensor normed = ForwardHidden(backend, ids);
        int seq = ids.Count, d = Config.HiddenSize, vocab = Config.VocabSize;
        Tensor last = new(new TensorShape(1, 1, 1, d), DType.F32);
        Tensor logits = new(new TensorShape(1, 1, vocab), DType.F32);
        try
        {
            backend.SliceTimeRange(last, normed.Reshape(new TensorShape(1, 1, seq, d)), seq - 1, 1);
            Tensor head = Config.TieWordEmbeddings ? W($"{Prefix}.embed_tokens.weight") : W("lm_head.weight");
            backend.Linear(logits, last, head, null);
            backend.Sync();
            return new ReadOnlySpan<float>((void*)logits.DataPointer, vocab).ToArray();
        }
        finally
        {
            normed.Dispose();
            last.Dispose();
            logits.Dispose();
        }
    }

    /// <summary>Runs the new tokens and returns every position's final-norm hidden state <c>[1, ids.Count, hidden]</c>; the caller owns it.</summary>
    public Tensor ForwardHidden(IBackend backend, IReadOnlyList<int> ids)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        int seq = ids.Count, d = Config.HiddenSize;
        if (seq == 0)
        {
            throw new ArgumentException("At least one token is required.", nameof(ids));
        }
        if (_pos + seq > _kv.MaxSequenceLength)
        {
            throw new InvalidOperationException($"{_pos + seq} tokens exceed the {_kv.MaxSequenceLength}-token KV cache.");
        }
        Tensor h = new(new TensorShape(1, seq, d), DType.F32);
        Tensor table = W($"{Prefix}.embed_tokens.weight");
        for (int s = 0; s < seq; s++)
        {
            if ((uint)ids[s] >= (uint)Config.VocabSize)
            {
                h.Dispose();
                throw new ArgumentOutOfRangeException(nameof(ids), $"Token id {ids[s]} is outside the {Config.VocabSize}-entry vocabulary.");
            }
            Qwen35Model.CopyEmbeddingRow(table, ids[s], d, (float*)h.DataPointer + (long)s * d);
        }
        for (int i = 0; i < Config.NumLayers; i++)
        {
            h = RunLayer(backend, h, i, seq);
        }
        Tensor normed = new(h.Shape, DType.F32);
        backend.RmsNorm(normed, h, W($"{Prefix}.embedding_norm.weight"), Config.NormEps);
        h.Dispose();
        _kv.AdvanceLength(seq);
        _pos += seq;
        return normed;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }
        foreach (Tensor? state in _convState)
        {
            state?.Dispose();
        }
        foreach (Tensor t in _owned)
        {
            t.Dispose();
        }
        _kv.Dispose();
    }

    private Tensor RunLayer(IBackend backend, Tensor h, int i, int seq)
    {
        string p = $"{Prefix}.layers.{i}";
        TensorShape shape = h.Shape;
        Tensor xn = new(shape, DType.F32);
        backend.RmsNorm(xn, h, W($"{p}.operator_norm.weight"), Config.NormEps);
        Tensor mixed = Config.Layers[i] == Lfm3LayerKind.Conv ? ConvMixer(backend, xn, i, seq) : AttentionMixer(backend, xn, i, seq);
        xn.Dispose();
        h = AddNormed(backend, h, mixed, W($"{p}.post_operator_norm.weight"));

        Tensor ffnIn = new(shape, DType.F32);
        backend.RmsNorm(ffnIn, h, W($"{p}.ffn_norm.weight"), Config.NormEps);
        int inner = Config.IntermediateSize;
        Tensor gate = new(new TensorShape(1, seq, inner), DType.F32);
        Tensor up = new(gate.Shape, DType.F32);
        backend.Linear(gate, ffnIn, W($"{p}.feed_forward.w1.weight"), null);
        backend.Linear(up, ffnIn, W($"{p}.feed_forward.w3.weight"), null);
        ffnIn.Dispose();
        backend.Silu(gate, gate);
        backend.Mul(gate, gate, up);
        up.Dispose();
        Tensor down = new(shape, DType.F32);
        backend.Linear(down, gate, W($"{p}.feed_forward.w2.weight"), null);
        gate.Dispose();
        return AddNormed(backend, h, down, W($"{p}.post_ffn_norm.weight"));
    }

    /// <summary>Returns <c>residual + RmsNorm(branch)</c> and disposes both operands.</summary>
    private Tensor AddNormed(IBackend backend, Tensor residual, Tensor branch, Tensor norm)
    {
        Tensor normed = new(branch.Shape, DType.F32);
        backend.RmsNorm(normed, branch, norm, Config.NormEps);
        branch.Dispose();
        Tensor sum = new(residual.Shape, DType.F32);
        backend.Add(sum, residual, normed);
        residual.Dispose();
        normed.Dispose();
        return sum;
    }

    /// <summary>The gated short convolution: <c>out_proj(C * conv(B * x))</c> over a window carried from the previous call.</summary>
    private Tensor ConvMixer(IBackend backend, Tensor xn, int i, int seq)
    {
        string p = $"{Prefix}.layers.{i}.conv";
        int d = Config.HiddenSize, window = _kernel - 1, total = window + seq;
        TensorShape token = new(1, seq, d);
        Tensor proj = new(new TensorShape(1, seq, 3 * d), DType.F32);
        backend.Linear(proj, xn, W($"{p}.in_proj.weight"), null);
        Tensor b = new(token, DType.F32), c = new(token, DType.F32), x = new(token, DType.F32);
        backend.SliceLastDim(b, proj, 0);
        backend.SliceLastDim(c, proj, d);
        backend.SliceLastDim(x, proj, 2 * d);
        proj.Dispose();
        backend.Mul(b, b, x);
        x.Dispose();

        Tensor state = _convState[i]!;
        Tensor joined = new(new TensorShape(1, total, d), DType.F32);
        backend.Concat(joined, [state.Reshape(new TensorShape(1, window, d)), b], 1);
        b.Dispose();
        backend.SliceTimeRange(state, joined.Reshape(new TensorShape(1, 1, total, d)), seq, window);

        Tensor channelsFirst = new(new TensorShape(1, d, total), DType.F32);
        backend.Transpose2D(channelsFirst, joined, total, d);
        joined.Dispose();
        Tensor convolved = new(new TensorShape(1, d, seq), DType.F32);
        backend.Conv1d(convolved, channelsFirst, W($"{p}.conv.weight"), null, stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: d);
        channelsFirst.Dispose();
        Tensor tokenMajor = new(token, DType.F32);
        backend.Transpose2D(tokenMajor, convolved, d, seq);
        convolved.Dispose();
        backend.Mul(tokenMajor, tokenMajor, c);
        c.Dispose();
        Tensor output = new(token, DType.F32);
        backend.Linear(output, tokenMajor, W($"{p}.out_proj.weight"), null);
        tokenMajor.Dispose();
        return output;
    }

    /// <summary>Grouped-query attention with per-head QK RMSNorm, zero-tail partial RoPE, an optional sliding window and a sigmoid output gate per head.</summary>
    private Tensor AttentionMixer(IBackend backend, Tensor xn, int i, int seq)
    {
        string p = $"{Prefix}.layers.{i}.self_attn";
        int hq = Config.NumHeads, hkv = Config.NumKvHeads, hd = Config.HeadDim;
        Tensor q = Project(backend, xn, W($"{p}.q_proj.weight"), seq, hq, hd, W($"{p}.q_norm.weight"));
        Tensor k = Project(backend, xn, W($"{p}.k_proj.weight"), seq, hkv, hd, W($"{p}.k_norm.weight"));
        Tensor v = Project(backend, xn, W($"{p}.v_proj.weight"), seq, hkv, hd, null);
        Tensor gateLogits = new(new TensorShape(1, seq, hq), DType.F32);
        backend.Linear(gateLogits, xn, W($"{p}.gate_proj.weight"), null);

        Tensor cos = new(new TensorShape(1, seq, hd), DType.F32);
        Tensor sin = new(cos.Shape, DType.F32);
        BuildZeroTailRope(cos, sin, seq, i);
        backend.ApplyRopeSingle(q, cos, sin);
        backend.ApplyRopeSingle(k, cos, sin);
        cos.Dispose();
        sin.Dispose();

        Tensor qMh = new(new TensorShape(1, hq, seq, hd), DType.F32);
        Tensor kMh = new(new TensorShape(1, hkv, seq, hd), DType.F32);
        Tensor vMh = new(kMh.Shape, DType.F32);
        backend.Permute0213(qMh, q, seq, hq, hd);
        backend.Permute0213(kMh, k, seq, hkv, hd);
        backend.Permute0213(vMh, v, seq, hkv, hd);
        q.Dispose();
        k.Dispose();
        v.Dispose();
        KvAttentionCall call = new()
        {
            KvLength = _pos + seq,
            QueryOffset = _pos,
            SlidingWindow = Config.Layers[i] == Lfm3LayerKind.SlidingAttention ? Config.SlidingWindow : 0,
        };
        Tensor attended = _attention.AttendOverKvHeadMajor(backend, qMh, kMh, vMh, _kv, i, i, call);

        Tensor flat = new(new TensorShape(1, seq, hq * hd), DType.F32);
        backend.Permute0213(flat, attended, hq, seq, hd);
        attended.Dispose();
        // The shared head gate multiplies by 2*sigmoid; halving restores sigmoid.
        backend.Ltx2HeadGate(flat, gateLogits, seq, hq, hd);
        gateLogits.Dispose();
        Tensor gated = new(flat.Shape, DType.F32);
        backend.Scale(gated, flat, 0.5f);
        flat.Dispose();
        Tensor output = new(new TensorShape(1, seq, Config.HiddenSize), DType.F32);
        backend.Linear(output, gated, W($"{p}.o_proj.weight"), null);
        gated.Dispose();
        return output;
    }

    /// <summary>One projection reshaped to <c>[1, seq, heads, headDim]</c> and, when <paramref name="norm"/> is set, RMS-normalized per head.</summary>
    private Tensor Project(IBackend backend, Tensor xn, Tensor weight, int seq, int heads, int hd, Tensor? norm)
    {
        Tensor perHead = new(new TensorShape(1, seq, heads, hd), DType.F32);
        backend.Linear(perHead, xn, weight, null);
        if (norm is null)
        {
            return perHead;
        }
        Tensor normed = new(perHead.Shape, DType.F32);
        backend.RmsNorm(normed, perHead, norm, Config.NormEps);
        perHead.Dispose();
        return normed;
    }

    /// <summary>Full-width rotate-half tables whose frequencies follow the whole head dimension but are zero past the rotated prefix, so those pairs pass through unchanged.</summary>
    private void BuildZeroTailRope(Tensor cos, Tensor sin, int seq, int layer)
    {
        int hd = Config.HeadDim, half = hd / 2;
        Lfm3RopeParameters rope = Config.RopeOf(layer);
        RopeTables.BuildRope(cos, sin, seq, _pos, hd, hd, rope.Theta, RopeScaling.None);
        float* c = (float*)cos.DataPointer;
        float* s = (float*)sin.DataPointer;
        for (int row = 0; row < seq; row++)
        {
            for (int pair = rope.RotatedPairs(hd); pair < half; pair++)
            {
                long at = (long)row * hd + pair;
                c[at] = c[at + half] = 1f;
                s[at] = s[at + half] = 0f;
            }
        }
    }

    private Tensor W(string key) => _weights[key];

    private void StageLayer(IReadOnlyDictionary<string, Tensor> src, Dictionary<string, Tensor> dst, int i)
    {
        string p = $"{Prefix}.layers.{i}";
        foreach (string norm in (string[])["operator_norm", "ffn_norm", "post_operator_norm", "post_ffn_norm"])
        {
            Stage(src, dst, $"{p}.{norm}.weight", asF32: true);
        }
        foreach (string proj in (string[])["w1", "w2", "w3"])
        {
            Stage(src, dst, $"{p}.feed_forward.{proj}.weight", asF32: false);
        }
        if (Config.Layers[i] == Lfm3LayerKind.Conv)
        {
            Stage(src, dst, $"{p}.conv.conv.weight", asF32: true);
            Stage(src, dst, $"{p}.conv.in_proj.weight", asF32: false);
            Stage(src, dst, $"{p}.conv.out_proj.weight", asF32: false);
            return;
        }
        foreach (string proj in (string[])["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"])
        {
            Stage(src, dst, $"{p}.self_attn.{proj}.weight", asF32: false);
        }
        Stage(src, dst, $"{p}.self_attn.q_norm.weight", asF32: true);
        Stage(src, dst, $"{p}.self_attn.k_norm.weight", asF32: true);
    }

    private void Stage(IReadOnlyDictionary<string, Tensor> src, Dictionary<string, Tensor> dst, string key, bool asF32)
    {
        if (!src.TryGetValue(key, out Tensor? t))
        {
            throw new KeyNotFoundException($"The LFM3 checkpoint has no tensor '{key}'.");
        }
        if (asF32 && t.DType != DType.F32)
        {
            t = t.CastTo(DType.F32);
            _owned.Add(t);
        }
        dst[key] = t;
    }
}
