using HartsyInference.Core.Backends;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Gguf;

namespace HartsyInference.LLM.Ssm;

/// <summary>Qwen3.5 (<c>qwen35</c> GGUF arch) — a hybrid decoder, NOT a transformer or a pure recurrent model:
/// most layers are Gated DeltaNet (a delta-rule linear attention: causal Conv1d over a fused QKV projection,
/// then a per-head recurrent state update <c>S_t = α_t·S_{t-1} + β_t(v_t - S_{t-1}k_t)k_t^T</c>, readout
/// <c>o_t = S_t·q_t</c>), interleaved every <see cref="FullAttentionInterval"/>-th layer with a REGULAR causal
/// self-attention layer (GQA + partial RoPE + KV cache, Qwen3-style QK-norm, plus a fused query+gate projection
/// unique to this arch). Text-only: the GGUF's 4-section M-RoPE degenerates to a single partial-rotary RoPE
/// (<see cref="RotaryDim"/> = sum of the 4 sections × 2) since every section gets the same position for
/// non-multimodal input — no M-RoPE machinery needed here. Ported from llama.cpp's
/// <c>src/models/{qwen35.cpp, delta-net-base.cpp}</c> (no local reference model existed at implementation
/// time). Linear projections run through <see cref="IBackend"/>; the conv/delta-rule/gated-norm run host-side,
/// matching every other recurrent model in this namespace (<see cref="Mamba2Model"/> et al.).</summary>
public sealed unsafe class Qwen35Model : IDisposable, ISsmModel
{
    private readonly GgufModelLoader.LoadedGgufModel _handle;
    private readonly IReadOnlyDictionary<string, Tensor> _w;
    private readonly bool[] _isLinear;   // per-layer: true = Gated DeltaNet, false = regular attention

    // Gated DeltaNet per-layer recurrent state (null for regular-attention layers). Carried across calls so
    // ForwardLastLogits can be fed only the NEW tokens each call, same contract as every other ISsmModel.
    // _convHistory[i]: last (ConvKernel-1) pre-conv rows of the fused QKV projection (causal conv left context).
    // _deltaState[i]: the [S_v, S_v] per-head state matrix, flattened [NumVHeads * S_v * S_v].
    private readonly float[]?[] _convHistory;
    private readonly float[]?[] _deltaState;
    // Regular-attention layers' KV cache (unused slots for linear-attn layers are simply never touched).
    // ISsmModel has no "expected max length" input, so this is a practical fixed cap, not a general solution.
    private const int MaxAttnSeqLen = 8192;
    private readonly FixedKvCache _kvCache;
    private int _pos;
    private int _disposed;

    public GgufMetadata Metadata => _handle.Metadata;
    public int DModel { get; }
    public int NumLayers { get; }
    public int VocabSize { get; }
    public float Eps { get; }
    public int FullAttentionInterval { get; }

    // Regular-attention layer dims.
    public int NumHeads { get; }
    public int NumKvHeads { get; }
    public int AttnHeadDim { get; }      // n_embd_head_k / n_embd_head_v
    public int RotaryDim { get; }        // sum(rope.dimension_sections) * 2 — text-only M-RoPE degenerate case
    public float RopeTheta { get; }

    // Gated DeltaNet layer dims.
    public int ConvKernel { get; }
    public int HeadKDim { get; }         // ssm.state_size (S_k == S_v for every known qwen35 checkpoint)
    public int HeadVDim { get; }
    public int NumKHeads { get; }        // ssm.group_count
    public int NumVHeads { get; }        // ssm.time_step_rank
    private int KeyDim => HeadKDim * NumKHeads;
    private int ValueDim => HeadVDim * NumVHeads;
    private int ConvDim => KeyDim * 2 + ValueDim;
    private int KHeadsPerVHead => NumVHeads / NumKHeads;   // repeat factor when NumKHeads < NumVHeads

    private Qwen35Model(GgufModelLoader.LoadedGgufModel handle, IReadOnlyDictionary<string, Tensor> w,
        int dModel, int layers, int vocab, float eps, int fullAttnInterval, bool[]? recrOverride,
        int numHeads, int numKvHeads, int attnHeadDim, int rotaryDim, float ropeTheta,
        int convK, int headKDim, int headVDim, int numKHeads, int numVHeads)
    {
        _handle = handle; _w = w;
        DModel = dModel; NumLayers = layers; VocabSize = vocab; Eps = eps <= 0f ? 1e-6f : eps;
        FullAttentionInterval = fullAttnInterval;
        NumHeads = numHeads; NumKvHeads = numKvHeads; AttnHeadDim = attnHeadDim;
        RotaryDim = rotaryDim; RopeTheta = ropeTheta;
        ConvKernel = convK; HeadKDim = headKDim; HeadVDim = headVDim; NumKHeads = numKHeads; NumVHeads = numVHeads;

        _isLinear = new bool[layers];
        for (int i = 0; i < layers; i++)
            _isLinear[i] = recrOverride is { Length: > 0 } ? recrOverride[i] : (i + 1) % fullAttnInterval != 0;

        _convHistory = new float[layers][];
        _deltaState = new float[layers][];
        for (int i = 0; i < layers; i++)
        {
            if (!_isLinear[i]) continue;
            _convHistory[i] = new float[(convK - 1) * ConvDim];
            _deltaState[i] = new float[NumVHeads * HeadVDim * HeadVDim];
        }
        _kvCache = new FixedKvCache(layers, 1, numKvHeads, attnHeadDim, MaxAttnSeqLen);
    }

    /// <summary>Zeroes every layer's carried state (Gated DeltaNet matrices + conv history) and the KV cache —
    /// call before the first <see cref="ForwardLastLogits"/> of a new generation.</summary>
    public void ResetState()
    {
        foreach (float[]? h in _convHistory) if (h is not null) Array.Clear(h);
        foreach (float[]? s in _deltaState) if (s is not null) Array.Clear(s);
        _kvCache.Reset();
        _pos = 0;
    }

    /// <summary>Shape-relabels an nn.Linear weight from GGUF's [in, out] to [out, in] (no data move).</summary>
    private static Tensor Relabel(Tensor t)
    {
        Tensor outp = new(new TensorShape((int)t.Shape[1], (int)t.Shape[0]), DType.F32);
        Buffer.MemoryCopy((void*)t.DataPointer, (void*)outp.DataPointer, outp.ElementCount * 4, t.ElementCount * 4);
        return outp;
    }

    public static Qwen35Model Load(string ggufPath)
    {
        (Dictionary<string, Tensor> w, GgufModelLoader.LoadedGgufModel handle) = GgufModelLoader.LoadDequantized(ggufPath, DType.F32);
        try
        {
            // Relabel every 2D nn.Linear weight to [out, in]; ssm_conv1d/ssm_a/ssm_dt.bias/ssm_norm/norms are
            // read directly with explicit indexing (same split as Mamba2Model.Load) and left alone.
            foreach (string key in w.Keys.ToList())
            {
                if (!key.EndsWith(".weight", StringComparison.Ordinal) || w[key].Shape.Rank != 2) continue;
                if (key.Contains("ssm_conv1d")) continue;
                w[key] = Relabel(w[key]);
            }
            GgufMetadata m = handle.Metadata;
            const string arch = "qwen35";
            int dModel = (int)m.GetUInt32($"{arch}.embedding_length");
            int layers = (int)m.GetUInt32($"{arch}.block_count");
            int vocab = (int)w["token_embd.weight"].Shape[0];   // relabeled to [vocab, dModel]
            float eps = m.GetFloat32($"{arch}.attention.layer_norm_rms_epsilon", 1e-6f);
            int fullAttnInterval = (int)m.GetUInt32($"{arch}.full_attention_interval", 4u);
            bool[]? recrOverride = m.GetBoolArray($"{arch}.attention.recurrent_layers");

            int numHeads = (int)m.GetUInt32($"{arch}.attention.head_count");
            int numKvHeads = (int)m.GetUInt32($"{arch}.attention.head_count_kv", (uint)numHeads);
            int attnHeadDim = (int)m.GetUInt32($"{arch}.attention.key_length", 0u);
            if (attnHeadDim == 0) attnHeadDim = dModel / numHeads;
            // Text-only M-RoPE degenerate case: every section gets the same position, so the 4-section split
            // collapses into one ordinary partial-rotary table spanning rope.dimension_count dims.
            int rotaryDim = (int)m.GetUInt32($"{arch}.rope.dimension_count", 0u);
            float ropeTheta = m.GetFloat32($"{arch}.rope.freq_base", 10_000_000f);

            int convK = (int)m.GetUInt32($"{arch}.ssm.conv_kernel");
            int headKvDim = (int)m.GetUInt32($"{arch}.ssm.state_size");   // S_k == S_v for every known checkpoint
            int numKHeads = (int)m.GetUInt32($"{arch}.ssm.group_count");
            int numVHeads = (int)m.GetUInt32($"{arch}.ssm.time_step_rank");
            int dInner = (int)m.GetUInt32($"{arch}.ssm.inner_size");
            int headVDim = dInner / numVHeads;

            return new Qwen35Model(handle, w, dModel, layers, vocab, eps, fullAttnInterval, recrOverride,
                numHeads, numKvHeads, attnHeadDim, rotaryDim, ropeTheta,
                convK, headKvDim, headVDim, numKHeads, numVHeads);
        }
        catch { handle.Dispose(); throw; }
    }

    private Tensor W(string key) => _w[key];
    private bool Has(string key) => _w.ContainsKey(key);

    /// <inheritdoc/>
    public float[] ForwardLastLogits(IBackend backend, IReadOnlyList<int> ids)
    {
        int seq = ids.Count, d = DModel;
        Tensor h = new(new TensorShape(1, seq, d), DType.F32);
        float* hp = (float*)h.DataPointer;
        float* emb = (float*)W("token_embd.weight").DataPointer;
        for (int s = 0; s < seq; s++)
            Buffer.MemoryCopy(emb + (long)ids[s] * d, hp + (long)s * d, (long)d * 4, (long)d * 4);

        for (int i = 0; i < NumLayers; i++)
        {
            Tensor next = _isLinear[i] ? BlockLinear(backend, h, i, seq) : BlockAttn(backend, h, i, seq);
            h.Dispose(); h = next;
        }
        _pos += seq;

        Tensor normed = new(new TensorShape(1, seq, d), DType.F32);
        backend.RmsNorm(normed, h, W("output_norm.weight"), Eps);
        h.Dispose();
        backend.Sync();
        using Tensor last = new(new TensorShape(1, 1, d), DType.F32);
        Buffer.MemoryCopy((byte*)normed.DataPointer + (long)(seq - 1) * d * 4, (void*)last.DataPointer, (long)d * 4, (long)d * 4);
        normed.Dispose();
        using Tensor logits = new(new TensorShape(1, 1, VocabSize), DType.F32);
        Tensor headW = Has("output.weight") ? W("output.weight") : W("token_embd.weight");   // tied by default
        backend.Linear(logits, last, headW, null);
        backend.Sync();
        float[] outv = new float[VocabSize];
        fixed (float* o = outv) Buffer.MemoryCopy((void*)logits.DataPointer, o, (long)VocabSize * 4, (long)VocabSize * 4);
        return outv;
    }

    /// <summary>Regular causal self-attention layer: GQA + partial RoPE + KV cache, Qwen3-style per-head QK-norm,
    /// and a fused query+gate projection (wq outputs <c>[query | gate]</c>, double width — the gate sigmoid-gates
    /// the attention output before <c>wo</c>, unique to this arch's full-attention layers).</summary>
    private Tensor BlockAttn(IBackend backend, Tensor hIn, int i, int seq)
    {
        int d = DModel, hq = NumHeads, hkv = NumKvHeads, hd = AttnHeadDim, group = hq / hkv;
        string p = $"blk.{i}";
        TensorShape flat = new(1, seq, d);

        Tensor xn = new(flat, DType.F32);
        backend.RmsNorm(xn, hIn, W($"{p}.attn_norm.weight"), Eps);

        // wq outputs [query | gate] fused, each hq*hd wide (double width); each head's row is
        // [query_0..query_{hd-1}, gate_0..gate_{hd-1}] — a strided de-interleave, not a flat slice — done
        // host-side since it's a one-time small reshuffle per call, not a hot loop.
        Tensor qgFull = new(new TensorShape(1, seq, hq * hd * 2), DType.F32);
        backend.Linear(qgFull, xn, W($"{p}.attn_q.weight"), null);
        Tensor q = new(new TensorShape(1, seq, hq, hd), DType.F32);
        Tensor gate = new(new TensorShape(1, seq, hq, hd), DType.F32);
        SplitInterleavedQueryGate(qgFull, q, gate, seq, hq, hd);
        qgFull.Dispose();

        Tensor qN = new(q.Shape, DType.F32);
        backend.RmsNorm(qN, q, W($"{p}.attn_q_norm.weight"), Eps);
        q.Dispose();

        Tensor k = new(new TensorShape(1, seq, hkv, hd), DType.F32);
        backend.Linear(k, xn, W($"{p}.attn_k.weight"), null);
        Tensor kN = new(k.Shape, DType.F32);
        backend.RmsNorm(kN, k, W($"{p}.attn_k_norm.weight"), Eps);
        k.Dispose();

        Tensor v = new(new TensorShape(1, seq, hkv, hd), DType.F32);
        backend.Linear(v, xn, W($"{p}.attn_v.weight"), null);
        xn.Dispose();

        Tensor cos = new(new TensorShape(1, seq, hd), DType.F32);
        Tensor sin = new(new TensorShape(1, seq, hd), DType.F32);
        BuildRope(cos, sin, seq, _pos, hd, RotaryDim, RopeTheta);
        backend.ApplyRopeSingle(qN, cos, sin, RotaryDim);
        backend.ApplyRopeSingle(kN, cos, sin, RotaryDim);
        cos.Dispose(); sin.Dispose();

        Tensor qMh = new(new TensorShape(1, hq, seq, hd), DType.F32);
        Tensor kMh = new(new TensorShape(1, hkv, seq, hd), DType.F32);
        Tensor vMh = new(new TensorShape(1, hkv, seq, hd), DType.F32);
        backend.Permute0213(qMh, qN, seq, hq, hd);
        backend.Permute0213(kMh, kN, seq, hkv, hd);
        backend.Permute0213(vMh, v, seq, hkv, hd);
        qN.Dispose(); kN.Dispose(); v.Dispose();

        _kvCache.AppendStep(backend, i, kMh, vMh);
        kMh.Dispose(); vMh.Dispose();
        int kvLen = _pos + seq;
        Tensor attn = new(new TensorShape(1, hq, seq, hd), DType.F32);
        backend.FlashAttention(attn, qMh, _kvCache.KeyPrefix(i), _kvCache.ValuePrefix(i), kvLen, group, causal: true, qOffset: _pos, 1f / MathF.Sqrt(hd));
        qMh.Dispose();

        Tensor attnFlat = new(new TensorShape(1, seq, hq * hd), DType.F32);
        backend.Permute0213(attnFlat, attn, hq, seq, hd);
        attn.Dispose();

        Tensor gateFlat = gate.Reshape(new TensorShape(1, seq, hq * hd));   // view, no copy — same layout already
        Tensor gateSig = new(gateFlat.Shape, DType.F32);
        backend.Sigmoid(gateSig, gateFlat);
        gate.Dispose();
        Tensor gated = new(attnFlat.Shape, DType.F32);
        backend.Mul(gated, attnFlat, gateSig);
        attnFlat.Dispose(); gateSig.Dispose();

        Tensor attnOut = new(flat, DType.F32);
        backend.Linear(attnOut, gated, W($"{p}.attn_output.weight"), null);
        gated.Dispose();
        Tensor afterAttn = new(flat, DType.F32);
        backend.Add(afterAttn, hIn, attnOut);
        attnOut.Dispose();

        Tensor preFfn = new(flat, DType.F32);
        backend.RmsNorm(preFfn, afterAttn, W($"{p}.post_attention_norm.weight"), Eps);
        Tensor ffnOut = DenseSwiGlu(backend, preFfn, p, seq);
        preFfn.Dispose();
        Tensor result = new(flat, DType.F32);
        backend.Add(result, afterAttn, ffnOut);
        afterAttn.Dispose(); ffnOut.Dispose();
        return result;
    }

    /// <summary>Splits the fused <c>[query | gate]</c> projection into two <c>[1,seq,hq,hd]</c> tensors — each
    /// head's output row is <c>[query_0..query_{hd-1}, gate_0..gate_{hd-1}]</c> (query first, gate second),
    /// so this is a per-head strided de-interleave, not a contiguous slice.</summary>
    private static void SplitInterleavedQueryGate(Tensor qgFull, Tensor q, Tensor gate, int seq, int hq, int hd)
    {
        float* src = (float*)qgFull.DataPointer;
        float* qp = (float*)q.DataPointer;
        float* gp = (float*)gate.DataPointer;
        int headStride = hd * 2;
        for (int s = 0; s < seq; s++)
        {
            long rowBase = (long)s * hq * headStride;
            long outBase = (long)s * hq * hd;
            for (int hh = 0; hh < hq; hh++)
            {
                float* srcHead = src + rowBase + hh * headStride;
                Buffer.MemoryCopy(srcHead, qp + outBase + (long)hh * hd, hd * 4, hd * 4);
                Buffer.MemoryCopy(srcHead + hd, gp + outBase + (long)hh * hd, hd * 4, hd * 4);
            }
        }
    }

    private static void BuildRope(Tensor cos, Tensor sin, int t, int posStart, int headDim, int rotaryDim, float theta)
    {
        int rdim = rotaryDim > 0 && rotaryDim < headDim ? rotaryDim : headDim;
        int half = rdim / 2;
        (double[] invFreq, double mscale) = RopeFrequencyBuilder.Build(rdim, theta, RopeScaling.None, posStart + t);
        float* pc = (float*)cos.DataPointer;
        float* ps = (float*)sin.DataPointer;
        for (int s = 0; s < t; s++)
        {
            int pos = posStart + s;
            long baseOff = (long)s * headDim;
            for (int i = 0; i < half; i++)
            {
                double angle = pos * invFreq[i];
                float c = (float)(Math.Cos(angle) * mscale);
                float si = (float)(Math.Sin(angle) * mscale);
                pc[baseOff + i] = c; pc[baseOff + i + half] = c;
                ps[baseOff + i] = si; ps[baseOff + i + half] = si;
            }
        }
    }

    /// <summary>Gated DeltaNet layer: fused QKV projection → causal depthwise Conv1d+SiLU → per-head L2-norm on
    /// Q/K → sequential delta-rule recurrence per token (<c>S_t = α_t·S_{t-1} + β_t(v_t − S_{t-1}k_t)k_t^T</c>,
    /// <c>o_t = S_t·q_t</c>) → gated RMSNorm(o, silu(z)) → output projection + residual. Linear projections and
    /// norms run through <see cref="IBackend"/>; the conv/recurrence/gated-norm run host-side (same split as
    /// <see cref="Mamba2Model"/>'s SSD scan).</summary>
    private Tensor BlockLinear(IBackend backend, Tensor hIn, int i, int seq)
    {
        int d = DModel;
        string p = $"blk.{i}";
        TensorShape flat = new(1, seq, d);

        Tensor xn = new(flat, DType.F32);
        backend.RmsNorm(xn, hIn, W($"{p}.attn_norm.weight"), Eps);

        Tensor qkvMixed = new(new TensorShape(1, seq, ConvDim), DType.F32);
        backend.Linear(qkvMixed, xn, W($"{p}.attn_qkv.weight"), null);
        Tensor z = new(new TensorShape(1, seq, ValueDim), DType.F32);
        backend.Linear(z, xn, W($"{p}.attn_gate.weight"), null);
        Tensor betaRaw = new(new TensorShape(1, seq, NumVHeads), DType.F32);
        backend.Linear(betaRaw, xn, W($"{p}.ssm_beta.weight"), null);
        Tensor beta = new(betaRaw.Shape, DType.F32);
        backend.Sigmoid(beta, betaRaw);
        betaRaw.Dispose();
        Tensor alphaRaw = new(new TensorShape(1, seq, NumVHeads), DType.F32);
        backend.Linear(alphaRaw, xn, W($"{p}.ssm_alpha.weight"), null);
        xn.Dispose();
        backend.Sync();

        int hk = NumKHeads, hv = NumVHeads, sk = HeadKDim, sv = HeadVDim, convDim = ConvDim, k = ConvKernel;
        int keyDim = KeyDim, valueDim = ValueDim, repeat = KHeadsPerVHead;

        // gate[s,h] = softplus(alpha + dt_bias) * ssm_a  (ssm_a is pre-baked -exp(A_log): negative, so exp(gate)
        // in (0,1] — a proper input-dependent decay factor, same convention as Mamba2's ssm_a).
        float* alphaP = (float*)alphaRaw.DataPointer;
        float* dtBiasP = (float*)W($"{p}.ssm_dt.bias").DataPointer;
        float* ssmAP = (float*)W($"{p}.ssm_a").DataPointer;
        float[] gateArr = new float[seq * hv];
        for (int s = 0; s < seq; s++)
            for (int hh = 0; hh < hv; hh++)
            {
                float av = alphaP[s * hv + hh] + dtBiasP[hh];
                float softplus = av > 20f ? av : MathF.Log(1f + MathF.Exp(av));
                gateArr[s * hv + hh] = softplus * ssmAP[hh];
            }
        alphaRaw.Dispose();

        float[] betaArr = new float[seq * hv];
        fixed (float* bp = betaArr) Buffer.MemoryCopy((float*)beta.DataPointer, bp, (long)seq * hv * 4, (long)seq * hv * 4);
        beta.Dispose();

        // Causal depthwise Conv1d over qkv_mixed (kernel k, left context = the (k-1) rows carried from the
        // previous call — zero on the very first call after ResetState) + SiLU.
        float* qkvP = (float*)qkvMixed.DataPointer;
        float[] history = _convHistory[i]!;
        float* convW = (float*)W($"{p}.ssm_conv1d.weight").DataPointer;   // [k, convDim], no bias on this arch
        float[] xc = new float[seq * convDim];
        for (int s = 0; s < seq; s++)
            for (int c = 0; c < convDim; c++)
            {
                float acc = 0f;
                for (int j = 0; j < k; j++)
                {
                    int combined = s + j;
                    acc += combined < k - 1
                        ? convW[c * k + j] * history[combined * convDim + c]
                        : convW[c * k + j] * qkvP[(long)(combined - (k - 1)) * convDim + c];
                }
                xc[s * convDim + c] = acc / (1f + MathF.Exp(-acc));   // SiLU
            }
        if (k > 1)
        {
            int carry = Math.Min(k - 1, seq);
            for (int r = 0; r < k - 1 - carry; r++)
                Array.Copy(history, (r + carry) * convDim, history, r * convDim, convDim);
            for (int r = 0; r < carry; r++)
                fixed (float* hp = history)
                    Buffer.MemoryCopy(qkvP + (long)(seq - carry + r) * convDim, hp + (long)(k - 1 - carry + r) * convDim, convDim * 4L, convDim * 4L);
        }
        qkvMixed.Dispose();

        // Split conv output into q [hk*sk], k [hk*sk], v [hv*sv] and L2-normalize q/k per head (no learned scale).
        float[] qArr = new float[seq * hk * sk], kArr = new float[seq * hk * sk], vArr = new float[seq * hv * sv];
        for (int s = 0; s < seq; s++)
        {
            long b = (long)s * convDim;
            for (int c = 0; c < keyDim; c++) qArr[s * keyDim + c] = xc[b + c];
            for (int c = 0; c < keyDim; c++) kArr[s * keyDim + c] = xc[b + keyDim + c];
            for (int c = 0; c < valueDim; c++) vArr[s * valueDim + c] = xc[b + 2 * keyDim + c];
        }
        L2NormalizePerHead(qArr, seq, hk, sk, Eps);
        L2NormalizePerHead(kArr, seq, hk, sk, Eps);

        // q *= 1/sqrt(S_k) — applied inside llama.cpp's build_delta_net_autoregressive, right before the
        // recurrence (after L2-norm, not folded into it). Missing this scales the readout ~sqrt(HeadKDim)x too
        // large, compounding across every linear-attn layer.
        float qScale = 1f / MathF.Sqrt(sk);
        for (int idx = 0; idx < qArr.Length; idx++) qArr[idx] *= qScale;

        // Delta-rule recurrence, sequential per token. State is a persistent [hv, sv, sk] tensor per layer
        // (row = value dim, col = key dim — sv==sk on every known qwen35 checkpoint, but kept general).
        float[] state = _deltaState[i]!;
        float[] o = new float[seq * valueDim];
        Span<float> skRow = stackalloc float[sv];
        for (int s = 0; s < seq; s++)
        {
            for (int h = 0; h < hv; h++)
            {
                int kh = h / repeat;
                long qOff = (long)s * keyDim + kh * sk;
                long vOff = (long)s * valueDim + h * sv;
                float gExp = MathF.Exp(gateArr[s * hv + h]);
                float betaH = betaArr[s * hv + h];
                long stBase = (long)h * sv * sk;

                for (int idx = 0; idx < sv * sk; idx++) state[stBase + idx] *= gExp;

                for (int row = 0; row < sv; row++)
                {
                    float acc = 0f;
                    long rb = stBase + (long)row * sk;
                    for (int col = 0; col < sk; col++) acc += state[rb + col] * kArr[qOff + col];
                    skRow[row] = acc;
                }
                for (int row = 0; row < sv; row++)
                {
                    float delta = (vArr[vOff + row] - skRow[row]) * betaH;
                    long rb = stBase + (long)row * sk;
                    for (int col = 0; col < sk; col++) state[rb + col] += kArr[qOff + col] * delta;
                }
                for (int row = 0; row < sv; row++)
                {
                    float acc = 0f;
                    long rb = stBase + (long)row * sk;
                    for (int col = 0; col < sk; col++) acc += state[rb + col] * qArr[qOff + col];
                    o[vOff + row] = acc;
                }
            }
        }

        // Gated RMSNorm per head: norm(o) FIRST (with the learned ssm_norm weight), THEN elementwise-multiply
        // by silu(z) — Qwen3.5's own build_norm_gated order (opposite of Mamba2's gate-then-norm).
        float* normW = (float*)W($"{p}.ssm_norm.weight").DataPointer;   // [sv]
        float* zP = (float*)z.DataPointer;
        for (int s = 0; s < seq; s++)
            for (int h = 0; h < hv; h++)
            {
                long b = (long)s * valueDim + h * sv;
                float var = 0f;
                for (int c = 0; c < sv; c++) { float v = o[b + c]; var += v * v; }
                float inv = 1f / MathF.Sqrt(var / sv + Eps);
                for (int c = 0; c < sv; c++)
                {
                    float normed = o[b + c] * inv * normW[c];
                    float zv = zP[b + c];
                    o[b + c] = normed * (zv / (1f + MathF.Exp(-zv)));
                }
            }
        z.Dispose();

        Tensor oT = new(new TensorShape(1, seq, valueDim), DType.F32);
        fixed (float* src = o) Buffer.MemoryCopy(src, (void*)oT.DataPointer, (long)seq * valueDim * 4, (long)seq * valueDim * 4);
        Tensor attnOut = new(flat, DType.F32);
        backend.Linear(attnOut, oT, W($"{p}.ssm_out.weight"), null);
        oT.Dispose();
        Tensor afterAttn = new(flat, DType.F32);
        backend.Add(afterAttn, hIn, attnOut);
        attnOut.Dispose();

        Tensor preFfn = new(flat, DType.F32);
        backend.RmsNorm(preFfn, afterAttn, W($"{p}.post_attention_norm.weight"), Eps);
        Tensor ffnOut = DenseSwiGlu(backend, preFfn, p, seq);
        preFfn.Dispose();
        Tensor result = new(flat, DType.F32);
        backend.Add(result, afterAttn, ffnOut);
        afterAttn.Dispose(); ffnOut.Dispose();
        return result;
    }

    /// <summary>L2-normalizes each head's <paramref name="dim"/>-wide slice in place (no learned scale — Qwen3.5
    /// applies this to Q/K after the conv, ggml's <c>ggml_l2_norm</c>).</summary>
    private static void L2NormalizePerHead(float[] arr, int seq, int heads, int dim, float eps)
    {
        for (int s = 0; s < seq; s++)
            for (int h = 0; h < heads; h++)
            {
                long b = (long)s * heads * dim + (long)h * dim;
                float sumSq = 0f;
                for (int c = 0; c < dim; c++) sumSq += arr[b + c] * arr[b + c];
                float inv = 1f / MathF.Sqrt(sumSq + eps);
                for (int c = 0; c < dim; c++) arr[b + c] *= inv;
            }
    }

    /// <summary>Dense SwiGLU FFN: <c>down(silu(gate(x)) · up(x))</c>. Shared by every layer (Qwen3.5 has no
    /// MoE FFN on the dense 0.8B-9B tier — the MoE variant is a separate GGUF arch, <c>qwen35moe</c>).</summary>
    private Tensor DenseSwiGlu(IBackend backend, Tensor preFfn, string p, int seq)
    {
        int ffWidth = (int)W($"{p}.ffn_up.weight").Shape[0];   // Shape[0] = outDim (post-relabel [out,in])
        TensorShape ff = new(1, seq, ffWidth);
        Tensor up = new(ff, DType.F32);
        backend.Linear(up, preFfn, W($"{p}.ffn_up.weight"), null);
        Tensor gate = new(ff, DType.F32);
        backend.Linear(gate, preFfn, W($"{p}.ffn_gate.weight"), null);
        Tensor gateAct = new(ff, DType.F32);
        backend.Silu(gateAct, gate);
        gate.Dispose();
        Tensor comb = new(ff, DType.F32);
        backend.Mul(comb, gateAct, up);
        gateAct.Dispose(); up.Dispose();
        Tensor outp = new(new TensorShape(1, seq, DModel), DType.F32);
        backend.Linear(outp, comb, W($"{p}.ffn_down.weight"), null);
        comb.Dispose();
        return outp;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _kvCache.Dispose();
        _handle.Dispose();
    }
}
