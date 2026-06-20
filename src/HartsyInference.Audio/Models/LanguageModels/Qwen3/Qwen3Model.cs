using HartsyInference.Audio.Models.Dia;
using HartsyInference.Audio.Models.Moonshine;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.LanguageModels.Qwen3;

/// <summary>Headless Qwen3 decoder body (layers + final RMSNorm), driven by <see cref="ForwardEmbeds"/>. Each
/// layer is pre-norm: RMSNorm → GQA attention with <b>per-head q_norm/k_norm</b> + RoPE → residual → RMSNorm →
/// SwiGLU → residual. Decoupled head_dim (q/k/v sized <c>n_heads·head_dim</c>), no QKV bias. Reuses
/// <see cref="DiaHeads"/>, <see cref="RotaryEmbedding"/> (interleaved; 3D mRoPE is a staged refinement),
/// <see cref="WhisperOps.ProjectLinear"/>, and <see cref="IBackend"/> RMSNorm/SDPA/SwiGLU.</summary>
public sealed unsafe class Qwen3Model : IDisposable
{
    private readonly Qwen3Config _cfg;
    private readonly Layer[] _layers;
    private Tensor? _finalNorm;
    private float[]? _cos, _sin;
    private int _disposed;

    public Qwen3Config Config => _cfg;
    public int NumLayers => _cfg.NumHiddenLayers;
    public int KvHeads => _cfg.NumKeyValueHeads;
    public int HeadDim => _cfg.HeadDim;

    public Qwen3Model(Qwen3Config cfg)
    {
        _cfg = cfg;
        _layers = new Layer[cfg.NumHiddenLayers];
        for (int i = 0; i < _layers.Length; i++) _layers[i] = new Layer(cfg);
    }

    public void LoadWeightsHeadless(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _finalNorm = WhisperOps.EnsureF32(w[$"{prefix}.norm.weight"]);
        for (int i = 0; i < _layers.Length; i++) _layers[i].LoadWeights(w, $"{prefix}.layers.{i}");
        (_cos, _sin) = RotaryEmbedding.GetTables(_cfg.HeadDim, _cfg.RopeTheta, _cfg.MaxPositionEmbeddings);
    }

    public Tensor ForwardEmbeds(IBackend backend, Tensor embeds, int t, int posStart, StreamingKvCache cache)
    {
        Tensor? mask = t > 1 ? BuildCausalMask(t, posStart) : null;
        try
        {
            Tensor hidden = embeds;
            bool owns = false;
            for (int i = 0; i < _layers.Length; i++)
            {
                Tensor next = _layers[i].Forward(backend, hidden, t, posStart, cache, i, mask, _cos!, _sin!);
                if (owns) hidden.Dispose();
                hidden = next; owns = true;
            }
            cache.AdvanceLength(t);
            Tensor normed = new(hidden.Shape, DType.F32);
            backend.RmsNorm(normed, hidden, _finalNorm!, _cfg.RmsNormEps);
            if (owns) hidden.Dispose();
            return normed;
        }
        finally { mask?.Dispose(); }
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Layer l in _layers) foreach (Tensor t in l.EnumerateWeights()) yield return t;
        if (_finalNorm is not null) yield return _finalNorm;
    }

    private static Tensor BuildCausalMask(int tq, int qOffset)
    {
        int tk = qOffset + tq;
        Tensor mask = new(new TensorShape(1, 1, tq, tk), DType.F32);
        float* p = (float*)mask.DataPointer;
        const float NegInf = -1e30f;
        for (int q = 0; q < tq; q++) { int aq = qOffset + q; for (int k = 0; k < tk; k++) p[q * tk + k] = k <= aq ? 0f : NegInf; }
        return mask;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _finalNorm = null;
        GC.SuppressFinalize(this);
    }

    private sealed class Layer
    {
        private readonly Qwen3Config _cfg;
        private Tensor? _inNorm, _postNorm, _qW, _kW, _vW, _oW, _qNorm, _kNorm, _gateW, _upW, _downW;

        public Layer(Qwen3Config cfg) => _cfg = cfg;

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
        {
            _inNorm = WhisperOps.EnsureF32(w[$"{p}.input_layernorm.weight"]);
            _postNorm = WhisperOps.EnsureF32(w[$"{p}.post_attention_layernorm.weight"]);
            _qW = WhisperOps.EnsureF32(w[$"{p}.self_attn.q_proj.weight"]);
            _kW = WhisperOps.EnsureF32(w[$"{p}.self_attn.k_proj.weight"]);
            _vW = WhisperOps.EnsureF32(w[$"{p}.self_attn.v_proj.weight"]);
            _oW = WhisperOps.EnsureF32(w[$"{p}.self_attn.o_proj.weight"]);
            _qNorm = WhisperOps.EnsureF32(w[$"{p}.self_attn.q_norm.weight"]);
            _kNorm = WhisperOps.EnsureF32(w[$"{p}.self_attn.k_norm.weight"]);
            _gateW = WhisperOps.EnsureF32(w[$"{p}.mlp.gate_proj.weight"]);
            _upW = WhisperOps.EnsureF32(w[$"{p}.mlp.up_proj.weight"]);
            _downW = WhisperOps.EnsureF32(w[$"{p}.mlp.down_proj.weight"]);
        }

        public Tensor Forward(IBackend backend, Tensor x, int t, int posStart, StreamingKvCache cache, int layer, Tensor? mask, float[] cos, float[] sin)
        {
            int h = _cfg.HiddenSize, qh = _cfg.NumAttentionHeads, kvh = _cfg.NumKeyValueHeads, hd = _cfg.HeadDim;
            Tensor pre = new(new TensorShape(1, t, h), DType.F32);
            backend.RmsNorm(pre, x, _inNorm!, _cfg.RmsNormEps);

            Tensor q = WhisperOps.ProjectLinear(backend, pre, _qW!, null, 1, t, h, _cfg.QDim);
            Tensor k = WhisperOps.ProjectLinear(backend, pre, _kW!, null, 1, t, h, _cfg.KvDim);
            Tensor v = WhisperOps.ProjectLinear(backend, pre, _vW!, null, 1, t, h, _cfg.KvDim);
            pre.Dispose();
            Tensor qM = new(new TensorShape(1, qh, t, hd), DType.F32);
            Tensor kM = new(new TensorShape(1, kvh, t, hd), DType.F32);
            Tensor vM = new(new TensorShape(1, kvh, t, hd), DType.F32);
            DiaHeads.FlatToHeads(qM, q, t, qh, hd); q.Dispose();
            DiaHeads.FlatToHeads(kM, k, t, kvh, hd); k.Dispose();
            DiaHeads.FlatToHeads(vM, v, t, kvh, hd); v.Dispose();

            // Qwen3: per-head RMSNorm on q and k (over head_dim), then RoPE.
            HeadRmsNorm(qM, qh, t, hd, _qNorm!, _cfg.RmsNormEps);
            HeadRmsNorm(kM, kvh, t, hd, _kNorm!, _cfg.RmsNormEps);
            RotaryEmbedding.ApplyInPlace(qM, qh, t, hd, hd, posStart, cos, sin);
            RotaryEmbedding.ApplyInPlace(kM, kvh, t, hd, hd, posStart, cos, sin);

            cache.Append(layer, kM, vM); kM.Dispose(); vM.Dispose();
            int kvLen = posStart + t;
            Tensor kP = DiaHeads.SliceLeading(cache.GetK(layer), kvh, kvLen, hd);
            Tensor vP = DiaHeads.SliceLeading(cache.GetV(layer), kvh, kvLen, hd);
            Tensor kR, vR;
            int group = _cfg.KvGroup;
            if (group == 1) { kR = kP; vR = vP; }
            else
            {
                kR = new Tensor(new TensorShape(1, qh, kvLen, hd), DType.F32);
                vR = new Tensor(new TensorShape(1, qh, kvLen, hd), DType.F32);
                DiaHeads.RepeatKv(kR, kP, kvh, group, kvLen, hd);
                DiaHeads.RepeatKv(vR, vP, kvh, group, kvLen, hd);
                kP.Dispose(); vP.Dispose();
            }
            Tensor attn = new(new TensorShape(1, qh, t, hd), DType.F32);
            backend.ScaledDotProductAttention(attn, qM, kR, vR, mask, 1f / MathF.Sqrt(hd));
            qM.Dispose(); kR.Dispose(); vR.Dispose();
            Tensor flat = new(new TensorShape(1, t, _cfg.QDim), DType.F32);
            DiaHeads.HeadsToFlat(flat, attn, t, qh, hd); attn.Dispose();
            Tensor attnOut = WhisperOps.ProjectLinear(backend, flat, _oW!, null, 1, t, _cfg.QDim, h); flat.Dispose();

            Tensor afterAttn = new(x.Shape, DType.F32); backend.Add(afterAttn, x, attnOut); attnOut.Dispose();
            Tensor pre2 = new(new TensorShape(1, t, h), DType.F32); backend.RmsNorm(pre2, afterAttn, _postNorm!, _cfg.RmsNormEps);
            Tensor gate = WhisperOps.ProjectLinear(backend, pre2, _gateW!, null, 1, t, h, _cfg.IntermediateSize);
            Tensor up = WhisperOps.ProjectLinear(backend, pre2, _upW!, null, 1, t, h, _cfg.IntermediateSize); pre2.Dispose();
            backend.Silu(gate, gate);
            Tensor act = new(gate.Shape, DType.F32); backend.Mul(act, gate, up); gate.Dispose(); up.Dispose();
            Tensor mlp = WhisperOps.ProjectLinear(backend, act, _downW!, null, 1, t, _cfg.IntermediateSize, h); act.Dispose();
            Tensor outT = new(x.Shape, DType.F32); backend.Add(outT, afterAttn, mlp); afterAttn.Dispose(); mlp.Dispose();
            return outT;
        }

        private static void HeadRmsNorm(Tensor x, int heads, int t, int hd, Tensor weight, float eps)
        {
            float* xp = (float*)x.DataPointer; float* wp = (float*)weight.DataPointer;
            for (int h = 0; h < heads; h++)
                for (int s = 0; s < t; s++)
                {
                    long off = ((long)h * t + s) * hd;
                    double ms = 0; for (int d = 0; d < hd; d++) ms += (double)xp[off + d] * xp[off + d];
                    float inv = (float)(1.0 / Math.Sqrt(ms / hd + eps));
                    for (int d = 0; d < hd; d++) xp[off + d] = xp[off + d] * inv * wp[d];
                }
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            Tensor?[] a = [_inNorm, _postNorm, _qW, _kW, _vW, _oW, _qNorm, _kNorm, _gateW, _upW, _downW];
            foreach (Tensor? t in a) if (t is not null) yield return t;
        }
    }
}
