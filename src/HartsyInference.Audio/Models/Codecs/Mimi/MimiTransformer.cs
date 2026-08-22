using HartsyInference.Audio.Layers;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs.Mimi;

/// <summary>Mimi's transformer-of-codecs, matching the HF <c>transformers</c> MimiModel layout (the format the real kyutai/mimi checkpoint ships).</summary>
/// <remarks>Pre-LayerNorm blocks with split Q/K/V projections, split-half (rotate_half)
/// RoPE, sliding-window causal attention (<see cref="MimiConfig.TransformerContext"/> = 250), per-block
/// <c>LayerScale</c>, and a bias-free exact-GELU MLP. Keys: <c>{prefix}.layers.N.{input_layernorm,
/// post_attention_layernorm}.{weight,bias}</c>, <c>self_attn.{q,k,v,o}_proj.weight</c>, <c>mlp.fc1/fc2.weight</c>,
/// <c>self_attn_layer_scale.scale</c>, <c>mlp_layer_scale.scale</c>.</remarks>
internal sealed unsafe class MimiTransformer
{
    private readonly MimiConfig _cfg;
    private readonly string _prefix;
    private readonly int _layers, _heads, _headDim, _dim, _ffn;
    private bool _interleavedRope;   // moshi/DSM checkpoints rotate interleaved pairs; HF permutes → split-half
    private const float LnEps = 1e-5f;

    private readonly Tensor?[] _n1W, _n1B, _n2W, _n2B, _qW, _kW, _vW, _oW, _fc1W, _fc2W, _ls1, _ls2;

    public MimiTransformer(MimiConfig cfg, string prefix)
    {
        _cfg = cfg; _prefix = prefix;
        _layers = cfg.TransformerLayers; _heads = cfg.TransformerHeads; _dim = cfg.TransformerDim;
        _headDim = _dim / _heads; _ffn = cfg.TransformerFfnDim;
        _n1W = new Tensor?[_layers]; _n1B = new Tensor?[_layers]; _n2W = new Tensor?[_layers]; _n2B = new Tensor?[_layers];
        _qW = new Tensor?[_layers]; _kW = new Tensor?[_layers]; _vW = new Tensor?[_layers]; _oW = new Tensor?[_layers];
        _fc1W = new Tensor?[_layers]; _fc2W = new Tensor?[_layers]; _ls1 = new Tensor?[_layers]; _ls2 = new Tensor?[_layers];
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        // Two on-disk layouts share this forward: the HF `transformers` MimiModel naming (input_layernorm /
        // split q,k,v_proj / mlp.fc1,fc2 / self_attn_layer_scale — kyutai/mimi, PocketTTS) and the moshi-native
        // DSM naming (an extra `.transformer.` level, norm1/norm2, a FUSED self_attn.in_proj_weight [3·dim,dim],
        // out_proj, linear1/linear2, layer_scale_1/2 — the tts-1.6b-en_fr `tokenizer-*.safetensors`). Detect and
        // adapt; the math (pre-norm, split-half RoPE, sliding-window causal attn, LayerScale, GELU FFN) is identical.
        bool dsm = w.ContainsKey($"{_prefix}.transformer.layers.0.norm1.weight");
        _interleavedRope = dsm;
        for (int i = 0; i < _layers; i++)
        {
            if (dsm)
            {
                string p = $"{_prefix}.transformer.layers.{i}";
                _n1W[i] = WhisperOps.EnsureF32(w[$"{p}.norm1.weight"]);
                _n1B[i] = WhisperOps.EnsureF32(w[$"{p}.norm1.bias"]);
                _n2W[i] = WhisperOps.EnsureF32(w[$"{p}.norm2.weight"]);
                _n2B[i] = WhisperOps.EnsureF32(w[$"{p}.norm2.bias"]);
                Tensor inProj = WhisperOps.EnsureF32(w[$"{p}.self_attn.in_proj_weight"]);  // [3·dim, dim] = [Q;K;V]
                _qW[i] = SliceRows(inProj, 0, _dim);
                _kW[i] = SliceRows(inProj, _dim, 2 * _dim);
                _vW[i] = SliceRows(inProj, 2 * _dim, 3 * _dim);
                _oW[i] = WhisperOps.EnsureF32(w[$"{p}.self_attn.out_proj.weight"]);
                _fc1W[i] = WhisperOps.EnsureF32(w[$"{p}.linear1.weight"]);
                _fc2W[i] = WhisperOps.EnsureF32(w[$"{p}.linear2.weight"]);
                _ls1[i] = WhisperOps.EnsureF32(w[$"{p}.layer_scale_1.scale"]);
                _ls2[i] = WhisperOps.EnsureF32(w[$"{p}.layer_scale_2.scale"]);
            }
            else
            {
                string p = $"{_prefix}.layers.{i}";
                _n1W[i] = WhisperOps.EnsureF32(w[$"{p}.input_layernorm.weight"]);
                _n1B[i] = WhisperOps.EnsureF32(w[$"{p}.input_layernorm.bias"]);
                _n2W[i] = WhisperOps.EnsureF32(w[$"{p}.post_attention_layernorm.weight"]);
                _n2B[i] = WhisperOps.EnsureF32(w[$"{p}.post_attention_layernorm.bias"]);
                _qW[i] = WhisperOps.EnsureF32(w[$"{p}.self_attn.q_proj.weight"]);
                _kW[i] = WhisperOps.EnsureF32(w[$"{p}.self_attn.k_proj.weight"]);
                _vW[i] = WhisperOps.EnsureF32(w[$"{p}.self_attn.v_proj.weight"]);
                _oW[i] = WhisperOps.EnsureF32(w[$"{p}.self_attn.o_proj.weight"]);
                _fc1W[i] = WhisperOps.EnsureF32(w[$"{p}.mlp.fc1.weight"]);
                _fc2W[i] = WhisperOps.EnsureF32(w[$"{p}.mlp.fc2.weight"]);
                _ls1[i] = WhisperOps.EnsureF32(w[$"{p}.self_attn_layer_scale.scale"]);
                _ls2[i] = WhisperOps.EnsureF32(w[$"{p}.mlp_layer_scale.scale"]);
            }
        }
    }

    /// <summary>Copies rows <c>[r0, r1)</c> of a <c>[R, inDim]</c> row-major weight into a fresh owned <c>[r1-r0, inDim]</c> tensor (splits a fused <c>in_proj_weight</c> into q/k/v); runs once at load.</summary>
    private static Tensor SliceRows(Tensor w, int r0, int r1)
    {
        int inDim = (int)w.Shape[1];
        Tensor outT = new(new TensorShape(r1 - r0, inDim), DType.F32);
        Buffer.MemoryCopy((float*)w.DataPointer + (long)r0 * inDim, (void*)outT.DataPointer,
            (long)(r1 - r0) * inDim * 4, (long)(r1 - r0) * inDim * 4);
        return outT;
    }

    /// <summary>x channels-last <c>[B,T,dim]</c> -> <c>[B,T,dim]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor x, int batch, int t)
    {
        Tensor cur = new(x.Shape, DType.F32);
        long n = x.Shape.ElementCount;
        Buffer.MemoryCopy((void*)x.DataPointer, (void*)cur.DataPointer, n * 4, n * 4);

        Tensor mask = BuildMask(t, _cfg.TransformerContext);
        for (int i = 0; i < _layers; i++)
        {
            Tensor normed = new(cur.Shape, DType.F32);
            backend.LayerNorm(normed, cur, _n1W[i]!, _n1B[i]!, LnEps);
            Tensor q = WhisperOps.ProjectLinear(backend, normed, _qW[i]!, null, batch, t, _dim, _dim);
            Tensor k = WhisperOps.ProjectLinear(backend, normed, _kW[i]!, null, batch, t, _dim, _dim);
            Tensor v = WhisperOps.ProjectLinear(backend, normed, _vW[i]!, null, batch, t, _dim, _dim);
            normed.Dispose();

            Tensor qMh = new(new TensorShape(batch, _heads, t, _headDim), DType.F32);
            Tensor kMh = new(new TensorShape(batch, _heads, t, _headDim), DType.F32);
            Tensor vMh = new(new TensorShape(batch, _heads, t, _headDim), DType.F32);
            WhisperOps.ReshapeToMultiHead4D(qMh, q, batch, t, _heads, _headDim);
            WhisperOps.ReshapeToMultiHead4D(kMh, k, batch, t, _heads, _headDim);
            WhisperOps.ReshapeToMultiHead4D(vMh, v, batch, t, _heads, _headDim);
            q.Dispose(); k.Dispose(); v.Dispose();
            ApplyRoPE(qMh, batch, _heads, t, _headDim, _cfg.TransformerRopeTheta, _interleavedRope);
            ApplyRoPE(kMh, batch, _heads, t, _headDim, _cfg.TransformerRopeTheta, _interleavedRope);

            Tensor attn = new(qMh.Shape, DType.F32);
            backend.ScaledDotProductAttention(attn, qMh, kMh, vMh, mask, 1f / MathF.Sqrt(_headDim));
            qMh.Dispose(); kMh.Dispose(); vMh.Dispose();

            Tensor flat = new(new TensorShape(batch, t, _dim), DType.F32);
            float* ap = (float*)attn.DataPointer; float* fp = (float*)flat.DataPointer;
            for (int b = 0; b < batch; b++)
                for (int ti = 0; ti < t; ti++)
                    for (int h = 0; h < _heads; h++)
                        for (int d = 0; d < _headDim; d++)
                            fp[((long)b * t + ti) * _dim + h * _headDim + d] = ap[(((long)b * _heads + h) * t + ti) * _headDim + d];
            attn.Dispose();
            Tensor outp = WhisperOps.ProjectLinear(backend, flat, _oW[i]!, null, batch, t, _dim, _dim);
            flat.Dispose();
            AddScaled(cur, outp, _ls1[i]!, batch, t);
            outp.Dispose();

            Tensor normed2 = new(cur.Shape, DType.F32);
            backend.LayerNorm(normed2, cur, _n2W[i]!, _n2B[i]!, LnEps);
            Tensor up = WhisperOps.ProjectLinear(backend, normed2, _fc1W[i]!, null, batch, t, _dim, _ffn);
            normed2.Dispose();
            Activations.ErfGelu(up);
            Tensor down = WhisperOps.ProjectLinear(backend, up, _fc2W[i]!, null, batch, t, _ffn, _dim);
            up.Dispose();
            AddScaled(cur, down, _ls2[i]!, batch, t);
            down.Dispose();
        }
        mask.Dispose();
        return cur;
    }

    /// <summary>x channels-last <c>[1,t,dim]</c> -> <c>[1,t,dim]</c>, attending against <paramref name="cache"/>'s carried prefix instead of only the frames in this call; batch must be 1.</summary>
    /// <remarks>Equivalent (to float rounding) to <see cref="Forward"/> run once over the full concatenated
    /// sequence — verified by <c>MimiStreamParityTests</c>: this is NOT an approximation, it's the same
    /// computation restructured to avoid recomputing the whole prefix every call.</remarks>
    public Tensor ForwardStreaming(IBackend backend, Tensor x, int t, MimiTransformerCache cache)
    {
        if (x.Shape[0] != 1) throw new NotSupportedException("MimiTransformer.ForwardStreaming supports batch=1 only.");
        cache.EnsureLayers(_layers);
        int priorLen = cache.Length;

        Tensor cur = new(x.Shape, DType.F32);
        long n = x.Shape.ElementCount;
        Buffer.MemoryCopy((void*)x.DataPointer, (void*)cur.DataPointer, n * 4, n * 4);

        Tensor mask = BuildWindowedMask(t, priorLen, _cfg.TransformerContext);
        for (int i = 0; i < _layers; i++)
        {
            Tensor normed = new(cur.Shape, DType.F32);
            backend.LayerNorm(normed, cur, _n1W[i]!, _n1B[i]!, LnEps);
            Tensor q = WhisperOps.ProjectLinear(backend, normed, _qW[i]!, null, 1, t, _dim, _dim);
            Tensor kNew = WhisperOps.ProjectLinear(backend, normed, _kW[i]!, null, 1, t, _dim, _dim);
            Tensor vNew = WhisperOps.ProjectLinear(backend, normed, _vW[i]!, null, 1, t, _dim, _dim);
            normed.Dispose();

            Tensor qMh = new(new TensorShape(1, _heads, t, _headDim), DType.F32);
            Tensor kNewMh = new(new TensorShape(1, _heads, t, _headDim), DType.F32);
            Tensor vNewMh = new(new TensorShape(1, _heads, t, _headDim), DType.F32);
            WhisperOps.ReshapeToMultiHead4D(qMh, q, 1, t, _heads, _headDim);
            WhisperOps.ReshapeToMultiHead4D(kNewMh, kNew, 1, t, _heads, _headDim);
            WhisperOps.ReshapeToMultiHead4D(vNewMh, vNew, 1, t, _heads, _headDim);
            q.Dispose(); kNew.Dispose(); vNew.Dispose();

            // RoPE at ABSOLUTE positions: the new frames start at priorLen, not 0 — this is the easiest thing to
            // get wrong in this change (right mask, wrong RoPE still produces plausible-but-wrong audio with no
            // seam signature to catch it, which is exactly why the parity test compares against a real monolithic
            // decode rather than just checking for boundary clicks).
            ApplyRoPE(qMh, 1, _heads, t, _headDim, _cfg.TransformerRopeTheta, _interleavedRope, posOffset: priorLen);
            ApplyRoPE(kNewMh, 1, _heads, t, _headDim, _cfg.TransformerRopeTheta, _interleavedRope, posOffset: priorLen);

            (Tensor kAll, Tensor vAll) = cache.AppendAndGet(i, kNewMh, vNewMh, t);
            kNewMh.Dispose(); vNewMh.Dispose();
            int totalKv = priorLen + t;

            Tensor attn = new(qMh.Shape, DType.F32);
            backend.ScaledDotProductAttention(attn, qMh, kAll, vAll, mask, 1f / MathF.Sqrt(_headDim));
            qMh.Dispose();
            // kAll/vAll are owned by the cache now (it returns its own stored tensors) — do not dispose here.
            _ = totalKv;

            Tensor flat = new(new TensorShape(1, t, _dim), DType.F32);
            float* ap = (float*)attn.DataPointer; float* fp = (float*)flat.DataPointer;
            for (int ti = 0; ti < t; ti++)
                for (int h = 0; h < _heads; h++)
                    for (int d = 0; d < _headDim; d++)
                        fp[(long)ti * _dim + h * _headDim + d] = ap[((long)h * t + ti) * _headDim + d];
            attn.Dispose();
            Tensor outp = WhisperOps.ProjectLinear(backend, flat, _oW[i]!, null, 1, t, _dim, _dim);
            flat.Dispose();
            AddScaled(cur, outp, _ls1[i]!, 1, t);
            outp.Dispose();

            Tensor normed2 = new(cur.Shape, DType.F32);
            backend.LayerNorm(normed2, cur, _n2W[i]!, _n2B[i]!, LnEps);
            Tensor up = WhisperOps.ProjectLinear(backend, normed2, _fc1W[i]!, null, 1, t, _dim, _ffn);
            normed2.Dispose();
            Activations.ErfGelu(up);
            Tensor down = WhisperOps.ProjectLinear(backend, up, _fc2W[i]!, null, 1, t, _ffn, _dim);
            up.Dispose();
            AddScaled(cur, down, _ls2[i]!, 1, t);
            down.Dispose();
        }
        mask.Dispose();
        cache.Length += t;
        return cur;
    }

    private void AddScaled(Tensor cur, Tensor upd, Tensor scale, int batch, int t)
    {
        float* cp = (float*)cur.DataPointer, up = (float*)upd.DataPointer, sp = (float*)scale.DataPointer;
        for (int b = 0; b < batch; b++)
            for (int ti = 0; ti < t; ti++)
                for (int c = 0; c < _dim; c++)
                {
                    long idx = ((long)b * t + ti) * _dim + c;
                    cp[idx] += sp[c] * up[idx];
                }
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        for (int i = 0; i < _layers; i++)
        {
            Tensor?[] a = [_n1W[i], _n1B[i], _n2W[i], _n2B[i], _qW[i], _kW[i], _vW[i], _oW[i], _fc1W[i], _fc2W[i], _ls1[i], _ls2[i]];
            foreach (Tensor? t in a) if (t is not null) yield return t;
        }
    }

    private static Tensor BuildMask(int t, int? context)
    {
        Tensor mask = new(new TensorShape(t, t), DType.F32);
        float* mp = (float*)mask.DataPointer;
        for (int q = 0; q < t; q++)
            for (int kk = 0; kk < t; kk++)
            {
                int delta = q - kk;
                bool ok = delta >= 0 && (context is null || delta < context.Value);
                mp[(long)q * t + kk] = ok ? 0f : float.NegativeInfinity;
            }
        return mask;
    }

    /// <summary>Same sliding-window causal rule as <see cref="BuildMask"/>, generalized to a query block starting at absolute position <paramref name="priorLen"/> and a key axis spanning the full cached prefix instead of assuming both start at 0.</summary>
    private static Tensor BuildWindowedMask(int t, int priorLen, int? context)
    {
        int totalKv = priorLen + t;
        Tensor mask = new(new TensorShape(t, totalKv), DType.F32);
        float* mp = (float*)mask.DataPointer;
        for (int qi = 0; qi < t; qi++)
        {
            int absQ = priorLen + qi;
            for (int kk = 0; kk < totalKv; kk++)
            {
                int delta = absQ - kk;
                bool ok = delta >= 0 && (context is null || delta < context.Value);
                mp[(long)qi * totalKv + kk] = ok ? 0f : float.NegativeInfinity;
            }
        }
        return mask;
    }

    // RoPE on [B,H,T,D]. HF `transformers` Mimi permutes q/k so a split-half (rotate_half) rotation applies
    // (pairs (i, i+D/2)); the moshi/DSM checkpoint keeps the native interleaved layout (pairs (2p, 2p+1)). Both
    // share the frequency schedule freq_p = theta^(-2p/D).
    private static void ApplyRoPE(Tensor x, int batch, int heads, int t, int headDim, float theta, bool interleave, int posOffset = 0)
    {
        float* xp = (float*)x.DataPointer;
        int half = headDim / 2;
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < heads; h++)
                for (int ti = 0; ti < t; ti++)
                {
                    long rb = (((long)b * heads + h) * t + ti) * headDim;
                    int absPos = posOffset + ti;
                    for (int p = 0; p < half; p++)
                    {
                        float freq = MathF.Pow(theta, -2f * p / headDim);
                        float ang = absPos * freq, cs = MathF.Cos(ang), sn = MathF.Sin(ang);
                        int i0 = interleave ? 2 * p : p;
                        int i1 = interleave ? 2 * p + 1 : half + p;
                        float x0 = xp[rb + i0], x1 = xp[rb + i1];
                        xp[rb + i0] = x0 * cs - x1 * sn;
                        xp[rb + i1] = x0 * sn + x1 * cs;
                    }
                }
    }
}
