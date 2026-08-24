using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Music;

/// <summary>Sana DCAE EfficientViT block (diffusers <c>SanaMultiscaleLinearAttention</c> + a 2-D GLUMBConv FFN):
/// fused-QKV linear projection plus a 5×5 grouped-conv multiscale branch, ReLU-kernel <b>linear</b> attention per
/// head (the padded-ones-row normalizer trick), RMSNorm'd output projection with residual, then the gated MBConv FFN
/// (1×1 expand ×2 → SiLU → depthwise 3×3 → SiLU-gate chunk → 1×1 project, residual). Operates on NCHW. Used in the
/// deepest DCAE stages; numerics validation-pending.</summary>
public sealed unsafe class EfficientVitBlock(int dim, int headDim = 32)
{
    private readonly int _dim = dim;
    private readonly int _headDim = headDim;

    private Tensor? _qW, _kW, _vW;                          // base qkv (linear, no bias)
    private Tensor? _msProjInW, _msProjInB;                  // multiscale 5×5 grouped conv over fused qkv
    private Tensor? _msProjOutW, _msProjOutB;                // multiscale 1×1 grouped conv
    private Tensor? _projOutW, _projOutB;                    // attention output projection
    private Tensor? _attnNormW, _attnNormB;                  // norm_out (RMS, channel)
    private Tensor? _ffInvW, _ffInvB, _ffDepthW, _ffDepthB, _ffPointW;   // GLUMBConv
    private Tensor? _ffNormW, _ffNormB;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
    {
        _qW = w[$"{p}.attn.to_q.weight"];
        _kW = w[$"{p}.attn.to_k.weight"];
        _vW = w[$"{p}.attn.to_v.weight"];
        if (w.TryGetValue($"{p}.attn.to_qkv_multiscale.0.proj_in.weight", out _msProjInW))
        {
            w.TryGetValue($"{p}.attn.to_qkv_multiscale.0.proj_in.bias", out _msProjInB);
            w.TryGetValue($"{p}.attn.to_qkv_multiscale.0.proj_out.weight", out _msProjOutW);
            w.TryGetValue($"{p}.attn.to_qkv_multiscale.0.proj_out.bias", out _msProjOutB);
        }
        if (!w.TryGetValue($"{p}.attn.to_out.weight", out _projOutW))
            _projOutW = w[$"{p}.attn.proj_out.weight"];
        if (!w.TryGetValue($"{p}.attn.to_out.bias", out _projOutB))
            w.TryGetValue($"{p}.attn.proj_out.bias", out _projOutB);
        _attnNormW = TensorCasts.LoadF32(w, $"{p}.attn.norm_out.weight"); w.TryGetValue($"{p}.attn.norm_out.bias", out _attnNormB);

        _ffInvW = w[$"{p}.conv_out.conv_inverted.weight"]; w.TryGetValue($"{p}.conv_out.conv_inverted.bias", out _ffInvB);
        _ffDepthW = w[$"{p}.conv_out.conv_depth.weight"]; w.TryGetValue($"{p}.conv_out.conv_depth.bias", out _ffDepthB);
        _ffPointW = w[$"{p}.conv_out.conv_point.weight"];
        _ffNormW = TensorCasts.LoadF32(w, $"{p}.conv_out.norm.weight"); w.TryGetValue($"{p}.conv_out.norm.bias", out _ffNormB);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _qW, _kW, _vW, _msProjInW, _msProjInB, _msProjOutW, _msProjOutB,
            _projOutW, _projOutB, _attnNormW, _attnNormB,
            _ffInvW, _ffInvB, _ffDepthW, _ffDepthB, _ffPointW, _ffNormW, _ffNormB })
            if (t is not null) yield return t;
    }

    public Tensor Forward(IBackend backend, Tensor x)
    {
        Tensor attended = Attention(backend, x);
        // Residual connection around the attention.
        AddInPlace(attended, x);
        Tensor ff = GlumbConv(backend, attended);
        AddInPlace(ff, attended);
        attended.Dispose();
        return ff;
    }

    /// <summary>Multiscale ReLU linear attention over the spatial token grid.</summary>
    private Tensor Attention(IBackend backend, Tensor x)
    {
        int b = (int)x.Shape[0], c = (int)x.Shape[1], hh = (int)x.Shape[2], ww = (int)x.Shape[3];
        long s = (long)hh * ww;

        // Tokens [S, C] from NCHW.
        Tensor tokens = ToTokens(x, c, s);
        Tensor q = Project(backend, tokens, _qW!, (int)s, c);
        Tensor k = Project(backend, tokens, _kW!, (int)s, c);
        Tensor v = Project(backend, tokens, _vW!, (int)s, c);
        tokens.Dispose();

        // Multiscale branch: 5×5 grouped conv over the fused qkv maps (adds a second head group).
        Tensor? qm = null, km = null, vm = null;
        if (_msProjInW is not null)
        {
            // diffusers SanaMultiscaleAttentionProjection: depthwise 5×5 over fused qkv, then a grouped 1×1
            // (groups = 3·heads).
            Tensor fused = FuseQkvMaps(q, k, v, c, hh, ww);
            Tensor ms = DepthwiseConv2d(fused, _msProjInW, _msProjInB);
            fused.Dispose();
            if (_msProjOutW is not null)
            {
                Tensor ms2 = GroupedPointwiseConv2d(ms, _msProjOutW, _msProjOutB, groups: 3 * (c / _headDim));
                ms.Dispose();
                ms = ms2;
            }
            (qm, km, vm) = SplitQkvMaps(ms, c, s);
            ms.Dispose();
        }

        // ReLU linear attention per head over [base heads | multiscale heads].
        Tensor outTokens = new Tensor(new TensorShape(s, c * (qm is null ? 1 : 2)), DType.F32);
        LinearAttentionHeads(q, k, v, outTokens, c, s, headOffset: 0);
        if (qm is not null)
        {
            LinearAttentionHeads(qm, km!, vm!, outTokens, c, s, headOffset: c);
            qm.Dispose(); km!.Dispose(); vm!.Dispose();
        }
        q.Dispose(); k.Dispose(); v.Dispose();

        Tensor projected = new Tensor(new TensorShape(s, c), DType.F32);
        backend.Linear(projected, outTokens, _projOutW!, _projOutB);
        outTokens.Dispose();

        Tensor result = FromTokens(projected, b, c, hh, ww);
        projected.Dispose();
        MusicDcaeDecoder.RmsNormChannels(result, _attnNormW!, _attnNormB);
        return result;
    }

    /// <summary>Per-head Sana LiteLA: <c>q' = relu(q), k' = relu(k); out = (V_aug·K'ᵀ·Q'ᵀ) row-normalized by the
    /// padded ones row</c> — linear in sequence length.</summary>
    private void LinearAttentionHeads(Tensor q, Tensor k, Tensor v, Tensor outTokens, int channels, long s, int headOffset)
    {
        int heads = channels / _headDim, d = _headDim;
        int outDim = (int)outTokens.Shape[1];
        float* qp = (float*)q.DataPointer;
        float* kp = (float*)k.DataPointer;
        float* vp = (float*)v.DataPointer;
        float* op = (float*)outTokens.DataPointer;
        double[,] vk = new double[d + 1, d];
        double[] acc = new double[d];
        for (int h = 0; h < heads; h++)
        {
            Array.Clear(vk);
            int baseC = h * d;
            // Accumulate VK = Σ_s v_aug(s) ⊗ relu(k(s)).
            for (long i = 0; i < s; i++)
            {
                long off = i * channels + baseC;
                for (int dk = 0; dk < d; dk++)
                {
                    float kv = MathF.Max(kp[off + dk], 0f);
                    if (kv == 0f) continue;
                    for (int dv = 0; dv < d; dv++) vk[dv, dk] += (double)vp[off + dv] * kv;
                    vk[d, dk] += kv;   // ones-row normalizer
                }
            }
            // Per token: out = VK·relu(q) / (norm + eps).
            for (long i = 0; i < s; i++)
            {
                long off = i * channels + baseC;
                double norm = 0;
                Array.Clear(acc);
                for (int dk = 0; dk < d; dk++)
                {
                    float qv = MathF.Max(qp[off + dk], 0f);
                    if (qv == 0f) continue;
                    for (int dv = 0; dv < d; dv++) acc[dv] += vk[dv, dk] * qv;
                    norm += vk[d, dk] * qv;
                }
                float inv = (float)(1.0 / (norm + 1e-15));
                long dst = i * outDim + headOffset + baseC;
                for (int dv = 0; dv < d; dv++) op[dst + dv] = (float)(acc[dv] * inv);
            }
        }
    }

    private Tensor GlumbConv(IBackend backend, Tensor x)
    {
        int b = (int)x.Shape[0], c = (int)x.Shape[1], hh = (int)x.Shape[2], ww = (int)x.Shape[3];
        int hidden2 = (int)_ffInvW!.Shape[0];
        Tensor inv = new Tensor(new TensorShape(b, hidden2, hh, ww), DType.F32);
        backend.Conv2D(inv, x, _ffInvW, _ffInvB, 1, 1, 0, 0);
        backend.Silu(inv, inv);
        Tensor depth = DepthwiseConv2d(inv, _ffDepthW!, _ffDepthB);
        inv.Dispose();

        // Chunk channels: x · SiLU(gate).
        int hidden = hidden2 / 2;
        Tensor gated = new Tensor(new TensorShape(b, hidden, hh, ww), DType.F32);
        float* dp = (float*)depth.DataPointer;
        float* gp = (float*)gated.DataPointer;
        long frame = (long)hh * ww;
        for (int ci = 0; ci < hidden; ci++)
            for (long i = 0; i < frame; i++)
            {
                float val = dp[(long)ci * frame + i];
                float gate = dp[(long)(hidden + ci) * frame + i];
                gp[(long)ci * frame + i] = val * (gate / (1f + MathF.Exp(-gate)));
            }
        depth.Dispose();

        Tensor outT = new Tensor(new TensorShape(b, c, hh, ww), DType.F32);
        backend.Conv2D(outT, gated, _ffPointW!, null, 1, 1, 0, 0);
        gated.Dispose();
        MusicDcaeDecoder.RmsNormChannels(outT, _ffNormW!, _ffNormB);
        return outT;
    }

    /// <summary>Depthwise 3×3 conv (groups = channels) — IBackend.Conv2D has no groups mode, and the GLUMB depth
    /// conv is tiny relative to the surrounding GEMMs, so a direct loop is fine on CPU.</summary>
    private static Tensor DepthwiseConv2d(Tensor x, Tensor w, Tensor? bias)
    {
        int c = (int)x.Shape[1], hh = (int)x.Shape[2], ww = (int)x.Shape[3];
        int k = (int)w.Shape[2];
        int pad = k / 2;
        Tensor o = new Tensor(x.Shape, DType.F32);
        float* xp = (float*)x.DataPointer;
        float* wp = (float*)w.DataPointer;
        float* bp = bias is null ? null : (float*)bias.DataPointer;
        float* op = (float*)o.DataPointer;
        long frame = (long)hh * ww;
        for (int ci = 0; ci < c; ci++)
        {
            float b = bp is null ? 0f : bp[ci];
            for (int y = 0; y < hh; y++)
                for (int xPos = 0; xPos < ww; xPos++)
                {
                    double acc = b;
                    for (int ky = 0; ky < k; ky++)
                    {
                        int sy = y + ky - pad;
                        if (sy < 0 || sy >= hh) continue;
                        for (int kx = 0; kx < k; kx++)
                        {
                            int sx = xPos + kx - pad;
                            if (sx < 0 || sx >= ww) continue;
                            acc += xp[(long)ci * frame + (long)sy * ww + sx] * wp[((long)ci * k + ky) * k + kx];
                        }
                    }
                    op[(long)ci * frame + (long)y * ww + xPos] = (float)acc;
                }
        }
        return o;
    }

    /// <summary>Grouped 1×1 conv (channels split into <paramref name="groups"/> independent slices).</summary>
    private static Tensor GroupedPointwiseConv2d(Tensor x, Tensor w, Tensor? bias, int groups)
    {
        int c = (int)x.Shape[1], hh = (int)x.Shape[2], ww = (int)x.Shape[3];
        int outC = (int)w.Shape[0];
        int inPerGroup = (int)w.Shape[1];
        int outPerGroup = outC / groups;
        Tensor o = new Tensor(new TensorShape(x.Shape[0], outC, hh, ww), DType.F32);
        float* xp = (float*)x.DataPointer;
        float* wp = (float*)w.DataPointer;
        float* bp = bias is null ? null : (float*)bias.DataPointer;
        float* op = (float*)o.DataPointer;
        long frame = (long)hh * ww;
        for (int g = 0; g < groups; g++)
            for (int oc = 0; oc < outPerGroup; oc++)
            {
                int outIdx = g * outPerGroup + oc;
                float b = bp is null ? 0f : bp[outIdx];
                for (long pos = 0; pos < frame; pos++)
                {
                    double acc = b;
                    for (int ic = 0; ic < inPerGroup; ic++)
                        acc += xp[(long)(g * inPerGroup + ic) * frame + pos] * wp[(long)outIdx * inPerGroup + ic];
                    op[(long)outIdx * frame + pos] = (float)acc;
                }
            }
        _ = c;
        return o;
    }

    private static Tensor Project(IBackend backend, Tensor tokens, Tensor w, int s, int c)
    {
        Tensor o = new Tensor(new TensorShape(s, c), DType.F32);
        backend.Linear(o, tokens, w, null);
        return o;
    }

    private static Tensor ToTokens(Tensor x, int c, long s)
    {
        Tensor o = new Tensor(new TensorShape(s, c), DType.F32);
        float* xp = (float*)x.DataPointer;
        float* op = (float*)o.DataPointer;
        for (int ci = 0; ci < c; ci++)
            for (long i = 0; i < s; i++)
                op[i * c + ci] = xp[(long)ci * s + i];
        return o;
    }

    private static Tensor FromTokens(Tensor tokens, int b, int c, int hh, int ww)
    {
        long s = (long)hh * ww;
        Tensor o = new Tensor(new TensorShape(b, c, hh, ww), DType.F32);
        float* tp = (float*)tokens.DataPointer;
        float* op = (float*)o.DataPointer;
        for (int ci = 0; ci < c; ci++)
            for (long i = 0; i < s; i++)
                op[(long)ci * s + i] = tp[i * c + ci];
        return o;
    }

    private Tensor FuseQkvMaps(Tensor q, Tensor k, Tensor v, int c, int hh, int ww)
    {
        long s = (long)hh * ww;
        Tensor fused = new Tensor(new TensorShape(1, 3 * c, hh, ww), DType.F32);
        float* fp = (float*)fused.DataPointer;
        float*[] sources = [(float*)q.DataPointer, (float*)k.DataPointer, (float*)v.DataPointer];
        for (int part = 0; part < 3; part++)
            for (int ci = 0; ci < c; ci++)
                for (long i = 0; i < s; i++)
                    fp[((long)(part * c + ci)) * s + i] = sources[part][i * c + ci];
        return fused;
    }

    private (Tensor Q, Tensor K, Tensor V) SplitQkvMaps(Tensor fused, int c, long s)
    {
        Tensor q = new Tensor(new TensorShape(s, c), DType.F32);
        Tensor k = new Tensor(new TensorShape(s, c), DType.F32);
        Tensor v = new Tensor(new TensorShape(s, c), DType.F32);
        float* fp = (float*)fused.DataPointer;
        float*[] dsts = [(float*)q.DataPointer, (float*)k.DataPointer, (float*)v.DataPointer];
        for (int part = 0; part < 3; part++)
            for (int ci = 0; ci < c; ci++)
                for (long i = 0; i < s; i++)
                    dsts[part][i * c + ci] = fp[((long)(part * c + ci)) * s + i];
        return (q, k, v);
    }

    private static void AddInPlace(Tensor target, Tensor value)
    {
        long n = target.Shape.ElementCount;
        float* tp = (float*)target.DataPointer;
        float* vp = (float*)value.DataPointer;
        for (long i = 0; i < n; i++) tp[i] += vp[i];
    }
}
