using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Norm;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.TrellisV2;

/// <summary>
/// Upstream <c>ModulatedTransformerCrossBlock</c> / <c>ModulatedSparseTransformerCrossBlock</c> with shared modulation: the
/// block's own <c>modulation</c> parameter is added to the model-wide adaLN output and chunked into shift, scale and gate for
/// the self-attention and for the MLP. Self-attention (non-affine norm1, QK RMS norm, 3D RoPE) and the MLP (non-affine norm3,
/// tanh GELU) are modulated and gated; the cross-attention to the image tokens (affine norm2, QK RMS norm) is ungated and
/// has no rotation. Borrows its weights from <see cref="Trellis2FlowWeights"/>.
/// </summary>
internal sealed class Trellis2FlowBlock : IDisposable
{
    private const float NormEps = 1e-6f;

    private readonly int _width, _heads, _headDim, _mlpHidden;
    private readonly float _scale, _headNormEps;
    private readonly Tensor _ones;
    private readonly Tensor _modulation;
    private readonly Tensor _qkvW, _qkvB, _outW, _outB, _norm2W, _norm2B;
    private readonly Tensor _qW, _qB, _kvW, _kvB, _crossOutW, _crossOutB, _mlp0W, _mlp0B, _mlp2W, _mlp2B;
    private readonly Tensor? _selfQGamma, _selfKGamma, _crossQGamma, _crossKGamma;
    private int _disposed;

    internal Trellis2FlowBlock(Trellis2FlowWeights weights, int index)
    {
        Trellis2FlowConfig config = weights.Config;
        string p = $"blocks.{index}.";
        _width = config.ModelChannels;
        _heads = config.NumHeads;
        _headDim = config.HeadDim;
        _mlpHidden = config.MlpHidden;
        _scale = 1f / MathF.Sqrt(_headDim);
        // F.normalize clamps the norm at 1e-12, which is 1e-24 / D on the mean square, so a zero head stays zero.
        _headNormEps = 1e-24f / _headDim;
        _ones = new Tensor(new TensorShape(_headDim), DType.F32);
        _ones.AsSpan<float>().Fill(1f);
        _modulation = weights[p + "modulation"].Reshape(new TensorShape(1, 6L * _width));
        _qkvW = weights[p + "self_attn.to_qkv.weight"];
        _qkvB = weights[p + "self_attn.to_qkv.bias"];
        _outW = weights[p + "self_attn.to_out.weight"];
        _outB = weights[p + "self_attn.to_out.bias"];
        _norm2W = weights[p + "norm2.weight"];
        _norm2B = weights[p + "norm2.bias"];
        _qW = weights[p + "cross_attn.to_q.weight"];
        _qB = weights[p + "cross_attn.to_q.bias"];
        _kvW = weights[p + "cross_attn.to_kv.weight"];
        _kvB = weights[p + "cross_attn.to_kv.bias"];
        _crossOutW = weights[p + "cross_attn.to_out.weight"];
        _crossOutB = weights[p + "cross_attn.to_out.bias"];
        _mlp0W = weights[p + "mlp.mlp.0.weight"];
        _mlp0B = weights[p + "mlp.mlp.0.bias"];
        _mlp2W = weights[p + "mlp.mlp.2.weight"];
        _mlp2B = weights[p + "mlp.mlp.2.bias"];
        if (config.QkRmsNorm)
        {
            _selfQGamma = weights[p + "self_attn.q_rms_norm.gamma"];
            _selfKGamma = weights[p + "self_attn.k_rms_norm.gamma"];
        }

        if (config.QkRmsNormCross)
        {
            _crossQGamma = weights[p + "cross_attn.q_rms_norm.gamma"];
            _crossKGamma = weights[p + "cross_attn.k_rms_norm.gamma"];
        }
    }

    /// <summary>The tensors the block reads, for backend weight residency. The modulation is its reshaped view.</summary>
    internal IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all =
        [
            _ones, _modulation, _qkvW, _qkvB, _outW, _outB, _norm2W, _norm2B, _qW, _qB, _kvW, _kvB, _crossOutW, _crossOutB,
            _mlp0W, _mlp0B, _mlp2W, _mlp2B, _selfQGamma, _selfKGamma, _crossQGamma, _crossKGamma,
        ];
        foreach (Tensor? t in all)
        {
            if (t is not null)
                yield return t;
        }
    }

    /// <summary>One block over <c>x [1, N, C]</c>. <paramref name="sharedMod"/> is the model-wide adaLN output <c>[1, 6C]</c> and
    /// <paramref name="cond"/> the image tokens <c>[1, L, condChannels]</c>. Returns a new tensor; borrows every input.</summary>
    internal Tensor Forward(IBackend backend, Tensor x, Tensor sharedMod, Tensor cond, Trellis2RopeTable rope)
    {
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        int n = (int)x.Shape[1];
        Tensor[] m = SplitModulation(backend, sharedMod);
        try
        {
            // m: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp.
            Tensor x1 = SelfAttention(backend, x, n, m[0], m[1], m[2], rope);
            Tensor x2 = CrossAttention(backend, x1, n, cond);
            x1.Dispose();
            Tensor x3 = FeedForward(backend, x2, n, m[3], m[4], m[5]);
            x2.Dispose();
            return x3;
        }
        finally
        {
            foreach (Tensor t in m)
                t.Dispose();
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _ones.Dispose();
    }

    private Tensor[] SplitModulation(IBackend backend, Tensor sharedMod)
    {
        Tensor sum = new(new TensorShape(1, 6L * _width), DType.F32);
        backend.Add(sum, _modulation, sharedMod);
        Tensor[] parts = new Tensor[6];
        try
        {
            for (int i = 0; i < parts.Length; i++)
            {
                parts[i] = new Tensor(new TensorShape(1, _width), DType.F32);
                backend.SliceLastDim(parts[i], sum, i * _width);
            }
        }
        catch
        {
            foreach (Tensor? t in parts)
                t?.Dispose();
            throw;
        }
        finally
        {
            sum.Dispose();
        }

        return parts;
    }

    private Tensor Modulated(IBackend backend, Tensor x, Tensor scale, Tensor shift)
    {
        Tensor h = new(x.Shape, DType.F32);
        backend.NormModulate(NormModulateCall.Rows(h, x, NormKind.LayerNorm, NormEps)
            .WithModulation(ModulationSpec.Broadcast(x, scale, shift, ScaleMode.OnePlus)));
        return h;
    }

    private Tensor SelfAttention(IBackend backend, Tensor x, int n, Tensor shift, Tensor scale, Tensor gate, Trellis2RopeTable rope)
    {
        Tensor h = Modulated(backend, x, scale, shift);
        Tensor qkv = new(new TensorShape(1, n, 3L * _width), DType.F32);
        backend.Linear(qkv, h, _qkvW, _qkvB);
        h.Dispose();
        Tensor q = HeadMajor(backend, qkv, 0, n);
        Tensor k = HeadMajor(backend, qkv, _width, n);
        Tensor v = HeadMajor(backend, qkv, 2 * _width, n);
        qkv.Dispose();
        q = HeadNorm(backend, q, _selfQGamma);
        k = HeadNorm(backend, k, _selfKGamma);
        rope.Apply(backend, q);
        rope.Apply(backend, k);
        Tensor ctx = Attend(backend, q, k, v, n);
        Tensor proj = Linear(backend, ctx, _outW, _outB, _width);
        ctx.Dispose();
        Tensor x1 = new(x.Shape, DType.F32);
        backend.GatedResidualLastDim(x1, x, proj, gate);
        proj.Dispose();
        return x1;
    }

    // Cross-attention keys and values are recomputed per call because one block sees both the conditional and the
    // unconditional tokens at every step.
    private Tensor CrossAttention(IBackend backend, Tensor x, int n, Tensor cond)
    {
        int lc = (int)cond.Shape[1];
        Tensor h = new(x.Shape, DType.F32);
        backend.LayerNorm(h, x, _norm2W, _norm2B, NormEps);
        Tensor qFlat = Linear(backend, h, _qW, _qB, _width);
        h.Dispose();
        Tensor q = HeadMajor(backend, qFlat, 0, n);
        qFlat.Dispose();
        Tensor kv = Linear(backend, cond, _kvW, _kvB, 2L * _width);
        Tensor k = HeadMajor(backend, kv, 0, lc);
        Tensor v = HeadMajor(backend, kv, _width, lc);
        kv.Dispose();
        q = HeadNorm(backend, q, _crossQGamma);
        k = HeadNorm(backend, k, _crossKGamma);
        Tensor ctx = Attend(backend, q, k, v, n);
        Tensor proj = Linear(backend, ctx, _crossOutW, _crossOutB, _width);
        ctx.Dispose();
        Tensor sum = new(x.Shape, DType.F32);
        backend.Add(sum, x, proj);
        proj.Dispose();
        return sum;
    }

    private Tensor FeedForward(IBackend backend, Tensor x, int n, Tensor shift, Tensor scale, Tensor gate)
    {
        Tensor h = Modulated(backend, x, scale, shift);
        Tensor hidden = Linear(backend, h, _mlp0W, _mlp0B, _mlpHidden);
        h.Dispose();
        Tensor act = new(hidden.Shape, DType.F32);
        backend.Gelu(act, hidden);
        hidden.Dispose();
        Tensor down = Linear(backend, act, _mlp2W, _mlp2B, _width);
        act.Dispose();
        Tensor x3 = new(x.Shape, DType.F32);
        backend.GatedResidualLastDim(x3, x, down, gate);
        down.Dispose();
        return x3;
    }

    private static Tensor Linear(IBackend backend, Tensor x, Tensor weight, Tensor bias, long outDim)
    {
        Tensor o = new(new TensorShape(1, x.Shape[1], outDim), DType.F32);
        backend.Linear(o, x, weight, bias);
        return o;
    }

    /// <summary>Columns <c>[offset, offset + C)</c> of a packed <c>[1, tokens, k·C]</c> tensor as head-major <c>[1, H, tokens, D]</c>.</summary>
    private Tensor HeadMajor(IBackend backend, Tensor packed, int offset, int tokens)
    {
        Tensor tokenMajor = new(new TensorShape(1, tokens, _heads, _headDim), DType.F32);
        backend.SliceLastDim(tokenMajor, packed, offset);
        Tensor o = new(new TensorShape(1, _heads, tokens, _headDim), DType.F32);
        backend.Permute0213(o, tokenMajor, tokens, _heads, _headDim);
        tokenMajor.Dispose();
        return o;
    }

    /// <summary>Per-head <c>F.normalize(x) · gamma · sqrt(D)</c> as RMS norm times <c>gamma [H, D]</c>; consumes <paramref name="heads"/>. A null gamma means no norm.</summary>
    private Tensor HeadNorm(IBackend backend, Tensor heads, Tensor? gamma)
    {
        if (gamma is null)
            return heads;
        Tensor rms = new(heads.Shape, DType.F32);
        backend.RmsNorm(rms, heads, _ones, _headNormEps);
        heads.Dispose();
        Tensor o = new(rms.Shape, DType.F32);
        backend.AffineBroadcastLastDim(o, rms, gamma, null);
        rms.Dispose();
        return o;
    }

    /// <summary>Attention over head-major operands, returned flat as <c>[1, tokens, C]</c>; consumes q, k and v.</summary>
    private Tensor Attend(IBackend backend, Tensor q, Tensor k, Tensor v, int tokens)
    {
        Tensor att = new(new TensorShape(1, _heads, tokens, _headDim), DType.F32);
        backend.ScaledDotProductAttention(att, q, k, v, null, _scale);
        q.Dispose();
        k.Dispose();
        v.Dispose();
        Tensor flat = new(new TensorShape(1, tokens, _width), DType.F32);
        backend.Permute0213(flat, att, _heads, tokens, _headDim);
        att.Dispose();
        return flat;
    }
}
