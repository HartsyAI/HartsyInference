using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Norm;
using HartsyInference.Core.Layers;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Idol;

/// <summary>One Idol block (upstream <c>UnifiedTransformerBlock</c>): rotary self-attention and a tanh-GELU MLP over <c>[1, L, C]</c>.
/// A modulated block (noise refiner, joint blocks) adds its own shift table to the shared time modulation, normalises without affine,
/// and gates both residuals; the image-condition refiner is not modulated, uses affine LayerNorms and plain residuals. The rotary angles
/// come from the block's own <see cref="IdolRePo"/> on its input, then the per-head RMS norm of q and k follows the rotation.
/// Borrows its weights.</summary>
internal sealed class IdolBlock : IDisposable
{
    private const float NormEps = 1e-6f;

    private readonly int _width, _heads, _headDim;
    private readonly float _scale;
    private readonly bool _modulated;
    private readonly HeadRmsNorm _headNorm;
    private readonly Tensor? _shiftTable, _norm1W, _norm1B, _norm2W, _norm2B;
    private readonly Tensor _qkvW, _qkvB, _outW, _outB, _qGamma, _kGamma, _mlp0W, _mlp0B, _mlp2W, _mlp2B;
    private int _disposed;

    internal IdolBlock(IdolWeights weights, string prefix, IdolConfig config, bool modulated)
    {
        _width = config.ModelChannels;
        _heads = config.Heads;
        _headDim = config.HeadDim;
        _scale = 1f / MathF.Sqrt(_headDim);
        _modulated = modulated;
        _headNorm = new HeadRmsNorm(_headDim);
        _qkvW = weights[$"{prefix}.attn.qkv.weight"]; _qkvB = weights[$"{prefix}.attn.qkv.bias"];
        _outW = weights[$"{prefix}.attn.out.weight"]; _outB = weights[$"{prefix}.attn.out.bias"];
        // gamma [heads, headDim] as the flat [heads·headDim] the token-major norm broadcasts.
        _qGamma = weights[$"{prefix}.attn.q_norm.gamma"].Reshape(new TensorShape(_width));
        _kGamma = weights[$"{prefix}.attn.k_norm.gamma"].Reshape(new TensorShape(_width));
        _mlp0W = weights[$"{prefix}.mlp.mlp.0.weight"]; _mlp0B = weights[$"{prefix}.mlp.mlp.0.bias"];
        _mlp2W = weights[$"{prefix}.mlp.mlp.2.weight"]; _mlp2B = weights[$"{prefix}.mlp.mlp.2.bias"];
        if (modulated)
        {
            _shiftTable = weights[$"{prefix}.shift_table"];
        }
        else
        {
            _norm1W = weights[$"{prefix}.norm1.weight"]; _norm1B = weights[$"{prefix}.norm1.bias"];
            _norm2W = weights[$"{prefix}.norm2.weight"]; _norm2B = weights[$"{prefix}.norm2.bias"];
        }
    }

    /// <summary>The unit scale of the head norm, the one tensor the block owns; the weights it reads belong to <see cref="IdolWeights"/>.</summary>
    internal Tensor HeadNormOnes => _headNorm.Ones;

    /// <summary><paramref name="x"/> <c>[1, L, C]</c> through the block; <paramref name="timeModulation"/> is the shared <c>[1, 6C]</c>
    /// adaLN output (null for the unmodulated block). Returns a new tensor; borrows both inputs.</summary>
    internal Tensor Forward(IBackend backend, Tensor x, Tensor? timeModulation, IdolRePo repo, Action<Tensor, Tensor>? rotationTap = null)
    {
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        int n = (int)x.Shape[1];
        Tensor[]? m = _modulated ? SplitModulation(backend, timeModulation ?? throw new ArgumentNullException(nameof(timeModulation))) : null;
        try
        {
            // m: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp.
            Tensor x1 = SelfAttention(backend, x, n, m, repo, rotationTap);
            Tensor x2 = FeedForward(backend, x1, m);
            x1.Dispose();
            return x2;
        }
        finally
        {
            if (m is not null)
                foreach (Tensor t in m)
                    t.Dispose();
        }
    }

    private Tensor[] SplitModulation(IBackend backend, Tensor timeModulation)
    {
        Tensor sum = new(new TensorShape(1, 6L * _width), DType.F32);
        backend.Add(sum, timeModulation, _shiftTable!);
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

    private Tensor Normed(IBackend backend, Tensor x, Tensor? scale, Tensor? shift, Tensor? weight, Tensor? bias)
    {
        Tensor h = new(x.Shape, DType.F32);
        if (_modulated)
            backend.NormModulate(NormModulateCall.Rows(h, x, NormKind.LayerNorm, NormEps).WithModulation(ModulationSpec.Broadcast(x, scale!, shift!, ScaleMode.OnePlus)));
        else
            backend.LayerNorm(h, x, weight!, bias!, NormEps);
        return h;
    }

    private Tensor SelfAttention(IBackend backend, Tensor x, int n, Tensor[]? m, IdolRePo repo, Action<Tensor, Tensor>? rotationTap)
    {
        (Tensor cos, Tensor sin) = repo.Angles(backend, x);
        try
        {
            rotationTap?.Invoke(cos, sin);
            using Tensor h = Normed(backend, x, m?[1], m?[0], _norm1W, _norm1B);
            using Tensor qkv = new(new TensorShape(1, n, 3L * _width), DType.F32);
            backend.Linear(qkv, h, _qkvW, _qkvB);
            Tensor q = IdolAttention.Slice(backend, qkv, 0, n, _heads, _headDim);
            Tensor k = IdolAttention.Slice(backend, qkv, _width, n, _heads, _headDim);
            Tensor v = IdolAttention.Slice(backend, qkv, 2 * _width, n, _heads, _headDim);
            repo.Apply(backend, q, cos, sin);
            repo.Apply(backend, k, cos, sin);
            Tensor qn = _headNorm.Apply(backend, q, _qGamma, new TensorShape(1, n, _width));
            Tensor kn = _headNorm.Apply(backend, k, _kGamma, new TensorShape(1, n, _width));
            using Tensor ctx = IdolAttention.Attend(backend, qn, kn, v, n, n, _heads, _headDim, _scale);
            using Tensor proj = Linear(backend, ctx, _outW, _outB, _width);
            Tensor x1 = new(x.Shape, DType.F32);
            if (_modulated)
                backend.GatedResidualLastDim(x1, x, proj, m![2]);
            else
                backend.Add(x1, x, proj);
            return x1;
        }
        finally
        {
            cos.Dispose();
            sin.Dispose();
        }
    }

    private Tensor FeedForward(IBackend backend, Tensor x, Tensor[]? m)
    {
        using Tensor h = Normed(backend, x, m?[4], m?[3], _norm2W, _norm2B);
        using Tensor hidden = Linear(backend, h, _mlp0W, _mlp0B, _mlp0W.Shape[0]);
        using Tensor act = new(hidden.Shape, DType.F32);
        backend.Gelu(act, hidden);
        using Tensor down = Linear(backend, act, _mlp2W, _mlp2B, _width);
        Tensor x2 = new(x.Shape, DType.F32);
        if (_modulated)
            backend.GatedResidualLastDim(x2, x, down, m![5]);
        else
            backend.Add(x2, x, down);
        return x2;
    }

    private static Tensor Linear(IBackend backend, Tensor x, Tensor weight, Tensor bias, long outDim)
    {
        Tensor o = new(new TensorShape(1, x.Shape[1], outDim), DType.F32);
        backend.Linear(o, x, weight, bias);
        return o;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _headNorm.Dispose();
    }
}
