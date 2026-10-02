using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Auk;

/// <summary>Bias-free SwiGLU feed-forward: <c>linear_out(silu(first half) * second half of linear_in(x))</c>.</summary>
public sealed class AukSwiGluFfn
{
    private Tensor? _inW;
    private Tensor? _outW;
    private int _dim;
    private int _inner;

    public void Load(IReadOnlyDictionary<string, Tensor> weights, string prefix, int dim, int inner)
    {
        _inW = AukOps.Take(weights, $"{prefix}.linear_in.weight", 2 * inner, dim);
        _outW = AukOps.Take(weights, $"{prefix}.linear_out.weight", dim, inner);
        _dim = dim;
        _inner = inner;
    }

    /// <summary>Input and output are <c>[1, n, dim]</c>; the caller owns the result.</summary>
    public Tensor Forward(IBackend backend, Tensor x, int n)
    {
        Tensor gateUp = WhisperOps.ProjectLinear(backend, x, _inW!, null, 1, n, _dim, 2 * _inner);
        Tensor act = new(new TensorShape(1, n, _inner), DType.F32);
        backend.GluActivate(act, gateUp, _inner, false);
        gateUp.Dispose();
        Tensor result = WhisperOps.ProjectLinear(backend, act, _outW!, null, 1, n, _inner, _dim);
        act.Dispose();
        return result;
    }

    /// <summary>The block's MLP half: <c>x + gate_mlp * ffn(LN(x) * scale_mlp + shift_mlp)</c>; the caller owns the result and keeps <paramref name="x"/>.</summary>
    public Tensor ForwardResidual(IBackend backend, Tensor x, AukModulation mod, int n, float eps)
    {
        Tensor modded = AukOps.AdaNorm(backend, x, mod.ScaleMlp, mod.ShiftMlp, eps);
        Tensor ff = Forward(backend, modded, n);
        modded.Dispose();
        Tensor result = AukOps.GatedResidual(backend, x, ff, mod.GateMlp);
        ff.Dispose();
        return result;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_inW is not null) yield return _inW;
        if (_outW is not null) yield return _outW;
    }
}
