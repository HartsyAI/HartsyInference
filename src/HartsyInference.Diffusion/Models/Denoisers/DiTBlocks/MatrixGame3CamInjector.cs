using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Matrix-Game 3.0 per-block Plücker camera injection (<c>WanAttentionBlock</c>'s <c>cam_*</c> layers in
/// <c>wan/modules/model.py</c>). Applied between the self-attention residual and cross-attention on the memory-mode
/// blocks: refines the shared per-token Plücker embedding through a SiLU MLP with residual
/// (<c>cam_injector_layer1/2</c>), derives a per-token affine (<c>cam_scale_layer</c>/<c>cam_shift_layer</c>), and
/// modulates the hidden state <c>x = (1 + cam_scale)·x + cam_shift</c>. The global Plücker embedding it consumes is
/// produced once per forward by <see cref="MatrixGame3Transformer"/> (patch_embedding_wancamctrl + c2ws refine).</summary>
public sealed unsafe class MatrixGame3CamInjector
{
    private readonly int _dim;
    private Tensor? _inj1W, _inj1B, _inj2W, _inj2B, _scaleW, _scaleB, _shiftW, _shiftB;

    public MatrixGame3CamInjector(int dim) => _dim = dim;

    /// <summary>True once the block's <c>cam_*</c> weights are present and loaded.</summary>
    public bool Loaded => _inj1W is not null;

    public bool TryLoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        if (!w.TryGetValue($"{prefix}.cam_injector_layer1.weight", out _inj1W)) return false;
        w.TryGetValue($"{prefix}.cam_injector_layer1.bias", out _inj1B);
        _inj2W = w[$"{prefix}.cam_injector_layer2.weight"]; w.TryGetValue($"{prefix}.cam_injector_layer2.bias", out _inj2B);
        _scaleW = w[$"{prefix}.cam_scale_layer.weight"]; w.TryGetValue($"{prefix}.cam_scale_layer.bias", out _scaleB);
        _shiftW = w[$"{prefix}.cam_shift_layer.weight"]; w.TryGetValue($"{prefix}.cam_shift_layer.bias", out _shiftB);
        return true;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _inj1W, _inj1B, _inj2W, _inj2B, _scaleW, _scaleB, _shiftW, _shiftB })
            if (t is not null) yield return t;
    }

    /// <summary>Modulates <paramref name="hidden"/> <c>[S, dim]</c> in place given the shared per-token Plücker embedding <paramref name="pluckerEmb"/> <c>[S, dim]</c>.</summary>
    public void Apply(IBackend backend, Tensor hidden, Tensor pluckerEmb)
    {
        int s = (int)hidden.Shape[0];
        // c2ws = cam_injector_layer2(silu(cam_injector_layer1(plucker))) + plucker
        Tensor h1 = new(new TensorShape(s, _dim), DType.F32); backend.Linear(h1, pluckerEmb, _inj1W!, _inj1B);
        Tensor a = new(h1.Shape, DType.F32); backend.Silu(a, h1); h1.Dispose();
        Tensor h2 = new(new TensorShape(s, _dim), DType.F32); backend.Linear(h2, a, _inj2W!, _inj2B); a.Dispose();
        Tensor c2ws = new(new TensorShape(s, _dim), DType.F32); backend.Add(c2ws, h2, pluckerEmb); h2.Dispose();

        Tensor scale = new(new TensorShape(s, _dim), DType.F32); backend.Linear(scale, c2ws, _scaleW!, _scaleB);
        Tensor shift = new(new TensorShape(s, _dim), DType.F32); backend.Linear(shift, c2ws, _shiftW!, _shiftB);
        c2ws.Dispose();

        // hidden = (1 + scale)·hidden + shift  (per-token affine, elementwise over [S, dim])
        Tensor scaleP1 = new(new TensorShape(s, _dim), DType.F32); backend.AddScalar(scaleP1, scale, 1.0f); scale.Dispose();
        Tensor prod = new(new TensorShape(s, _dim), DType.F32); backend.Mul(prod, hidden, scaleP1); scaleP1.Dispose();
        backend.Add(hidden, prod, shift);   // hidden is output-only here (reads prod + shift)
        prod.Dispose(); shift.Dispose();
    }
}
