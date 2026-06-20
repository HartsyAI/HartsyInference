using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Wan VACE control block (<c>WanVACETransformerBlock</c>), ported from diffusers <c>transformer_wan_vace.py</c>.
/// Structurally a base <see cref="WanVideoBlock"/> (self-attn + 3D RoPE / text cross-attn / FFN, 6-param AdaLN) that
/// runs on the <b>control</b> stream, wrapped by an optional input projection (block 0 only: <c>control = proj_in(control)
/// + hidden</c>) and an always-on output projection that emits the per-layer hint (<c>conditioning = proj_out(control)</c>).
/// The hint is added back into the main stream at the matching <c>vace_layers</c> index, scaled by the per-layer
/// control scale.</summary>
public sealed unsafe class WanVaceBlock
{
    private readonly int _dim;
    private readonly bool _applyInputProj;
    private readonly WanVideoBlock _inner;
    private Tensor? _projInW, _projInB, _projOutW, _projOutB;

    public WanVaceBlock(WanVideoConfig c, bool applyInputProjection)
    {
        _dim = c.InnerDim;
        _applyInputProj = applyInputProjection;
        _inner = new WanVideoBlock(c, crossAttnNorm: true);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _inner.LoadWeights(w, prefix);   // attn1/attn2/ffn/norm2/scale_shift_table under vace_blocks.{i}.*
        if (_applyInputProj)
        {
            _projInW = w[$"{prefix}.proj_in.weight"]; w.TryGetValue($"{prefix}.proj_in.bias", out _projInB);
        }
        _projOutW = w[$"{prefix}.proj_out.weight"]; w.TryGetValue($"{prefix}.proj_out.bias", out _projOutB);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _inner.EnumerateWeights()) yield return t;
        foreach (Tensor? t in new[] { _projInW, _projInB, _projOutW, _projOutB }) if (t is not null) yield return t;
    }

    /// <summary>Runs one VACE control block. <paramref name="control"/> is the control stream <c>[S, dim]</c>;
    /// <paramref name="hidden"/> is the current main stream (read only — used by the block-0 input projection). Returns
    /// the per-layer hint (<c>[S, dim]</c>, to add into the main stream) and the updated control stream; the input
    /// <paramref name="control"/> is consumed.</summary>
    public (Tensor Hint, Tensor Control) Forward(IBackend backend, Tensor hidden, Tensor encoder, Tensor control,
        Tensor temb, WanRope rope, Tensor cos, Tensor sin, int tokensPerGroup)
    {
        int s = (int)control.Shape[0];
        Tensor cur = control;
        if (_applyInputProj)
        {
            Tensor proj = new Tensor(new TensorShape(s, _dim), DType.F32);
            backend.Linear(proj, control, _projInW!, _projInB);
            AddInPlace(proj, hidden);     // control = proj_in(control) + hidden
            control.Dispose();
            cur = proj;
        }

        Tensor updated = _inner.Forward(backend, cur, encoder, temb, rope, cos, sin, tokensPerGroup);
        cur.Dispose();

        Tensor hint = new Tensor(new TensorShape(s, _dim), DType.F32);
        backend.Linear(hint, updated, _projOutW!, _projOutB);
        return (hint, updated);
    }

    private static void AddInPlace(Tensor acc, Tensor add)
    {
        long n = acc.Shape.ElementCount;
        float* ap = (float*)acc.DataPointer, dp = (float*)add.DataPointer;
        for (long i = 0; i < n; i++) ap[i] += dp[i];
    }
}
