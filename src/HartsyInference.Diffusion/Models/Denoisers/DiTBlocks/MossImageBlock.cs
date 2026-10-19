using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>MOSS-Image joint block (<c>_Block</c>) over <c>[image ; text]</c>, modulated by the one
/// <see cref="MossImageModulation"/> every block shares. No biases. Reference images of an edit enter either as cached
/// keys and values (<see cref="ForwardPrefix"/>, then the cached <c>Forward</c>) or as a third segment modulated at
/// timestep zero (<see cref="ForwardSegmented"/>).
/// <code>
/// y = rms(x)·w1·(1 + scale_a) + shift_a
/// q, k = rope(rms_head(q(y))), rope(rms_head(k(y)))
/// x = x + gate_a · o(attention(q, k, v(y)))
/// y = rms(x)·w2·(1 + scale_f) + shift_f
/// x = x + gate_f · ffn_down(situ(ffn_up_gate(y)))      // ffn_up_gate rows are [hidden | gate]
/// </code>
/// The residual stream stays F32; the branch between the two norms and the two gated adds runs in the caller's
/// activation dtype (F16 on backends with F16 activations), which SiTU's soft caps keep in range.</summary>
public sealed class MossImageBlock(int hidden, int numHeads, int ffnHidden, float eps)
{
    private readonly int _hidden = hidden;
    private readonly int _numHeads = numHeads;
    private readonly int _headDim = hidden / numHeads;
    private readonly int _ffnHidden = ffnHidden;
    private readonly float _eps = eps;
    private readonly QkNorm _normQ = new(hidden / numHeads, eps);
    private readonly QkNorm _normK = new(hidden / numHeads, eps);
    private Tensor? _norm1, _norm2;
    private Tensor? _q, _k, _v, _o, _ffnUpGate, _ffnDown;

    /// <summary>Binds this block's weights under <paramref name="prefix"/> (<c>layers.N</c>).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _norm1 = TensorCasts.LoadF32(weights, $"{prefix}.norm1.weight");
        _norm2 = TensorCasts.LoadF32(weights, $"{prefix}.norm2.weight");
        _normQ.LoadWeights(weights[$"{prefix}.q_norm.weight"]);
        _normK.LoadWeights(weights[$"{prefix}.k_norm.weight"]);
        if (weights.TryGetValue($"{prefix}.proj.q.weight", out Tensor? q))
        {
            _q = q;
            _k = weights[$"{prefix}.proj.k.weight"];
            _v = weights[$"{prefix}.proj.v.weight"];
        }
        else
        {
            string qkvKey = $"{prefix}.proj.qkv.weight";
            Tensor qkv = weights[qkvKey];
            _q = HiddenRows(qkv, 0, _hidden, qkvKey);
            _k = HiddenRows(qkv, 1, _hidden, qkvKey);
            _v = HiddenRows(qkv, 2, _hidden, qkvKey);
        }
        _o = weights[$"{prefix}.proj.o.weight"];
        _ffnUpGate = weights[$"{prefix}.proj.ffn_up_gate.weight"];
        _ffnDown = weights[$"{prefix}.proj.ffn_down.weight"];
    }

    /// <summary>The <paramref name="index"/>th run of <paramref name="hidden"/> rows of a fused projection, as a view that
    /// carries its own slice of the per-row quantization companion.</summary>
    private static Tensor HiddenRows(Tensor fused, int index, int hidden, string key)
    {
        Tensor rows = fused.SliceRows(index * hidden, hidden);
        rows.QuantInfo = fused.QuantInfo?.SliceRows(index * hidden, hidden, key);
        return rows;
    }

    /// <summary>Every weight this block reads.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_norm1, _norm2, _q, _k, _v, _o, _ffnUpGate, _ffnDown];
        foreach (Tensor? w in all)
        {
            if (w is not null) yield return w;
        }
        foreach (Tensor w in _normQ.EnumerateWeights()) yield return w;
        foreach (Tensor w in _normK.EnumerateWeights()) yield return w;
    }

    /// <summary>Runs the block on the F32 residual stream <paramref name="x"/> <c>[1, S, hidden]</c> and returns a new
    /// F32 tensor; <paramref name="cos"/>/<paramref name="sin"/> are the joint sequence's <c>[1, S, headDim]</c> tables
    /// and <paramref name="act"/> the branch dtype.</summary>
    public Tensor Forward(IBackend backend, Tensor x, MossImageModulation modulation, Tensor cos, Tensor sin, DType act) =>
        Forward(backend, x, modulation, cos, sin, act, null, null);

    /// <summary><see cref="Forward(IBackend, Tensor, MossImageModulation, Tensor, Tensor, DType)"/> with cached reference
    /// keys and values (<see cref="ReferenceKvCache"/>, <c>[1, S_ref, heads, headDim]</c> in <paramref name="act"/>)
    /// appended after the live ones, so every live query also attends to the references.</summary>
    public Tensor Forward(IBackend backend, Tensor x, MossImageModulation modulation, Tensor cos, Tensor sin, DType act,
        Tensor? cachedKeys, Tensor? cachedValues)
    {
        ArgumentNullException.ThrowIfNull(modulation);
        if ((cachedKeys is null) != (cachedValues is null))
            throw new ArgumentException("Cached reference keys and values come together.", nameof(cachedValues));
        int seq = (int)x.Shape[1];
        TensorShape flat = new(1, seq, _hidden);
        Tensor y = new(flat, act);
        backend.RmsNormModulate(y, x, _norm1!, modulation.ScaleAttn, modulation.ShiftAttn, _eps);
        (Tensor q, Tensor k, Tensor v) = ProjectQkv(backend, y, seq, act, cos, sin);
        int kvSeq = seq;
        if (cachedKeys is not null)
        {
            try
            {
                Tensor keys = ReferenceKvCache.ConcatTokens(backend, k, cachedKeys);
                k.Dispose();
                k = keys;
                Tensor values = ReferenceKvCache.ConcatTokens(backend, v, cachedValues!);
                v.Dispose();
                v = values;
            }
            catch
            {
                q.Dispose();
                k.Dispose();
                v.Dispose();
                throw;
            }
            kvSeq = (int)k.Shape[1];
        }
        Tensor attn = TokenMajorAttention.Run(backend, q, k, v, seq, kvSeq, _numHeads, _headDim);
        return Finish(backend, x, attn, modulation, act);
    }

    /// <summary>Runs the block on reference tokens alone (a prepass at the references' own modulation, references
    /// attending only to each other) and hands back this block's post-norm, rotated <paramref name="keys"/> and its
    /// <paramref name="values"/>, both <c>[1, S_ref, heads, headDim]</c> in <paramref name="act"/>, for a
    /// <see cref="ReferenceKvCache"/>. Returns the next F32 reference stream.</summary>
    public Tensor ForwardPrefix(IBackend backend, Tensor x, MossImageModulation modulation, Tensor cos, Tensor sin, DType act,
        out Tensor keys, out Tensor values)
    {
        ArgumentNullException.ThrowIfNull(modulation);
        int seq = (int)x.Shape[1];
        Tensor y = new(new TensorShape(1, seq, _hidden), act);
        backend.RmsNormModulate(y, x, _norm1!, modulation.ScaleAttn, modulation.ShiftAttn, _eps);
        (Tensor q, Tensor k, Tensor v) = ProjectQkv(backend, y, seq, act, cos, sin);
        Tensor attn;
        try
        {
            attn = TokenMajorAttention.Run(backend, q, k, v, seq, seq, _numHeads, _headDim, disposeInputs: false);
        }
        catch
        {
            k.Dispose();
            v.Dispose();
            throw;
        }
        finally
        {
            q.Dispose();
        }
        Tensor result;
        try
        {
            result = Finish(backend, x, attn, modulation, act);
        }
        catch
        {
            k.Dispose();
            v.Dispose();
            throw;
        }
        keys = k;
        values = v;
        return result;
    }

    /// <summary>Runs the block over a sequence whose segments take different modulation (live tokens at the step's
    /// timestep, reference tokens at timestep zero): each norm goes F32 through the segment tables, then to
    /// <paramref name="act"/>; both gated adds read the segment gates.</summary>
    public Tensor ForwardSegmented(IBackend backend, Tensor x, SegmentModulation attention, SegmentModulation ffn, Tensor cos,
        Tensor sin, DType act)
    {
        ArgumentNullException.ThrowIfNull(attention);
        ArgumentNullException.ThrowIfNull(ffn);
        int seq = (int)x.Shape[1];
        Tensor y = NormSegmented(backend, x, _norm1!, attention, act);
        (Tensor q, Tensor k, Tensor v) = ProjectQkv(backend, y, seq, act, cos, sin);
        Tensor attn = TokenMajorAttention.Run(backend, q, k, v, seq, seq, _numHeads, _headDim);
        Tensor projected = OutProject(backend, attn, seq);
        Tensor x1 = attention.GatedResidual(backend, x, projected);
        projected.Dispose();

        Tensor y2 = NormSegmented(backend, x1, _norm2!, ffn, act);
        Tensor down = Ffn(backend, y2, seq, act);
        Tensor result = ffn.GatedResidual(backend, x1, down);
        x1.Dispose();
        down.Dispose();
        return result;
    }

    /// <summary>Normalized, rotated Q and K and raw V from the modulated input <paramref name="y"/>, which it disposes.</summary>
    private (Tensor Q, Tensor K, Tensor V) ProjectQkv(IBackend backend, Tensor y, int seq, DType act, Tensor cos, Tensor sin)
    {
        TensorShape heads = new(1, seq, _numHeads, _headDim);
        Tensor q = new(heads, act);
        backend.Linear(q, y, _q!, null);
        Tensor k = new(heads, act);
        backend.Linear(k, y, _k!, null);
        Tensor v = new(heads, act);
        backend.Linear(v, y, _v!, null);
        y.Dispose();
        Tensor qNormed = new(heads, act);
        backend.RmsNorm(qNormed, q, _normQ.Weight, _normQ.Eps);
        q.Dispose();
        Tensor kNormed = new(heads, act);
        backend.RmsNorm(kNormed, k, _normK.Weight, _normK.Eps);
        k.Dispose();
        backend.ApplyRope(qNormed, kNormed, cos, sin);
        return (qNormed, kNormed, v);
    }

    /// <summary>The attention output projection, the gated residual and the feed-forward half under the uniform
    /// <paramref name="modulation"/>; disposes <paramref name="attn"/>.</summary>
    private Tensor Finish(IBackend backend, Tensor x, Tensor attn, MossImageModulation modulation, DType act)
    {
        int seq = (int)x.Shape[1];
        TensorShape flat = new(1, seq, _hidden);
        Tensor projected = OutProject(backend, attn, seq);
        Tensor x1 = new(flat, DType.F32);
        backend.GatedResidualLastDim(x1, x, projected, modulation.GateAttn);
        projected.Dispose();

        Tensor y2 = new(flat, act);
        backend.RmsNormModulate(y2, x1, _norm2!, modulation.ScaleFfn, modulation.ShiftFfn, _eps);
        Tensor down = Ffn(backend, y2, seq, act);
        Tensor result = new(flat, DType.F32);
        backend.GatedResidualLastDim(result, x1, down, modulation.GateFfn);
        x1.Dispose();
        down.Dispose();
        return result;
    }

    /// <summary>The F32 output projection of <paramref name="attn"/>, which it disposes.</summary>
    private Tensor OutProject(IBackend backend, Tensor attn, int seq)
    {
        Tensor projected = new(new TensorShape(1, seq, _hidden), DType.F32);
        backend.Linear(projected, attn, _o!, null);
        attn.Dispose();
        return projected;
    }

    /// <summary>The F32 SiTU-GLU feed-forward of <paramref name="y"/>, which it disposes.</summary>
    private Tensor Ffn(IBackend backend, Tensor y, int seq, DType act)
    {
        Tensor pair = new(new TensorShape(1, seq, 2 * _ffnHidden), act);
        backend.Linear(pair, y, _ffnUpGate!, null);
        y.Dispose();
        Tensor activated = new(new TensorShape(1, seq, _ffnHidden), act);
        backend.SituGlu(activated, pair);
        pair.Dispose();
        Tensor down = new(new TensorShape(1, seq, _hidden), DType.F32);
        backend.Linear(down, activated, _ffnDown!, null);
        activated.Dispose();
        return down;
    }

    /// <summary>F32 <c>rms(x)·weight</c> through the segment tables, cast to <paramref name="act"/>.</summary>
    private Tensor NormSegmented(IBackend backend, Tensor x, Tensor weight, SegmentModulation modulation, DType act)
    {
        Tensor normed = new(x.Shape, DType.F32);
        backend.RmsNorm(normed, x, weight, _eps);
        Tensor modulated = modulation.Affine(backend, normed);
        normed.Dispose();
        if (act == DType.F32)
            return modulated;
        Tensor cast = new(modulated.Shape, act);
        backend.CastToF16(cast, modulated);
        modulated.Dispose();
        return cast;
    }
}
