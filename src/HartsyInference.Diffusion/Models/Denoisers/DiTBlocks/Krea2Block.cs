using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Krea 2 main transformer block (<c>Krea2TransformerBlock</c>). Zero-centered RMSNorm sandwich around a
/// sigmoid-output-gate GQA attention and a SwiGLU MLP, modulated by a shared timestep vector plus a per-block learned
/// <c>scale_shift_table[6, hidden]</c>:
/// <code>
/// prescale, preshift, pregate, postscale, postshift, postgate = (temb_mod + table).unbind   // each [B, hidden]
/// h = h + pregate  · attn((1 + prescale)·norm1(h) + preshift, rope)
/// h = h + postgate · ff  ((1 + postscale)·norm2(h) + postshift)
/// </code>
/// Gates are raw (not tanh'd). The shared <c>temb_mod</c> (width <c>6·hidden</c>) is computed once by the transformer
/// and broadcast across tokens; each block only owns the additive table.</summary>
public sealed unsafe class Krea2Block
{
    private readonly int _hidden;
    private readonly int _ffnInner;
    private readonly float _eps;

    private readonly Krea2Attention _attn;
    private Tensor? _scaleShiftTable;   // [6, hidden]
    private Tensor? _norm1, _norm2;     // zero-centered RMSNorm scales (+1 folded)
    private Tensor? _ffGate, _ffUp, _ffDown;

    public Krea2Block(int hidden, int ffnInner, int numHeads, int numKvHeads, float eps)
    {
        _hidden = hidden;
        _ffnInner = ffnInner;
        _eps = eps;
        _attn = new Krea2Attention(hidden, numHeads, numKvHeads, eps);
    }

    /// <summary>Loads <c>{prefix}.scale_shift_table</c>, <c>{prefix}.norm1/norm2.weight</c>,
    /// <c>{prefix}.attn.*</c>, and <c>{prefix}.ff.gate/up/down.weight</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
    {
        _scaleShiftTable = F32(w[$"{p}.scale_shift_table"]);
        _norm1 = Krea2Norm.LoadZeroCentered(w[$"{p}.norm1.weight"]);
        _norm2 = Krea2Norm.LoadZeroCentered(w[$"{p}.norm2.weight"]);
        _attn.LoadWeights(w, $"{p}.attn");
        _ffGate = w[$"{p}.ff.gate.weight"];
        _ffUp = w[$"{p}.ff.up.weight"];
        _ffDown = w[$"{p}.ff.down.weight"];
    }

    /// <summary>Enumerates weight tensors for GPU preload.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_scaleShiftTable is not null) yield return _scaleShiftTable;
        if (_norm1 is not null) yield return _norm1;
        if (_norm2 is not null) yield return _norm2;
        foreach (Tensor t in _attn.EnumerateWeights()) yield return t;
        if (_ffGate is not null) yield return _ffGate;
        if (_ffUp is not null) yield return _ffUp;
        if (_ffDown is not null) yield return _ffDown;
    }

    /// <summary>Runs one block. <paramref name="tembMod"/> is the shared <c>[B, 6·hidden]</c> modulation;
    /// <paramref name="rope"/> is precomputed for the joint sequence.</summary>
    public Tensor Forward(IBackend backend, Tensor hidden, Tensor tembMod, FluxRope rope, int batch, int seqLen)
    {
        // 6 modulation vectors [B, hidden] = tembMod.unflatten(6, hidden) + scale_shift_table.
        Tensor[] mod = SplitModulation(tembMod, _scaleShiftTable!, batch);

        TensorShape hShape = new TensorShape(batch, seqLen, _hidden);
        Tensor n1 = new Tensor(hShape, DType.F32);
        backend.RmsNorm(n1, hidden, _norm1!, _eps);
        Tensor preIn = AffineScaleShift(n1, mod[0], mod[1], batch, seqLen); // (1+prescale)·n1 + preshift
        n1.Dispose();

        Tensor attnOut = _attn.Forward(backend, preIn, rope, batch, seqLen);
        preIn.Dispose();
        Tensor h1 = GatedResidual(hidden, attnOut, mod[2], batch, seqLen);  // h + pregate·attn
        attnOut.Dispose();

        Tensor n2 = new Tensor(hShape, DType.F32);
        backend.RmsNorm(n2, h1, _norm2!, _eps);
        Tensor postIn = AffineScaleShift(n2, mod[3], mod[4], batch, seqLen); // (1+postscale)·n2 + postshift
        n2.Dispose();

        Tensor ffOut = SwiGlu(backend, postIn, batch, seqLen);
        postIn.Dispose();
        Tensor outp = GatedResidual(h1, ffOut, mod[5], batch, seqLen);      // h + postgate·ff
        ffOut.Dispose(); h1.Dispose();

        foreach (Tensor m in mod) m.Dispose();
        return outp;
    }

    /// <summary>Splits <c>tembMod [B, 6·hidden] + table [6, hidden]</c> into 6 <c>[B, hidden]</c> modulation vectors.</summary>
    private Tensor[] SplitModulation(Tensor tembMod, Tensor table, int batch)
    {
        Tensor[] mod = new Tensor[6];
        for (int i = 0; i < 6; i++) mod[i] = new Tensor(new TensorShape(batch, _hidden), DType.F32);
        float* tm = (float*)tembMod.DataPointer;
        float* tb = (float*)table.DataPointer;
        for (int b = 0; b < batch; b++)
            for (int i = 0; i < 6; i++)
            {
                float* dst = (float*)mod[i].DataPointer + (long)b * _hidden;
                long tmBase = (long)b * 6 * _hidden + (long)i * _hidden;
                long tbBase = (long)i * _hidden;
                for (int d = 0; d < _hidden; d++)
                    dst[d] = tm[tmBase + d] + tb[tbBase + d];
            }
        return mod;
    }

    private Tensor AffineScaleShift(Tensor input, Tensor scale, Tensor shift, int batch, int seqLen)
    {
        Tensor output = new Tensor(new TensorShape(batch, seqLen, _hidden), DType.F32);
        float* ip = (float*)input.DataPointer, sc = (float*)scale.DataPointer, sh = (float*)shift.DataPointer, op = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            long cb = (long)b * _hidden;
            for (int s = 0; s < seqLen; s++)
            {
                long vb = ((long)b * seqLen + s) * _hidden;
                for (int d = 0; d < _hidden; d++)
                    op[vb + d] = (1.0f + sc[cb + d]) * ip[vb + d] + sh[cb + d];
            }
        }
        return output;
    }

    private Tensor GatedResidual(Tensor residual, Tensor value, Tensor gate, int batch, int seqLen)
    {
        Tensor output = new Tensor(new TensorShape(batch, seqLen, _hidden), DType.F32);
        float* rp = (float*)residual.DataPointer, vp = (float*)value.DataPointer, gp = (float*)gate.DataPointer, op = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            long gb = (long)b * _hidden;
            for (int s = 0; s < seqLen; s++)
            {
                long vb = ((long)b * seqLen + s) * _hidden;
                for (int d = 0; d < _hidden; d++)
                    op[vb + d] = rp[vb + d] + gp[gb + d] * vp[vb + d];
            }
        }
        return output;
    }

    private Tensor SwiGlu(IBackend backend, Tensor input, int batch, int seqLen)
    {
        TensorShape ffShape = new TensorShape(batch, seqLen, _ffnInner);
        Tensor g = new Tensor(ffShape, DType.F32);
        Tensor u = new Tensor(ffShape, DType.F32);
        backend.Linear(g, input, _ffGate!, null);
        backend.Linear(u, input, _ffUp!, null);
        Tensor act = new Tensor(ffShape, DType.F32);
        backend.Silu(act, g);
        g.Dispose();
        Tensor gated = new Tensor(ffShape, DType.F32);
        backend.Mul(gated, act, u);
        act.Dispose(); u.Dispose();
        Tensor outp = new Tensor(new TensorShape(batch, seqLen, _hidden), DType.F32);
        backend.Linear(outp, gated, _ffDown!, null);
        gated.Dispose();
        return outp;
    }

    private static Tensor F32(Tensor t) => t.DType == DType.F32 ? t : t.CastTo(DType.F32);
}
