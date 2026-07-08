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
        // Store the [6, hidden] table flattened to [1, 6·hidden] so the whole modulation split is one device Add
        // (tembMod + table) followed by 6 row slices — see SplitModulation. The host-path indexing (i·hidden) is
        // unchanged. Reshape is a host view (the weight is host-resident until preloaded).
        _scaleShiftTable = F32(w[$"{p}.scale_shift_table"]).Reshape(new TensorShape(1, 6 * _hidden));
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
    // GPU-residency rewrite (mirrors the verified Flux/SD3/Hunyuan ports): the modulation split, the two
    // (1+scale)·norm+shift affines, and the two gated residuals all run as IBackend ops so the activation stays
    // device-resident — no per-op DataPointer D2H sync barriers around the attention/FFN GEMMs. AffineScaleShift →
    // DiTUtils.Modulate (AddScalar(scale,+1) + AffineBroadcastLastDim, same (1+scale) convention); GatedResidual →
    // GatedResidualLastDim; the [6,hidden]-table split → SliceRows + Add (B=1; B>1 keeps the host loop).
    public Tensor Forward(IBackend backend, Tensor hidden, Tensor tembMod, FluxRope rope, int batch, int seqLen)
    {
        // 6 modulation vectors [B, hidden] = tembMod.unflatten(6, hidden) + scale_shift_table. These stay F32 (tiny
        // per-channel vectors); the F16 norm/modulate/gate kernels take an F16 activation + F32 params.
        Tensor[] mod = SplitModulation(backend, tembMod, _scaleShiftTable!, batch);
        DType act = Krea2Dtype.Act;   // F16 hot path (HARTSY_KREA2_F16) else F32; `hidden` arrives in this dtype

        TensorShape hShape = new TensorShape(batch, seqLen, _hidden);
        Tensor n1 = new Tensor(hShape, act);
        backend.RmsNorm(n1, hidden, _norm1!, _eps);
        Tensor preIn = DiTUtils.Modulate(backend, n1, mod[1], mod[0], hShape); // (1+prescale)·n1 + preshift (follows n1 dtype)
        n1.Dispose();

        Tensor attnOut = _attn.Forward(backend, preIn, rope, batch, seqLen);
        preIn.Dispose();
        Tensor h1 = new Tensor(hShape, act);
        backend.GatedResidualLastDim(h1, hidden, attnOut, mod[2]);            // h + pregate·attn
        attnOut.Dispose();

        Tensor n2 = new Tensor(hShape, act);
        backend.RmsNorm(n2, h1, _norm2!, _eps);
        Tensor postIn = DiTUtils.Modulate(backend, n2, mod[4], mod[3], hShape); // (1+postscale)·n2 + postshift
        n2.Dispose();

        Tensor ffOut = SwiGlu(backend, postIn, batch, seqLen);
        postIn.Dispose();
        Tensor outp = new Tensor(hShape, act);
        backend.GatedResidualLastDim(outp, h1, ffOut, mod[5]);               // h + postgate·ff
        ffOut.Dispose(); h1.Dispose();

        foreach (Tensor m in mod) m.Dispose();
        return outp;
    }

    /// <summary>Splits <c>tembMod [B, 6·hidden] + table [6, hidden]</c> into 6 <c>[B, hidden]</c> modulation vectors.
    /// B=1 (the only case pipelines exercise) runs device-resident: each mod[i] is the i-th <c>hidden</c>-wide chunk
    /// of <paramref name="tembMod"/> (SliceRows) plus table row i (SliceRows) added on-device.</summary>
    private Tensor[] SplitModulation(IBackend backend, Tensor tembMod, Tensor table, int batch)
    {
        if (batch != 1)
            return SplitModulationHost(tembMod, table, batch);

        // One device Add over the whole [1, 6·hidden] modulation (tembMod + flattened table), then 6 row slices —
        // 7 ops instead of 18 (6× SliceRows(tembMod) + 6× SliceRows(table) + 6× Add). The 6·hidden-wide Add is a
        // single kernel; per-op host overhead (new Tensor / alloc / launch / dispose) dominates at this op count,
        // so collapsing 18→7 across 28 blocks × 8 steps removes ~2.5k tiny ops per generation.
        Tensor packed = new Tensor(new TensorShape(1, 6 * _hidden), DType.F32);
        backend.Add(packed, tembMod, table);           // table is stored flattened to [1, 6·hidden]
        Tensor[] mod = new Tensor[6];
        TensorShape vecShape = new TensorShape(batch, _hidden);
        for (int i = 0; i < 6; i++)
        {
            mod[i] = new Tensor(vecShape, DType.F32);
            backend.SliceRows(mod[i], packed, i);       // row i = chunk [i·hidden, (i+1)·hidden) of packed
        }
        packed.Dispose();
        return mod;
    }

    /// <summary>Host fallback for the modulation split (B&gt;1 only; pipelines run B=1).</summary>
    private Tensor[] SplitModulationHost(Tensor tembMod, Tensor table, int batch)
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

    private Tensor SwiGlu(IBackend backend, Tensor input, int batch, int seqLen)
    {
        // Follows the block activation dtype. F16 note: the silu(g)·u product is THE F16 overflow risk site
        // (65504 ceiling — why BF16 is the fp8-GEMM default for F32 activations). Krea2's fp8 checkpoint was
        // trained/quantized with e4m3 activations in mind (≤448 per-tensor-scaled), so overflow is not expected;
        // validated empirically per the F16 bring-up (incoherent output here ⇒ overflow ⇒ keep FFN inner F32/BF16).
        DType act = Krea2Dtype.Act;
        TensorShape ffShape = new TensorShape(batch, seqLen, _ffnInner);
        Tensor g = new Tensor(ffShape, act);
        Tensor u = new Tensor(ffShape, act);
        backend.Linear(g, input, _ffGate!, null);
        backend.Linear(u, input, _ffUp!, null);
        Tensor silu = new Tensor(ffShape, act);
        backend.Silu(silu, g);
        g.Dispose();
        Tensor gated = new Tensor(ffShape, act);
        backend.Mul(gated, silu, u);
        silu.Dispose(); u.Dispose();
        Tensor outp = new Tensor(new TensorShape(batch, seqLen, _hidden), act);
        backend.Linear(outp, gated, _ffDown!, null);
        gated.Dispose();
        return outp;
    }

    private static Tensor F32(Tensor t) => t.DType == DType.F32 ? t : t.CastTo(DType.F32);
}
