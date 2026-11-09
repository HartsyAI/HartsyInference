using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Gating;
using HartsyInference.Core.Backends.Norm;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>The device-resident building blocks every LTX-2 transformer block is made of, shared by the dual-stream
/// <see cref="LtxVideo2Block"/> and the video-only <see cref="LtxVideo2VideoBlock"/>: AdaLN-Single modulation vectors,
/// the norm-and-modulate pass, the gated residual, the gelu-approximate FFN and the output layer. All take token rows
/// <c>[S, dim]</c> of one sample and follow the stream's activation dtype.</summary>
internal static class LtxVideo2BlockOps
{
    /// <summary>AdaLN params on-device: rows <c>[row, row+n)</c> of <paramref name="table"/> plus <paramref name="temb"/>
    /// (<c>[n·dim]</c>), sliced into n <c>[dim]</c> vectors. Must stay on-device — a host loop here is a D2H drain of
    /// temb + per-vector H2D re-uploads on every block of every step (the Krea2/Chroma host-glue pathology).</summary>
    public static Tensor[] Modulation(IBackend backend, Tensor table, int row, Tensor temb, int n, int dim)
    {
        // Always slice the table rows first: elementwise Add sizes its launch from the FIRST operand, so the operand
        // must be exactly [n, dim] even when row==0 (the av-ca table is [5, dim] but only 4 or 1 rows participate).
        Tensor rows = new(new TensorShape(n, dim), DType.F32);
        backend.SliceRows(rows, table, row);
        Tensor sum = new(new TensorShape(n, dim), DType.F32);
        backend.Add(sum, rows, temb);
        rows.Dispose();
        Tensor[] outs = new Tensor[n];
        for (int m = 0; m < n; m++)
        {
            Tensor o = new(new TensorShape(dim), DType.F32);
            backend.SliceRows(o, sum, m);
            outs[m] = o;
        }
        sum.Dispose();
        return outs;
    }

    /// <summary>Affine-free RMS norm and the AdaLN shift/scale in ONE pass — the pair a block runs before every
    /// modulated stage. The separate ops walked the activation twice.</summary>
    public static Tensor NormModulate(IBackend backend, Tensor x, Tensor scale, Tensor shift, int dim, float eps)
    {
        Tensor o = new(new TensorShape((int)x.Shape[0], dim), x.DType);
        backend.NormModulate(NormModulateCall.Rows(o, x, NormKind.RmsNorm, eps).WithModulation(ModulationSpec.Broadcast(x, scale, shift, ScaleMode.OnePlus)));
        return o;
    }

    /// <summary>Affine-free RMS norm of the rows, in the stream's dtype.</summary>
    public static Tensor RmsNoAffine(IBackend backend, Tensor x, int dim, float eps)
    {
        Tensor o = new(new TensorShape((int)x.Shape[0], dim), x.DType);
        backend.NormModulate(NormModulateCall.Rows(o, x, NormKind.RmsNorm, eps));
        return o;
    }

    /// <summary>Out-of-place <c>x·(1+scale)+shift</c> (used for inputs we must not mutate, e.g. the shared
    /// connector encoder and the shared a2v/v2a norms).</summary>
    public static Tensor ModulateRows(IBackend backend, Tensor x, Tensor scale, Tensor shift, int dim)
    {
        Tensor o = new(new TensorShape((int)x.Shape[0], dim), x.DType);
        backend.NormModulate(NormModulateCall.Rows(o, x, NormKind.None, 0f).WithModulation(ModulationSpec.Broadcast(x, scale, shift, ScaleMode.OnePlus)));
        return o;
    }

    /// <summary><c>residual + gate · value</c> into a fresh tensor; disposes the residual.</summary>
    public static Tensor GatedAddInto(IBackend backend, Tensor residual, Tensor value, Tensor gate)
    {
        // GatedResidualLastDim requires residual/value/output to share a dtype (gate stays F32); the value is an
        // attention or FFN output, which inherited the residual's dtype upstream.
        Tensor o = new(residual.Shape, residual.DType);
        backend.GatedResidualLastDim(o, residual, value, gate);
        residual.Dispose();
        return o;
    }

    /// <summary>The gelu-approximate MLP: up-projection with the activation fused, then the down-projection.</summary>
    public static Tensor Ffn(IBackend backend, Tensor x, Tensor w0, Tensor? b0, Tensor w2, Tensor? b2, int dim)
    {
        int s = (int)x.Shape[0];
        int inner = (int)w0.Shape[0];
        // The 4096->16384 GELU intermediate is the F16 overflow risk site (65504 ceiling). Verified empirically on
        // the real int8-convrot checkpoint rather than assumed: incoherent frames here would mean overflow, and the
        // fix would be to pin `proj`/`act` to F32 while the rest of the block stays F16.
        // gelu-approximate (tanh) folded into the up-projection's dequant epilogue where the backend supports
        // it — this intermediate is [tokens, 4*dim], the widest tensor in the block.
        Tensor act = new(new TensorShape(s, inner), x.DType);
        backend.LinearFused(act, x, w0, b0, LinearFusion.Gelu);
        Tensor o = new(new TensorShape(s, dim), x.DType);
        backend.Linear(o, act, w2, b2);
        act.Dispose();
        return o;
    }

    /// <summary>AdaLN-Single output, fully device-resident: <c>shift/scale = scale_shift_table + embedded</c> ([dim],
    /// broadcast over the sequence; table rows are [shift; scale] — keep that order), LayerNorm-no-affine,
    /// <c>·(1+scale)+shift</c>, then <c>proj_out</c>. Must stay device-resident — a host loop drains the full hidden
    /// state per forward.</summary>
    public static Tensor OutputLayer(IBackend backend, Tensor hidden, Tensor embedded, Tensor scaleShift,
        Tensor projW, Tensor? projB, int s, int dim, int outChannels)
    {
        Tensor ssShift = new Tensor(new TensorShape(dim), DType.F32);
        backend.SliceRows(ssShift, scaleShift, 0);
        Tensor ssScale = new Tensor(new TensorShape(dim), DType.F32);
        backend.SliceRows(ssScale, scaleShift, 1);
        Tensor shift = new Tensor(new TensorShape(dim), DType.F32);
        backend.Add(shift, ssShift, embedded);
        ssShift.Dispose();
        Tensor scale1 = new Tensor(new TensorShape(dim), DType.F32);
        backend.Add(scale1, ssScale, embedded);
        ssScale.Dispose();
        Tensor scale1p = new Tensor(new TensorShape(dim), DType.F32);
        backend.AddScalar(scale1p, scale1, 1f);
        scale1.Dispose();
        // LayerNormNoAffine and AffineBroadcastLastDim both reject a mixed input/output pair, so these follow the
        // stream; the final proj_out writes F32 because the pipeline's GuidedEulerStep is F32-only.
        Tensor normed = new Tensor(new TensorShape(s, dim), hidden.DType);
        // The reference hardcodes norm_out at 1e-6 independent of norm_eps (transformer_ltx2.py:1313).
        backend.LayerNormNoAffine(normed, hidden, 1e-6f);
        Tensor modded = new Tensor(new TensorShape(s, dim), hidden.DType);
        backend.AffineBroadcastLastDim(modded, normed, scale1p, shift);
        normed.Dispose(); scale1p.Dispose(); shift.Dispose();
        Tensor outVel = new Tensor(new TensorShape(s, outChannels), DType.F32);
        backend.Linear(outVel, modded, projW, projB);
        modded.Dispose();
        return outVel;
    }
}
