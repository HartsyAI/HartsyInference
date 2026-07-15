using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Trellis;

/// <summary>FlowEuler guidance-interval sampler for the TRELLIS sparse-structure stage. Rectified-flow Euler with
/// interval-CFG: <c>t_seq = rescale·linspace(1,0,steps+1)/(1+(rescale−1)·t)</c>; per step feed <c>t·1000</c> to the
/// model; guide <c>(1+cfg)·v_cond − cfg·v_uncond</c> when <c>t∈interval</c>, else plain <c>v_cond</c>;
/// <c>x_prev = x − (t−t_prev)·v</c> (via <see cref="IBackend.CfgEulerStep"/>). Defaults: 25 steps, cfg 5.0,
/// interval [0.5,1.0], rescale_t 3.0 (ckpt pipeline.json). See <c>docs/Research/TRELLIS_ARCHITECTURE.md</c>.</summary>
public sealed class TrellisSparseStructureSampler
{
    /// <summary>Denoises <paramref name="noise"/> <c>[1,8,16³]</c> to the structure latent <c>z_s</c>.</summary>
    public Tensor Sample(IBackend backend, SparseStructureFlow flow, Tensor noise, Tensor cond, Tensor negCond,
        int steps = 25, float cfg = 5.0f, float rescaleT = 3.0f, float cfgLo = 0.5f, float cfgHi = 1.0f)
    {
        float[] t = new float[steps + 1];
        for (int i = 0; i <= steps; i++)
        {
            float lin = 1f - (float)i / steps;   // linspace(1, 0, steps+1)
            t[i] = rescaleT * lin / (1f + (rescaleT - 1f) * lin);
        }

        Tensor x = new(noise.Shape, DType.F32); backend.CopyInto(x, noise);
        for (int k = 0; k < steps; k++)
        {
            float tc = t[k], tp = t[k + 1], tModel = tc * 1000f, delta = tp - tc;
            Tensor vCond = flow.Forward(backend, x, tModel, cond);
            if (tc >= cfgLo && tc <= cfgHi)
            {
                Tensor vUncond = flow.Forward(backend, x, tModel, negCond);
                backend.CfgEulerStep(x, vCond, vUncond, 1f + cfg, delta);   // v = (1+cfg)·vCond − cfg·vUncond; x += v·delta
                vUncond.Dispose();
            }
            else
            {
                backend.CfgEulerStep(x, vCond, vCond, 1f, delta);   // v = vCond; x += vCond·delta
            }
            vCond.Dispose();
        }
        return x;
    }
}
