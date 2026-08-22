using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Trellis;

/// <summary>FlowEuler guidance-interval sampler for the TRELLIS sparse-structure stage: rectified-flow Euler with interval-gated CFG via <see cref="IBackend.CfgEulerStep"/>.</summary>
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
