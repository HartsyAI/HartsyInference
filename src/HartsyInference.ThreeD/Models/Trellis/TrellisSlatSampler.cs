using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Trellis;

/// <summary>FlowEuler guidance-interval sampler for the TRELLIS stage-2 structured-latent (SLAT) flow (same schedule
/// as the stage-1 sampler, over a <see cref="SparseTensor"/> of active voxels). Defaults: 25 steps, cfg 5.0,
/// interval [0.5,1.0], rescale_t 3.0.</summary>
public sealed unsafe class TrellisSlatSampler
{
    /// <summary>Denoises <paramref name="noise"/> (SparseTensor with random feats at the active coords) to the SLAT.</summary>
    public SparseTensor Sample(IBackend backend, SlatFlowModel flow, SparseTensor noise, Tensor cond, Tensor negCond,
        int steps = 25, float cfg = 5.0f, float rescaleT = 3.0f, float cfgLo = 0.5f, float cfgHi = 1.0f)
    {
        float[] t = new float[steps + 1];
        for (int i = 0; i <= steps; i++) { float lin = 1f - (float)i / steps; t[i] = rescaleT * lin / (1f + (rescaleT - 1f) * lin); }

        int n = noise.Count;
        Tensor xf = new(new TensorShape(1, n, 8), DType.F32); backend.CopyInto(xf, noise.Feats.Reshape(new TensorShape(1, n, 8)));
        SparseTensor x = noise.Replace(xf);
        for (int k = 0; k < steps; k++)
        {
            float tc = t[k], tp = t[k + 1], tModel = tc * 1000f, delta = tp - tc;
            SparseTensor vc = flow.Forward(backend, x, tModel, cond);
            if (tc >= cfgLo && tc <= cfgHi)
            {
                SparseTensor vu = flow.Forward(backend, x, tModel, negCond);
                backend.CfgEulerStep(x.Feats, vc.Feats, vu.Feats, 1f + cfg, delta);
                vu.Feats.Dispose();
            }
            else backend.CfgEulerStep(x.Feats, vc.Feats, vc.Feats, 1f, delta);
            vc.Feats.Dispose();
        }
        return x;
    }

    /// <summary>Denormalizes the sampled SLAT: <c>slat = slat·std + mean</c> (per-channel, <paramref name="mean"/>/
    /// <paramref name="std"/> are <c>[8]</c>).</summary>
    public static void Denormalize(SparseTensor slat, Tensor mean, Tensor std)
    {
        int n = slat.Count, c = slat.Channels;
        float* f = (float*)slat.Feats.DataPointer, m = (float*)mean.DataPointer, s = (float*)std.DataPointer;
        for (int i = 0; i < n; i++) for (int ch = 0; ch < c; ch++) f[(long)i * c + ch] = f[(long)i * c + ch] * s[ch] + m[ch];
    }
}
