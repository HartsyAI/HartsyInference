using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Music;

/// <summary>Load-time collapse of PyTorch <c>weight_norm</c>, shared by the audio VAE's encoder and decoder halves —
/// both sides of the MiniMax-H3 codec store their convolutions reparametrized.</summary>
public static unsafe class AudioWeightNorm
{
    /// <summary>Fuses <c>w[i,…] = g[i] · v[i,…] / ‖v[i,…]‖₂</c>, the norm taken over every axis but axis 0 (out-channels
    /// for <c>Conv1d</c>, in-channels for <c>ConvTranspose1d</c> — <c>weight_norm</c> always reparametrizes along axis 0
    /// of the stored parameter). Returns the plain <c>.weight</c> untouched, with <paramref name="allocated"/> false,
    /// when the checkpoint was already fused.</summary>
    public static Tensor Fuse(IReadOnlyDictionary<string, Tensor> weights, string prefix, out bool allocated)
    {
        ArgumentNullException.ThrowIfNull(weights);
        allocated = false;
        if (!weights.TryGetValue($"{prefix}.weight_g", out Tensor? gSrc))
        {
            return weights[$"{prefix}.weight"];
        }
        allocated = true;
        Tensor g = AsF32(gSrc, out bool ownG);
        Tensor v = AsF32(weights[$"{prefix}.weight_v"], out bool ownV);
        int axis0 = (int)v.Shape[0];
        long inner = v.ElementCount / axis0;
        if (g.ElementCount != axis0)
        {
            throw new ArgumentException($"{prefix}.weight_g has {g.ElementCount} elements, expected {axis0}.");
        }

        Tensor fused = new Tensor(v.Shape, DType.F32);
        float* gp = (float*)g.DataPointer;
        float* vp = (float*)v.DataPointer;
        float* fp = (float*)fused.DataPointer;
        for (int i = 0; i < axis0; i++)
        {
            long row = i * inner;
            double sumSq = 0d;
            for (long k = 0; k < inner; k++)
            {
                sumSq += (double)vp[row + k] * vp[row + k];
            }
            float norm = MathF.Sqrt((float)sumSq);
            float scale = norm > 0f ? gp[i] / norm : 0f;
            for (long k = 0; k < inner; k++)
            {
                fp[row + k] = vp[row + k] * scale;
            }
        }
        if (ownG) { g.Dispose(); }
        if (ownV) { v.Dispose(); }
        return fused;
    }

    /// <summary>The tensor as F32, casting only when it is not already.</summary>
    public static Tensor AsF32(Tensor t, out bool owned)
    {
        ArgumentNullException.ThrowIfNull(t);
        owned = t.DType != DType.F32;
        return owned ? t.CastTo(DType.F32) : t;
    }
}
