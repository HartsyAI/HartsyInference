using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>Shared primitives for the SeedVR2 causal video VAE, encoding the two semantics that differ from
/// the HunyuanVideo classes and would fail SILENTLY if inherited: (1) GroupNorm uses PER-FRAME statistics —
/// the reference's <c>causal_norm_wrapper</c> rearranges <c>b c t h w → (b t) c h w</c> before nn.GroupNorm,
/// so stats span (C/g,H,W) of one frame, never the whole clip; (2) spatial conv padding is ZEROS (plain
/// diffusers Conv3d), not replicate — only the temporal head uses frame-0 replication. The asymmetric pad
/// and pixel-shuffle run device-resident on CUDA (IBackend default = host reference on CPU).</summary>
public static class SeedVr2VaeOps
{
    /// <summary>Per-frame GroupNorm + SiLU on <c>[B,C,T,H,W]</c> (reference <c>causal_norm_wrapper</c>).
    /// One tensor identity across norm and SiLU (see cuda-activation-cache-reshape-identity).</summary>
    public static Tensor NormSiluPerFrame(IBackend backend, Tensor x, Tensor weight, Tensor bias, int groups, float eps)
    {
        int b = (int)x.Shape[0], c = (int)x.Shape[1], t = (int)x.Shape[2], h = (int)x.Shape[3], w = (int)x.Shape[4];
        Tensor frames = Vae3dLayout.ToFrames(backend, x);
        Tensor normed = new Tensor(frames.Shape, frames.DType);
        backend.GroupNorm(normed, frames, weight, bias, groups, eps);
        backend.Silu(normed, normed);
        frames.Dispose();
        Tensor output = Vae3dLayout.FromFrames(backend, normed, b, c, t, h, w);
        normed.Dispose();
        return output;
    }

    /// <summary>Per-frame GroupNorm + SiLU when the activation is already frame-major <c>[B·T,C,H,W]</c>.</summary>
    public static Tensor NormSiluFrames(IBackend backend, Tensor frames, Tensor weight, Tensor bias, int groups, float eps)
    {
        Tensor normed = new Tensor(frames.Shape, frames.DType);
        backend.GroupNorm(normed, frames, weight, bias, groups, eps);
        backend.Silu(normed, normed);
        return normed;
    }

    /// <summary>Asymmetric zero pad (right/bottom only) on <c>[B,C,T,H,W]</c> — the diffusers
    /// <c>Downsample2D(padding=0)</c> convention: <c>F.pad(x, (0,1,0,1))</c> before the stride-2 conv.
    /// Device-resident via <see cref="IBackend.SeedVr2PadBottomRight"/> (host reference on CPU).</summary>
    public static Tensor PadBottomRight(IBackend backend, Tensor x)
    {
        int b = (int)x.Shape[0], c = (int)x.Shape[1], t = (int)x.Shape[2], h = (int)x.Shape[3], w = (int)x.Shape[4];
        Tensor padded = new Tensor(new TensorShape([(long)b, c, t, h + 1, w + 1]), x.DType);
        backend.SeedVr2PadBottomRight(padded, x);
        return padded;
    }

    /// <summary>MAGViT-style channel→space shuffle (reference Upsample3D):
    /// <c>b (x y z c) f h w → b c (f·z) (h·x) (w·y)</c> with x,y spatial and z temporal ratios, followed by
    /// <c>remove_head</c> when temporal (drops OUTPUT FRAME INDEX 1 — the duplicated half of frame 0 — giving
    /// T→2T−1; note LTX drops index 0, the conventions differ). Device-resident via
    /// <see cref="IBackend.SeedVr2PixelShuffle"/> (host reference on CPU).</summary>
    public static Tensor PixelShuffle(IBackend backend, Tensor x, int spatialRatio, int temporalRatio)
    {
        int b = (int)x.Shape[0], cIn = (int)x.Shape[1], f = (int)x.Shape[2], h = (int)x.Shape[3], w = (int)x.Shape[4];
        int ratio = spatialRatio * spatialRatio * temporalRatio;
        int c = cIn / ratio;
        if (c * ratio != cIn)
            throw new ArgumentException($"Channels {cIn} not divisible by shuffle ratio {ratio}.");
        int fOut = f * temporalRatio;
        int fFinal = temporalRatio > 1 ? fOut - 1 : fOut;
        Tensor output = new Tensor(new TensorShape([(long)b, c, fFinal, h * spatialRatio, w * spatialRatio]), x.DType);
        backend.SeedVr2PixelShuffle(output, x, spatialRatio, temporalRatio);
        return output;
    }

    /// <summary>Builds a SeedVR2 conv from checkpoint weights: pads derived from the kernel's own shape
    /// (kt=1 → padT 0 — passing padT 1 to a kt=1 causal conv ADDS frames), frame-0-replicate causal head,
    /// zero spatial padding.</summary>
    public static CausalConv3d Conv(IReadOnlyDictionary<string, Tensor> weights, string baseKey,
        int strideT = 1, int strideH = 1, int strideW = 1, bool padSpatial = true, DType? computeDtype = null)
    {
        Tensor weight = weights[$"{baseKey}.weight"];
        weights.TryGetValue($"{baseKey}.bias", out Tensor? bias);
        int kt = (int)weight.Shape[2], kh = (int)weight.Shape[3], kw = (int)weight.Shape[4];
        return new CausalConv3d(weight, bias, strideT, strideH, strideW, padT: (kt - 1) / 2,
            padH: padSpatial ? (kh - 1) / 2 : 0, padW: padSpatial ? (kw - 1) / 2 : 0, replicateFirstPad: true,
            causal: true, computeDtype: computeDtype);
    }
}
