using System.Runtime.CompilerServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Utilities;

/// <summary>Latent layout shared by every family on the Flux.2 VAE (Flux.2, FLUX.3, Lens, IrisZ, MOSS, Fibo2): 2×2 patchify of the
/// 32-channel VAE latent to 128 channels, the VAE's BatchNorm running-stat normalization on those 128 channels, and the
/// <c>[B, C, H, W] ↔ [B, H·W, C]</c> token packing. The layouts are one <see cref="IBackend.CopyStrided"/> over a permuted
/// <see cref="StridedView"/>, and the BatchNorm is one <see cref="IBackend.AffineBroadcastLastDim"/> over the tokens, whose last
/// axis is the channel. <see cref="PackHost"/> is the only op that never touches a backend: it is for host tensors.</summary>
public static class Flux2LatentOps
{
    /// <summary>Packs <c>[B, C, H, W]</c> to <c>[B, H·W, C]</c> (diffusers <c>_pack_latents</c> with no spatial reshuffle), on the backend.</summary>
    public static Tensor Pack(IBackend backend, Tensor latent)
    {
        Tensor packed = PackedShell(latent);
        backend.CopyStrided(packed, latent, PackSpec(latent));
        return packed;
    }

    /// <summary>Packs a HOST <c>[B, C, H, W]</c> latent with the host reference copy: the same layout as <see cref="Pack"/>, with no
    /// backend involved. Use it for host noise (fresh per step on the inpaint paths and once per img2img). A device tensor passed
    /// here is downloaded, so it is not a substitute for <see cref="Pack"/> on a device latent.</summary>
    public static Tensor PackHost(Tensor latent)
    {
        Tensor packed = PackedShell(latent);
        StridedCopyReference.Copy(packed, latent, PackSpec(latent));
        return packed;
    }

    /// <summary>Unpacks <c>[B, H·W, C]</c> back to <c>[B, C, H, W]</c>.</summary>
    public static Tensor Unpack(IBackend backend, Tensor packed, int h, int w)
    {
        int batch = (int)packed.Shape[0], channels = (int)packed.Shape[2];
        Tensor unpacked = new Tensor(new TensorShape(batch, channels, h, w), DType.F32);
        backend.CopyStrided(unpacked, packed,
            StridedCopySpec.Gather(StridedView.Of(packed.Shape).Reshape([batch, h, w, channels]).Permute([0, 3, 1, 2])));
        return unpacked;
    }

    /// <summary>2×2 spatial patchify <c>[B, C, H·P, W·P] → [B, C·P², H, W]</c>; output channel <c>c·P² + py·P + px</c>.</summary>
    public static Tensor Patchify(IBackend backend, Tensor input, int outChannels, int patchSize)
    {
        int batch = (int)input.Shape[0], inChannels = (int)input.Shape[1], inH = (int)input.Shape[2], inW = (int)input.Shape[3];
        if (outChannels != inChannels * patchSize * patchSize)
        {
            throw new InvalidOperationException(
                $"Patchify: out_channels ({outChannels}) must equal in_channels * patch² ({inChannels} * {patchSize}²).");
        }
        int outH = inH / patchSize, outW = inW / patchSize;
        Tensor output = new Tensor(new TensorShape(batch, outChannels, outH, outW), DType.F32);
        // View the input as [B, C, H, py, W, px] and gather it as [B, C, py, px, H, W].
        StridedView source = StridedView.Of(input.Shape).Split(2, outH, patchSize).Split(4, outW, patchSize).Permute([0, 1, 3, 5, 2, 4]);
        backend.CopyStrided(output, input, StridedCopySpec.Gather(source));
        return output;
    }

    /// <summary>Inverse of <see cref="Patchify"/>: <c>[B, C·P², H, W] → [B, C, H·P, W·P]</c>.</summary>
    public static Tensor Unpatchify(IBackend backend, Tensor input, int outChannels, int patchSize)
    {
        int batch = (int)input.Shape[0], inChannels = (int)input.Shape[1], h = (int)input.Shape[2], w = (int)input.Shape[3];
        if (inChannels != outChannels * patchSize * patchSize)
        {
            throw new InvalidOperationException(
                $"Unpatchify: in_channels ({inChannels}) must equal out_channels * patch² ({outChannels} * {patchSize}²).");
        }
        Tensor output = new Tensor(new TensorShape(batch, outChannels, h * patchSize, w * patchSize), DType.F32);
        // The input is contiguous [B, C, P, P, H, W]; scatter it into the view [B, C, H, py, W, px] of the output.
        StridedView destination = StridedView.Of(output.Shape).Split(2, h, patchSize).Split(4, w, patchSize).Permute([0, 1, 3, 5, 2, 4]);
        backend.CopyStrided(output, input, StridedCopySpec.Scatter(destination));
        return output;
    }

    /// <summary>VAE latent <c>[B, C, H·P, W·P]</c> → BN-normalized packed tokens <c>[B, H·W, C·P²]</c>: one gather that patchifies
    /// and packs, then the per-channel affine.</summary>
    public static Tensor ToPackedTokens(IBackend backend, Tensor vaeLatent, Tensor mean, Tensor var, float eps, int patchSize)
    {
        Tensor tokens = GatherTokens(backend, vaeLatent, patchSize);
        Tensor normalized = Affine(backend, tokens, mean, var, eps, unnormalize: false);
        tokens.Dispose();
        return normalized;
    }

    /// <summary>Host form of <see cref="ToPackedTokens"/>, for a packed source that feeds a host scheduler loop (AddNoise, once or per
    /// step). It runs on a throwaway CPU backend, so the result is host-resident and nothing is uploaded or read back per step.</summary>
    public static Tensor ToPackedTokensHost(Tensor vaeLatent, Tensor mean, Tensor var, float eps, int patchSize)
    {
        using CpuBackend cpu = new();
        return ToPackedTokens(cpu, vaeLatent, mean, var, eps, patchSize);
    }

    /// <summary>Inverse of <see cref="ToPackedTokens"/>: packed tokens on an <c>h × w</c> grid → VAE latent <c>[B, C, h·P, w·P]</c>.</summary>
    public static Tensor FromPackedTokens(IBackend backend, Tensor packed, int h, int w, Tensor mean, Tensor var, float eps, int vaeChannels, int patchSize)
    {
        Tensor unnormalized = Affine(backend, packed, mean, var, eps, unnormalize: true);
        Tensor vae = ScatterTokens(backend, unnormalized, h, w, vaeChannels, patchSize);
        unnormalized.Dispose();
        return vae;
    }

    /// <summary>BatchNorm normalize of a patchified latent <c>[B, C·P², H, W]</c> returned in the same layout: one gather to tokens, one
    /// per-channel affine, one scatter back. This is <see cref="Patchify"/>, BatchNorm, <see cref="Unpatchify"/> in three device ops.
    /// Numerically the affine is <c>z·(1/√(var+eps)) − mean/√(var+eps)</c>, not <c>(z − mean)/√(var+eps)</c>; the two differ by about one
    /// float32 ulp per element.</summary>
    public static Tensor NormalizePatchified(IBackend backend, Tensor latent, Tensor mean, Tensor var, float eps, int patchSize)
        => PatchifiedBatchNorm(backend, latent, mean, var, eps, patchSize, unnormalize: false);

    /// <summary>Inverse of <see cref="NormalizePatchified"/>: <c>z·√(var+eps) + mean</c> on the patchified layout, in three device ops.</summary>
    public static Tensor UnnormalizePatchified(IBackend backend, Tensor latent, Tensor mean, Tensor var, float eps, int patchSize)
        => PatchifiedBatchNorm(backend, latent, mean, var, eps, patchSize, unnormalize: true);

    private static Tensor PatchifiedBatchNorm(IBackend backend, Tensor latent, Tensor mean, Tensor var, float eps, int patchSize, bool unnormalize)
    {
        int channels = (int)latent.Shape[1], h = (int)latent.Shape[2] / patchSize, w = (int)latent.Shape[3] / patchSize;
        Tensor tokens = GatherTokens(backend, latent, patchSize);
        Tensor affine = Affine(backend, tokens, mean, var, eps, unnormalize);
        tokens.Dispose();
        Tensor output = ScatterTokens(backend, affine, h, w, channels, patchSize);
        affine.Dispose();
        return output;
    }

    /// <summary><c>[B, C, h·P, w·P]</c> → tokens <c>[B, h·w, C·P²]</c> (channel <c>c·P² + py·P + px</c>), one gather, no normalization.</summary>
    private static Tensor GatherTokens(IBackend backend, Tensor latent, int patchSize)
    {
        int batch = (int)latent.Shape[0], channels = (int)latent.Shape[1];
        int h = (int)latent.Shape[2] / patchSize, w = (int)latent.Shape[3] / patchSize;
        Tensor tokens = new Tensor(new TensorShape(batch, h * w, channels * patchSize * patchSize), DType.F32);
        // [B, C, h, py, w, px] gathered as [B, h, w, C, py, px]: row-major [B, h·w, C·P²].
        StridedView source = StridedView.Of(latent.Shape).Split(2, h, patchSize).Split(4, w, patchSize).Permute([0, 2, 4, 1, 3, 5]);
        backend.CopyStrided(tokens, latent, StridedCopySpec.Gather(source));
        return tokens;
    }

    /// <summary>Tokens <c>[B, h·w, C·P²]</c> → <c>[B, C, h·P, w·P]</c>, one scatter, no normalization. The inverse of <see cref="GatherTokens"/>.</summary>
    private static Tensor ScatterTokens(IBackend backend, Tensor tokens, int h, int w, int channels, int patchSize)
    {
        Tensor vae = new Tensor(new TensorShape(tokens.Shape[0], channels, h * patchSize, w * patchSize), DType.F32);
        // The tokens are contiguous [B, h, w, C, P, P]; scatter them into the view [B, C, h, py, w, px] of the VAE latent.
        StridedView destination = StridedView.Of(vae.Shape).Split(2, h, patchSize).Split(4, w, patchSize).Permute([0, 2, 4, 1, 3, 5]);
        backend.CopyStrided(vae, tokens, StridedCopySpec.Scatter(destination));
        return vae;
    }

    private static Tensor Affine(IBackend backend, Tensor tokens, Tensor mean, Tensor var, float eps, bool unnormalize)
    {
        BnAffine affine = AffineFor(mean, var, eps);
        Tensor output = new Tensor(tokens.Shape, DType.F32);
        backend.AffineBroadcastLastDim(output, tokens, unnormalize ? affine.UnScale : affine.NormScale, unnormalize ? affine.UnShift : affine.NormShift);
        return output;
    }

    private static Tensor PackedShell(Tensor latent)
    {
        int batch = (int)latent.Shape[0], channels = (int)latent.Shape[1], h = (int)latent.Shape[2], w = (int)latent.Shape[3];
        return new Tensor(new TensorShape(batch, h * w, channels), DType.F32);
    }

    /// <summary>The gather that turns <c>[B, C, H, W]</c> into <c>[B, H·W, C]</c>: the output is row-major over <c>[B, H, W, C]</c>.</summary>
    private static StridedCopySpec PackSpec(Tensor latent) =>
        StridedCopySpec.Gather(StridedView.Of(latent.Shape).Permute([0, 2, 3, 1]));

    /// <summary>The per-channel affine of one BatchNorm, built once per <c>(mean, var, eps)</c> and kept for the lifetime of the
    /// running-stat tensors. Normalize is <c>scale = 1/√(var+eps)</c>, <c>shift = −mean·scale</c>; un-normalize is <c>scale = √(var+eps)</c>,
    /// <c>shift = mean</c>.</summary>
    private sealed class BnAffine
    {
        public BnAffine(Tensor var, float eps, Tensor normScale, Tensor normShift, Tensor unScale, Tensor unShift)
        {
            Var = var;
            Eps = eps;
            NormScale = normScale;
            NormShift = normShift;
            UnScale = unScale;
            UnShift = unShift;
        }

        public Tensor Var { get; }

        public float Eps { get; }

        public Tensor NormScale { get; }

        public Tensor NormShift { get; }

        public Tensor UnScale { get; }

        public Tensor UnShift { get; }
    }

    private static readonly ConditionalWeakTable<Tensor, BnAffine> Affines = new();

    /// <summary>The cached affine for <paramref name="mean"/>, rebuilt when the variance tensor or <paramref name="eps"/> changes.</summary>
    private static BnAffine AffineFor(Tensor mean, Tensor var, float eps)
    {
        if (Affines.TryGetValue(mean, out BnAffine? cached) && ReferenceEquals(cached.Var, var) && cached.Eps == eps)
        {
            return cached;
        }
        BnAffine built = BuildAffine(mean, var, eps);
        Affines.AddOrUpdate(mean, built);
        return built;
    }

    private static unsafe BnAffine BuildAffine(Tensor mean, Tensor var, float eps)
    {
        int channels = (int)mean.Shape[0];
        Tensor normScale = new Tensor(new TensorShape(channels), DType.F32);
        Tensor normShift = new Tensor(new TensorShape(channels), DType.F32);
        Tensor unScale = new Tensor(new TensorShape(channels), DType.F32);
        Tensor unShift = new Tensor(new TensorShape(channels), DType.F32);
        float* meanPtr = (float*)mean.DataPointer;
        float* varPtr = (float*)var.DataPointer;
        float* normScalePtr = (float*)normScale.DataPointer;
        float* normShiftPtr = (float*)normShift.DataPointer;
        float* unScalePtr = (float*)unScale.DataPointer;
        float* unShiftPtr = (float*)unShift.DataPointer;
        for (int c = 0; c < channels; c++)
        {
            float std = MathF.Sqrt(varPtr[c] + eps);
            float inv = 1.0f / std;
            normScalePtr[c] = inv;
            normShiftPtr[c] = -meanPtr[c] * inv;
            unScalePtr[c] = std;
            unShiftPtr[c] = meanPtr[c];
        }
        return new BnAffine(var, eps, normScale, normShift, unScale, unShift);
    }
}
