using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Utilities;

/// <summary>Host-side latent layout shared by every family on the Flux.2 VAE (Flux.2, FLUX.3, Lens): 2×2 patchify of the
/// 32-channel VAE latent to 128 channels, the VAE's BatchNorm running-stat normalization on those 128 channels, and
/// the <c>[B, C, H, W] ↔ [B, H·W, C]</c> token packing. Every op is an element permutation or a per-element affine.</summary>
public static unsafe class Flux2LatentOps
{
    /// <summary>Packs <c>[B, C, H, W]</c> to <c>[B, H·W, C]</c> (diffusers <c>_pack_latents</c> with no spatial reshuffle).</summary>
    public static Tensor Pack(Tensor latent)
    {
        int batch = (int)latent.Shape[0];
        int channels = (int)latent.Shape[1];
        int h = (int)latent.Shape[2];
        int w = (int)latent.Shape[3];
        int seqLen = h * w;
        Tensor packed = new Tensor(new TensorShape(batch, seqLen, channels), DType.F32);
        float* inPtr = (float*)latent.DataPointer;
        float* outPtr = (float*)packed.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    int seqIdx = y * w + x;
                    int outBase = (b * seqLen + seqIdx) * channels;
                    for (int c = 0; c < channels; c++)
                    {
                        int inIdx = ((b * channels + c) * h + y) * w + x;
                        outPtr[outBase + c] = inPtr[inIdx];
                    }
                }
            }
        }
        return packed;
    }

    /// <summary>Unpacks <c>[B, H·W, C]</c> back to <c>[B, C, H, W]</c>.</summary>
    public static Tensor Unpack(Tensor packed, int h, int w)
    {
        int batch = (int)packed.Shape[0];
        int channels = (int)packed.Shape[2];
        int seqLen = h * w;
        Tensor unpacked = new Tensor(new TensorShape(batch, channels, h, w), DType.F32);
        float* inPtr = (float*)packed.DataPointer;
        float* outPtr = (float*)unpacked.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    int seqIdx = y * w + x;
                    int inBase = (b * seqLen + seqIdx) * channels;
                    for (int c = 0; c < channels; c++)
                    {
                        int outIdx = ((b * channels + c) * h + y) * w + x;
                        outPtr[outIdx] = inPtr[inBase + c];
                    }
                }
            }
        }
        return unpacked;
    }

    /// <summary>2×2 spatial patchify <c>[B, C, H·P, W·P] → [B, C·P², H, W]</c>; output channel <c>c·P² + py·P + px</c>.</summary>
    public static Tensor Patchify(Tensor input, int outChannels, int patchSize)
    {
        int batch = (int)input.Shape[0];
        int inChannels = (int)input.Shape[1];
        int inH = (int)input.Shape[2];
        int inW = (int)input.Shape[3];
        if (outChannels != inChannels * patchSize * patchSize)
        {
            throw new InvalidOperationException(
                $"Patchify: out_channels ({outChannels}) must equal in_channels * patch² ({inChannels} * {patchSize}²).");
        }
        int outH = inH / patchSize;
        int outW = inW / patchSize;
        Tensor output = new Tensor(new TensorShape(batch, outChannels, outH, outW), DType.F32);
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < inChannels; c++)
            {
                for (int y = 0; y < outH; y++)
                {
                    for (int x = 0; x < outW; x++)
                    {
                        for (int py = 0; py < patchSize; py++)
                        {
                            for (int px = 0; px < patchSize; px++)
                            {
                                int outChannel = c * patchSize * patchSize + py * patchSize + px;
                                int outIdx = ((b * outChannels + outChannel) * outH + y) * outW + x;
                                int inIdx = ((b * inChannels + c) * inH + y * patchSize + py) * inW + x * patchSize + px;
                                outPtr[outIdx] = inPtr[inIdx];
                            }
                        }
                    }
                }
            }
        }
        return output;
    }

    /// <summary>Inverse of <see cref="Patchify"/>: <c>[B, C·P², H, W] → [B, C, H·P, W·P]</c> (PixelShuffle per channel group).</summary>
    public static Tensor Unpatchify(Tensor input, int outChannels, int patchSize)
    {
        int batch = (int)input.Shape[0];
        int inChannels = (int)input.Shape[1];
        int h = (int)input.Shape[2];
        int w = (int)input.Shape[3];
        if (inChannels != outChannels * patchSize * patchSize)
        {
            throw new InvalidOperationException(
                $"Unpatchify: in_channels ({inChannels}) must equal out_channels * patch² ({outChannels} * {patchSize}²).");
        }
        int outH = h * patchSize;
        int outW = w * patchSize;
        Tensor output = new Tensor(new TensorShape(batch, outChannels, outH, outW), DType.F32);
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < outChannels; c++)
            {
                for (int y = 0; y < h; y++)
                {
                    for (int x = 0; x < w; x++)
                    {
                        for (int py = 0; py < patchSize; py++)
                        {
                            for (int px = 0; px < patchSize; px++)
                            {
                                int inChannel = c * patchSize * patchSize + py * patchSize + px;
                                int inIdx = ((b * inChannels + inChannel) * h + y) * w + x;
                                int outIdx = ((b * outChannels + c) * outH + y * patchSize + py) * outW + x * patchSize + px;
                                outPtr[outIdx] = inPtr[inIdx];
                            }
                        }
                    }
                }
            }
        }
        return output;
    }

    /// <summary>BatchNorm normalize per channel: <c>(z − mean) / √(var + eps)</c>; <paramref name="mean"/> and
    /// <paramref name="var"/> are F32 <c>[C]</c>.</summary>
    public static Tensor BnNormalize(Tensor latent, Tensor mean, Tensor var, float eps)
    {
        int batch = (int)latent.Shape[0];
        int channels = (int)latent.Shape[1];
        int spatial = (int)latent.Shape[2] * (int)latent.Shape[3];
        Tensor output = new Tensor(latent.Shape, DType.F32);
        float* inPtr = (float*)latent.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        float* meanPtr = (float*)mean.DataPointer;
        float* varPtr = (float*)var.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                float invStd = 1.0f / MathF.Sqrt(varPtr[c] + eps);
                float m = meanPtr[c];
                int chanBase = (b * channels + c) * spatial;
                for (int s = 0; s < spatial; s++)
                    outPtr[chanBase + s] = (inPtr[chanBase + s] - m) * invStd;
            }
        }
        return output;
    }

    /// <summary>BatchNorm un-normalize per channel: <c>z·√(var + eps) + mean</c>.</summary>
    public static Tensor BnUnNormalize(Tensor latent, Tensor mean, Tensor var, float eps)
    {
        int batch = (int)latent.Shape[0];
        int channels = (int)latent.Shape[1];
        int spatial = (int)latent.Shape[2] * (int)latent.Shape[3];
        Tensor output = new Tensor(latent.Shape, DType.F32);
        float* inPtr = (float*)latent.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        float* meanPtr = (float*)mean.DataPointer;
        float* varPtr = (float*)var.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                float std = MathF.Sqrt(varPtr[c] + eps);
                float m = meanPtr[c];
                int chanBase = (b * channels + c) * spatial;
                for (int s = 0; s < spatial; s++)
                    outPtr[chanBase + s] = inPtr[chanBase + s] * std + m;
            }
        }
        return output;
    }

    /// <summary>VAE latent <c>[B, C, H·P, W·P]</c> → BN-normalized packed tokens <c>[B, H·W, C·P²]</c> (patchify, normalize, pack).</summary>
    public static Tensor ToPackedTokens(Tensor vaeLatent, Tensor mean, Tensor var, float eps, int patchSize)
    {
        int tokenChannels = (int)vaeLatent.Shape[1] * patchSize * patchSize;
        using Tensor patched = Patchify(vaeLatent, tokenChannels, patchSize);
        using Tensor normalized = BnNormalize(patched, mean, var, eps);
        return Pack(normalized);
    }

    /// <summary>Inverse of <see cref="ToPackedTokens"/>: packed tokens on an <c>h × w</c> grid → VAE latent <c>[B, C, h·P, w·P]</c>.</summary>
    public static Tensor FromPackedTokens(Tensor packed, int h, int w, Tensor mean, Tensor var, float eps, int vaeChannels, int patchSize)
    {
        using Tensor unpacked = Unpack(packed, h, w);
        using Tensor unnormalized = BnUnNormalize(unpacked, mean, var, eps);
        return Unpatchify(unnormalized, vaeChannels, patchSize);
    }
}
