using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Vae;

/// <summary>Wan2.1 VAE per-channel latent normalization constants (16 channels, verbatim from
/// <c>Wan-Video/Wan2.1 wan/modules/vae.py</c>). Encode: <c>z = (μ − mean) / std</c>; decode: <c>μ̃ = z · std + mean</c>.
/// Used by Matrix-Game 2.0 / SkyReels-V2 and any Wan2.1-VAE consumer (distinct from the 48-channel
/// <see cref="Wan22VaeLatentNorm"/>).</summary>
public static unsafe class Wan21VaeLatentNorm
{
    /// <summary>Latent channel count.</summary>
    public const int Channels = 16;

    /// <summary>Per-channel mean (16).</summary>
    public static readonly float[] Mean =
    [
        -0.7571f, -0.7089f, -0.9113f, 0.1075f, -0.1745f, 0.9653f, -0.1517f, 1.5508f,
        0.4134f, -0.0715f, 0.5517f, -0.3632f, -0.1922f, -0.9497f, 0.2503f, -0.2921f,
    ];

    /// <summary>Per-channel standard deviation (16).</summary>
    public static readonly float[] Std =
    [
        2.8184f, 1.4541f, 2.3275f, 2.6558f, 1.2196f, 1.7708f, 2.6052f, 2.0743f,
        3.2687f, 2.1526f, 2.8652f, 1.5579f, 1.6382f, 1.1253f, 2.8251f, 1.9160f,
    ];

    /// <summary>In-place decode normalization: <c>z = z · std + mean</c> per channel, on a <c>[B, 16, T, H, W]</c> latent.</summary>
    public static void Denormalize(Tensor z) => Apply(z, encode: false);

    /// <summary>In-place encode normalization: <c>z = (z − mean) / std</c> per channel.</summary>
    public static void Normalize(Tensor z) => Apply(z, encode: true);

    private static void Apply(Tensor z, bool encode)
    {
        int b = (int)z.Shape[0], c = (int)z.Shape[1];
        if (c != Channels) throw new ArgumentException($"latent channels {c} != {Channels}.", nameof(z));
        long spatial = z.Shape.ElementCount / ((long)b * c);
        float* p = (float*)z.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (int ci = 0; ci < c; ci++)
            {
                float std = Std[ci], mean = Mean[ci];
                long baseOff = ((long)bi * c + ci) * spatial;
                if (encode)
                {
                    float inv = 1f / std;
                    for (long i = 0; i < spatial; i++) p[baseOff + i] = (p[baseOff + i] - mean) * inv;
                }
                else
                {
                    for (long i = 0; i < spatial; i++) p[baseOff + i] = p[baseOff + i] * std + mean;
                }
            }
    }
}
