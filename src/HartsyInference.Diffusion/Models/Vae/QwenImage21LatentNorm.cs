using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>Per-channel latent statistics for the Qwen-Image 2.1 VAE (64 channels). The VAE itself is the Wan 2.2
/// architecture, but its latent distribution is not Wan's: these come from ComfyUI's
/// <c>latent_formats.QwenImage21</c>, whose <c>process_in</c>/<c>process_out</c> are the whitening the DiT was
/// trained against. Applying <see cref="Wan22VaeLatentNorm"/>'s embedded 48-channel table here would both
/// mis-shape and mis-scale.</summary>
public static class QwenImage21LatentNorm
{
    /// <summary>Latent channel count.</summary>
    public const int Channels = 64;

    /// <summary>Per-channel mean (64).</summary>
    public static readonly float[] Mean =
    [
        0.5126f, 0.7721f, -0.0631f, 1.3506f, -0.7855f, -2.1025f, -0.3458f, 1.3722f,
        1.8873f, -1.7177f, -0.6510f, 0.2732f, 0.7562f, -0.6163f, -1.0277f, 3.8363f,
        2.0210f, 0.0472f, 0.9320f, 2.0087f, 2.4954f, -0.1391f, -1.4249f, 1.8464f,
        -0.5236f, 1.2826f, 3.7046f, -1.3035f, 2.7286f, -1.4518f, -1.9036f, -1.9955f,
        -0.0342f, -1.0265f, -0.7636f, 3.0555f, 0.0746f, -3.0751f, -0.1076f, 1.7376f,
        -1.0914f, -1.9435f, -0.2784f, -1.3680f, 0.4809f, -0.4433f, 0.3764f, 0.5729f,
        -2.0595f, 1.0960f, -1.3260f, -2.0211f, -5.0179f, 0.5275f, 4.0162f, 1.8505f,
        0.3026f, 1.9373f, 1.4937f, 0.2632f, 0.5547f, -1.7121f, -0.1562f, 0.0304f,
    ];

    /// <summary>Per-channel standard deviation (64).</summary>
    public static readonly float[] Std =
    [
        3.2001f, 3.2936f, 3.4321f, 3.0091f, 3.1061f, 4.0379f, 4.0705f, 3.7910f,
        3.0785f, 3.6500f, 3.9308f, 3.0904f, 2.8778f, 3.7675f, 3.7320f, 5.0756f,
        3.2864f, 4.0397f, 3.1317f, 4.0443f, 2.9249f, 3.9454f, 3.0988f, 4.2489f,
        3.4896f, 3.8513f, 3.9323f, 3.4719f, 3.7498f, 4.2830f, 3.5694f, 4.2467f,
        3.9037f, 3.2947f, 5.0770f, 3.5075f, 3.2700f, 3.4767f, 2.8063f, 5.1125f,
        3.5327f, 4.7833f, 3.1286f, 4.1819f, 3.8527f, 3.8312f, 3.5605f, 4.3875f,
        3.9624f, 4.0168f, 3.5643f, 4.0550f, 5.5614f, 4.2963f, 4.4080f, 3.4959f,
        3.8747f, 3.7608f, 3.5735f, 3.1490f, 3.7662f, 3.6746f, 3.4563f, 3.8161f,
    ];

    /// <summary>In-place decode normalization: <c>z = z · std + mean</c> (ComfyUI <c>process_out</c>).</summary>
    public static void Denormalize(Tensor z) => Wan22VaeLatentNorm.Denormalize(z, Mean, Std);

    /// <summary>In-place encode normalization: <c>z = (z − mean) / std</c> (ComfyUI <c>process_in</c>).</summary>
    public static void Normalize(Tensor z) => Wan22VaeLatentNorm.Normalize(z, Mean, Std);
}
