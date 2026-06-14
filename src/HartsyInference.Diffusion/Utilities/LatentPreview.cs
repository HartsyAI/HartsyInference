using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Utilities;

/// <summary>Fast, model-free latent→RGB preview decoder. Multiplies each
/// latent channel by a small per-architecture <c>[C, 3]</c> factor matrix and
/// adds a per-channel bias to approximate the VAE-decoded image. Output is a
/// blurry, color-shifted preview at the latent's native resolution (typically
/// 1/8 of the final image dimensions) — good enough for "how is this gen
/// trending?" UI feedback without a real VAE pass.
///
/// <para>Factors and biases are the same constants Comfy ships in
/// <c>comfy/latent_formats.py</c> — calibrated empirically against each
/// model family's VAE so previews stay roughly hue-correct.</para>
///
/// <para>This runs in pure managed code on the CPU (a single pass over the
/// latent tensor) and finishes in well under 5 ms even for 1024×1024 latents,
/// so it can run inside the per-step progress callback without measurably
/// slowing diffusion. For higher-fidelity previews use TAESD instead.</para></summary>
public static unsafe class LatentPreview
{
    /// <summary>Returns true if a <c>latent2rgb</c> factor table is available for the given architecture
    /// (or the architecture is pixel-space, where the latent previews as-is).</summary>
    public static bool IsSupported(LatentArchitecture arch) => IsPixelSpace(arch) || GetFactors(arch) is not null;

    /// <summary>True for VAE-free architectures whose "latent" is already an RGB image in [-1, 1] —
    /// previews bypass the factor matrix entirely.</summary>
    public static bool IsPixelSpace(LatentArchitecture arch)
        => arch is LatentArchitecture.ChromaRadiance or LatentArchitecture.ZetaChroma;

    /// <summary>Unpacks a Flux-family packed latent <c>[B, latH/2 * latW/2, C*4]</c> back to
    /// canonical NCHW <c>[B, C, latH, latW]</c> (with C=16 for Flux.1 / Chroma / Z-Image's
    /// internal packed form). Returns a freshly-allocated F32 tensor — the caller owns it and
    /// must dispose. The input is not modified or disposed.
    /// <para>The packing pattern matches diffusers' Flux <c>_pack_latents</c>: each
    /// <c>[2, 2]</c> spatial patch of every channel is interleaved into the channel dim of
    /// one token, in <c>[(y,x)=(0,0), (0,1), (1,0), (1,1)]</c> order.</para></summary>
    public static Tensor UnpackFluxStylePacked(Tensor packed, int latentH, int latentW, int channels = 16)
    {
        if (packed.Shape.Rank != 3)
            throw new ArgumentException($"Expected packed shape [B, S, C*4]; got {packed.Shape}.", nameof(packed));
        int batch = (int)packed.Shape[0];
        int hPacked = latentH / 2;
        int wPacked = latentW / 2;
        int patchDim = channels * 4;
        int seqLen = hPacked * wPacked;
        if ((int)packed.Shape[1] != seqLen || (int)packed.Shape[2] != patchDim)
            throw new ArgumentException(
                $"Packed shape {packed.Shape} doesn't match expected [{batch}, {seqLen}, {patchDim}] " +
                $"for latH={latentH}, latW={latentW}, C={channels}.", nameof(packed));

        Tensor unpacked = new(new TensorShape(batch, channels, latentH, latentW), DType.F32);
        float* inPtr = (float*)packed.DataPointer;
        float* outPtr = (float*)unpacked.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int ph = 0; ph < hPacked; ph++)
            {
                for (int pw = 0; pw < wPacked; pw++)
                {
                    int seqIdx = ph * wPacked + pw;
                    int inBase = (b * seqLen + seqIdx) * patchDim;
                    for (int c = 0; c < channels; c++)
                    {
                        int outChannelBase = (b * channels + c) * latentH * latentW;
                        int patchBase = inBase + c * 4;
                        outPtr[outChannelBase + (ph * 2 + 0) * latentW + (pw * 2 + 0)] = inPtr[patchBase + 0];
                        outPtr[outChannelBase + (ph * 2 + 0) * latentW + (pw * 2 + 1)] = inPtr[patchBase + 1];
                        outPtr[outChannelBase + (ph * 2 + 1) * latentW + (pw * 2 + 0)] = inPtr[patchBase + 2];
                        outPtr[outChannelBase + (ph * 2 + 1) * latentW + (pw * 2 + 1)] = inPtr[patchBase + 3];
                    }
                }
            }
        }
        return unpacked;
    }

    /// <summary>Decodes the first batch element of <paramref name="latent"/> (shape <c>[1, C, H, W]</c>,
    /// or the 3-D video form <c>[1, C, T, H, W]</c> — the middle frame is previewed) into an
    /// HWC-interleaved RGB byte array in <c>[0, 255]</c>. Returns <c>null</c> if the architecture
    /// has no preview factors registered, the latent dtype isn't F32, or the channel count doesn't
    /// match the factor table (e.g., a 16-channel latent passed with <see cref="LatentArchitecture.Sd15"/>).</summary>
    /// <param name="latent">The in-flight diffusion latent. NOT disposed by this method.</param>
    /// <param name="arch">The model family the latent comes from. Used to pick the factor matrix.</param>
    /// <param name="width">Set to the latent's pixel width (= latent W).</param>
    /// <param name="height">Set to the latent's pixel height (= latent H).</param>
    public static byte[]? DecodeLatent2Rgb(Tensor latent, LatentArchitecture arch, out int width, out int height)
    {
        width = 0;
        height = 0;
        if (latent is null || latent.DType != DType.F32) return null;
        if (latent.Shape.Rank != 4 && latent.Shape.Rank != 5) return null;

        // Pixel-space architectures: the latent IS the image — convert [1, 3, H, W] directly.
        if (IsPixelSpace(arch))
        {
            if (latent.Shape.Rank != 4 || (int)latent.Shape[1] != 3) return null;
            int ph = (int)latent.Shape[2];
            int pw = (int)latent.Shape[3];
            width = pw;
            height = ph;
            byte[] direct = new byte[pw * ph * 3];
            float* pp = (float*)latent.DataPointer;
            int pixPlane = ph * pw;
            for (int i = 0; i < pixPlane; i++)
            {
                direct[i * 3 + 0] = ToByte(pp[i]);
                direct[i * 3 + 1] = ToByte(pp[pixPlane + i]);
                direct[i * 3 + 2] = ToByte(pp[2 * pixPlane + i]);
            }
            return direct;
        }

        float[,]? factors = GetFactors(arch);
        if (factors is null) return null;

        bool video = latent.Shape.Rank == 5;
        int c = (int)latent.Shape[1];
        int t = video ? (int)latent.Shape[2] : 1;
        int h = (int)latent.Shape[video ? 3 : 2];
        int w = (int)latent.Shape[video ? 4 : 3];
        if (factors.GetLength(0) != c) return null;

        width = w;
        height = h;
        float[] bias = GetBias(arch);
        int frame = t / 2;

        byte[] rgb = new byte[w * h * 3];
        float* lp = (float*)latent.DataPointer;
        int plane = h * w;
        // NCHW: channel stride = plane. NCTHW: channel stride = t·plane, offset to the middle frame.
        long chanStride = (long)t * plane;
        long frameOff = (long)frame * plane;

        // out[r,g,b] = bias + sum_c (latent[c,(f),y,x] * factors[c, r/g/b]),
        // then clamp to [-1, 1], shift to [0, 1], scale to [0, 255].
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                float r = bias[0], g = bias[1], b = bias[2];
                int pix = y * w + x;
                for (int ch = 0; ch < c; ch++)
                {
                    float v = lp[ch * chanStride + frameOff + pix];
                    r += v * factors[ch, 0];
                    g += v * factors[ch, 1];
                    b += v * factors[ch, 2];
                }
                int outOff = pix * 3;
                rgb[outOff + 0] = ToByte(r);
                rgb[outOff + 1] = ToByte(g);
                rgb[outOff + 2] = ToByte(b);
            }
        }

        return rgb;
    }

    private static byte ToByte(float v)
    {
        // Latent2RGB factor calibration assumes ~[-1, 1] dynamic range. Clamp + shift to [0, 255].
        if (v < -1.0f) v = -1.0f;
        else if (v > 1.0f) v = 1.0f;
        return (byte)((v * 0.5f + 0.5f) * 255.0f + 0.5f);
    }

    private static float[,]? GetFactors(LatentArchitecture arch) => arch switch
    {
        LatentArchitecture.Sd15 => _sd15Factors,
        LatentArchitecture.Sdxl => _sdxlFactors,
        LatentArchitecture.Sd3 => _sd3Factors,
        LatentArchitecture.Flux => _fluxFactors,
        // Flux.2 uses Flux.1 factors as a starting point — replace with arch-specific
        // factors when published. Previews will be approximate but hue-trending correct.
        LatentArchitecture.Flux2 => _fluxFactors,
        LatentArchitecture.Chroma => _fluxFactors,
        LatentArchitecture.ZImage => _fluxFactors,
        // Anima uses the Qwen-Image VAE (16 ch). Flux factors are a reasonable approximation
        // for preview purposes; replace with Qwen-Image-specific factors when published.
        LatentArchitecture.Anima => _fluxFactors,
        LatentArchitecture.FLite => _sd3Factors,
        LatentArchitecture.AuraFlow => _sdxlFactors,
        LatentArchitecture.Wan => _wan22Factors,
        LatentArchitecture.Ltx => _ltxFactors,
        _ => null,
    };

    private static float[] GetBias(LatentArchitecture arch) => arch switch
    {
        LatentArchitecture.Sd3 => _sd3Bias,
        LatentArchitecture.Flux or LatentArchitecture.Flux2
            or LatentArchitecture.Chroma or LatentArchitecture.ZImage
            or LatentArchitecture.Anima => _fluxBias,
        LatentArchitecture.FLite => _sd3Bias,
        LatentArchitecture.Wan => _wan22Bias,
        LatentArchitecture.Ltx => _ltxBias,
        _ => _zeroBias,
    };

    private static readonly float[] _zeroBias = [0f, 0f, 0f];

    // SD 1.5: 4×3, no bias. Source: comfy/latent_formats.py SD15.latent_rgb_factors.
    private static readonly float[,] _sd15Factors = new float[,]
    {
        {  0.298f,  0.207f,  0.208f },
        {  0.187f,  0.286f,  0.173f },
        { -0.158f,  0.189f,  0.264f },
        { -0.184f, -0.271f, -0.473f },
    };

    // SDXL: 4×3, no bias. Source: comfy/latent_formats.py SDXL.latent_rgb_factors.
    private static readonly float[,] _sdxlFactors = new float[,]
    {
        {  0.3651f,  0.4232f,  0.4341f },
        { -0.2533f, -0.0042f,  0.1068f },
        {  0.1076f,  0.1111f, -0.0362f },
        { -0.3165f, -0.2492f, -0.2188f },
    };

    // SD3: 16×3 + bias. Source: comfy/latent_formats.py SD3.latent_rgb_factors / _bias.
    private static readonly float[,] _sd3Factors = new float[,]
    {
        { -0.0645f,  0.0177f,  0.1052f },
        {  0.0028f,  0.0312f,  0.0650f },
        {  0.1848f,  0.0762f,  0.0360f },
        {  0.0944f,  0.0360f,  0.0889f },
        {  0.0897f,  0.0506f, -0.0364f },
        { -0.0020f,  0.1203f,  0.0284f },
        {  0.0855f,  0.0118f,  0.0283f },
        { -0.0539f,  0.0658f,  0.1047f },
        { -0.0057f,  0.0116f,  0.0700f },
        { -0.0412f,  0.0281f, -0.0039f },
        {  0.1106f,  0.1171f,  0.1220f },
        { -0.0248f,  0.0682f, -0.0481f },
        {  0.0815f,  0.0846f,  0.1207f },
        { -0.0120f, -0.0055f, -0.0867f },
        { -0.0749f, -0.0634f, -0.0456f },
        { -0.1418f, -0.1457f, -0.1259f },
    };
    private static readonly float[] _sd3Bias = [-0.0571f, -0.1657f, -0.2512f];

    // Flux.1: 16×3 + bias. Source: comfy/latent_formats.py Flux.latent_rgb_factors / _bias.
    private static readonly float[,] _fluxFactors = new float[,]
    {
        { -0.0346f,  0.0244f,  0.0681f },
        {  0.0034f,  0.0210f,  0.0687f },
        {  0.0275f, -0.0668f, -0.0433f },
        { -0.0174f,  0.0160f,  0.0617f },
        {  0.0859f,  0.0721f,  0.0329f },
        {  0.0004f,  0.0383f,  0.0115f },
        {  0.0405f,  0.0861f,  0.0915f },
        { -0.0236f, -0.0185f, -0.0259f },
        { -0.0245f,  0.0250f,  0.1180f },
        {  0.1008f,  0.0755f, -0.0421f },
        { -0.0515f,  0.0201f,  0.0011f },
        {  0.0428f, -0.0012f, -0.0036f },
        {  0.0817f,  0.0765f,  0.0749f },
        { -0.1264f, -0.0522f, -0.1103f },
        { -0.0280f, -0.0881f, -0.0499f },
        { -0.1262f, -0.0982f, -0.0778f },
    };
    private static readonly float[] _fluxBias = [-0.0329f, -0.0718f, -0.0851f];

    // Wan2.2 video: 48×3 + bias. Source: comfy/latent_formats.py Wan22.latent_rgb_factors / _bias.
    private static readonly float[,] _wan22Factors = new float[,]
    {
        { 0.0119f, 0.0103f, 0.0046f },
        { -0.1062f, -0.0504f, 0.0165f },
        { 0.014f, 0.0409f, 0.0491f },
        { -0.0813f, -0.0677f, 0.0607f },
        { 0.0656f, 0.0851f, 0.0808f },
        { 0.0264f, 0.0463f, 0.0912f },
        { 0.0295f, 0.0326f, 0.059f },
        { -0.0244f, -0.027f, 0.0025f },
        { 0.0443f, -0.0102f, 0.0288f },
        { -0.0465f, -0.009f, -0.0205f },
        { 0.0359f, 0.0236f, 0.0082f },
        { -0.0776f, 0.0854f, 0.1048f },
        { 0.0564f, 0.0264f, 0.0561f },
        { 0.0006f, 0.0594f, 0.0418f },
        { -0.0319f, -0.0542f, -0.0637f },
        { -0.0268f, 0.0024f, 0.026f },
        { 0.0539f, 0.0265f, 0.0358f },
        { -0.0359f, -0.0312f, -0.0287f },
        { -0.0285f, -0.1032f, -0.1237f },
        { 0.1041f, 0.0537f, 0.0622f },
        { -0.0086f, -0.0374f, -0.0051f },
        { 0.039f, 0.067f, 0.2863f },
        { 0.0069f, 0.0144f, 0.0082f },
        { 0.0006f, -0.0167f, 0.0079f },
        { 0.0313f, -0.0574f, -0.0232f },
        { -0.1454f, -0.0902f, -0.0481f },
        { 0.0714f, 0.0827f, 0.0447f },
        { -0.0304f, -0.0574f, -0.0196f },
        { 0.0401f, 0.0384f, 0.0204f },
        { -0.0758f, -0.0297f, -0.0014f },
        { 0.0568f, 0.1307f, 0.1372f },
        { -0.0055f, -0.031f, -0.038f },
        { 0.0239f, -0.0305f, 0.0325f },
        { -0.0663f, -0.0673f, -0.014f },
        { -0.0416f, -0.0047f, -0.0023f },
        { 0.0166f, 0.0112f, -0.0093f },
        { -0.0211f, 0.0011f, 0.0331f },
        { 0.1833f, 0.1466f, 0.225f },
        { -0.0368f, 0.037f, 0.0295f },
        { -0.3441f, -0.3543f, -0.2008f },
        { -0.0479f, -0.0489f, -0.042f },
        { -0.066f, -0.0153f, 0.08f },
        { -0.0101f, 0.0068f, 0.0156f },
        { -0.069f, -0.0452f, -0.0927f },
        { -0.0145f, 0.0041f, 0.0015f },
        { 0.0421f, 0.0451f, 0.0373f },
        { 0.0504f, -0.0483f, -0.0356f },
        { -0.0837f, 0.0168f, 0.0055f },
    };
    private static readonly float[] _wan22Bias = [0.0317f, -0.0878f, -0.1388f];

    // LTX-Video: 128×3 + bias. Source: comfy/latent_formats.py LTXV.latent_rgb_factors / _bias.
    private static readonly float[,] _ltxFactors = new float[,]
    {
        { 0.011202f, -0.00063815f, -0.010021f },
        { 0.086031f, 0.065813f, 0.00095409f },
        { -0.012576f, -0.0075734f, -0.0040528f },
        { 0.0094063f, -0.0021688f, 0.0026093f },
        { 0.0037636f, 0.012765f, 0.0091548f },
        { 0.021024f, -0.0052973f, 0.0034373f },
        { -0.0088896f, -0.019703f, -0.018761f },
        { -0.01316f, -0.010523f, 0.0019709f },
        { -0.0015152f, -0.0069891f, -0.007581f },
        { -0.0017247f, 0.0004656f, -0.0033839f },
        { 0.013617f, 0.0047077f, -0.0020045f },
        { 0.010256f, 0.0077318f, 0.013948f },
        { -0.016108f, -0.0062151f, 0.0011561f },
        { 0.0073407f, 0.015628f, 0.00044865f },
        { 0.00095357f, -0.0029518f, -0.01476f },
        { 0.019143f, 0.010868f, 0.012264f },
        { 0.0044575f, 3.6682e-05f, -0.0068508f },
        { -0.00045681f, 0.003257f, 0.0077929f },
        { 0.033902f, 0.033405f, 0.037454f },
        { -0.023001f, -0.0024877f, -0.0031033f },
        { 0.050265f, 0.038841f, 0.033539f },
        { -0.0041018f, -0.0011095f, 0.0015859f },
        { -0.12689f, -0.13107f, -0.21005f },
        { 0.026276f, 0.014189f, -0.0035963f },
        { -0.0048679f, 0.0088486f, 0.0078029f },
        { -0.001661f, -0.0048597f, -0.005206f },
        { -0.002101f, 0.002361f, 0.0093796f },
        { -0.022482f, -0.021305f, -0.015087f },
        { -0.015753f, -0.010646f, -0.0065083f },
        { -0.0046975f, 0.0050288f, -0.006739f },
        { 0.011951f, 0.020712f, 0.016191f },
        { -0.0063704f, -0.0084827f, -0.0095483f },
        { 0.007261f, -0.0099326f, -0.022978f },
        { -0.00091904f, 0.0062882f, 0.009572f },
        { -0.037178f, -0.037123f, -0.056713f },
        { -0.13373f, -0.1072f, -0.053801f },
        { -0.0053702f, 0.0081256f, 0.0088397f },
        { -0.15247f, -0.21437f, -0.21843f },
        { 0.031441f, 0.0070335f, -0.0097541f },
        { 0.0021528f, -0.0089817f, -0.021023f },
        { 0.0038461f, -0.0058957f, -0.015014f },
        { -0.004347f, -0.01294f, -0.015972f },
        { -0.0054781f, -0.010842f, -0.0030204f },
        { -0.0065347f, 0.0030806f, -0.010163f },
        { -0.0050414f, -0.0071503f, -0.00089686f },
        { -0.0085851f, -0.0024351f, 0.0010674f },
        { -0.0090016f, -0.0096493f, 0.0015692f },
        { 0.0050914f, 0.012099f, 0.019968f },
        { 0.013758f, 0.011669f, 0.0081958f },
        { -0.010518f, -0.011575f, -0.0041307f },
        { -0.02841f, -0.031266f, -0.022149f },
        { 0.0029336f, 0.036511f, 0.018717f },
        { -0.016703f, -0.016696f, -0.0044529f },
        { 0.048818f, 0.040063f, 0.008741f },
        { -0.015066f, -0.00057328f, 0.0029785f },
        { -0.017613f, -0.0081034f, 0.013086f },
        { -0.0092633f, 0.010803f, -0.0063489f },
        { 0.0030851f, 0.0004775f, 0.012347f },
        { -0.022785f, -0.023043f, -0.026005f },
        { -0.024787f, -0.015389f, -0.022104f },
        { -0.023572f, 0.0010544f, 0.012361f },
        { -0.0078915f, -0.0012271f, -0.0060968f },
        { -0.011478f, -0.0012543f, 0.0062679f },
        { -0.054229f, 0.026644f, 0.0063394f },
        { 0.0044216f, -0.0073338f, -0.010464f },
        { -0.0045013f, 0.0016082f, 0.01442f },
        { 0.013673f, 0.0088877f, 0.0041253f },
        { -0.010145f, 0.0090072f, 0.015695f },
        { -0.0056234f, 0.0011847f, 0.0081261f },
        { -0.0037171f, -0.0053538f, 0.001259f },
        { 0.029476f, 0.021424f, 0.030424f },
        { -0.034925f, -0.02434f, -0.025316f },
        { -0.034127f, -0.022406f, -0.010589f },
        { -0.017342f, -0.013249f, -0.010719f },
        { -0.0021478f, -0.0086051f, -0.0029878f },
        { 0.0012089f, -0.0042391f, -0.0068569f },
        { 0.00090411f, -0.0066886f, -6.7547e-05f },
        { 0.016048f, -0.010057f, -0.028929f },
        { 0.001229f, 0.010163f, 0.018861f },
        { 0.017264f, 0.00027257f, 0.013785f },
        { -0.013482f, -0.0036427f, 0.00067481f },
        { 0.0046782f, -0.0052423f, 0.0024467f },
        { -0.0059113f, -0.0062244f, -0.0018162f },
        { 0.015496f, 0.014582f, 0.0019514f },
        { 0.0074958f, 0.0015886f, -0.0082305f },
        { 0.019086f, 0.001636f, -0.0039674f },
        { -0.0057021f, -0.0027307f, -0.0041066f },
        { 0.001745f, 0.014602f, 0.025794f },
        { -0.00082788f, 0.0022902f, 0.0045161f },
        { 0.011632f, 0.0089193f, -0.0072813f },
        { 0.0075721f, 0.0026784f, 0.011393f },
        { 0.0051939f, 0.0036903f, 0.014049f },
        { -0.018383f, -0.022529f, -0.024477f },
        { 0.00058842f, -0.0057874f, -0.01477f },
        { -0.016125f, -0.0086101f, -0.014533f },
        { 0.02054f, 0.020729f, 0.0064338f },
        { 0.0033587f, -0.011226f, -0.016444f },
        { -0.0014742f, -0.010489f, 0.0017097f },
        { 0.02813f, 0.023546f, 0.032791f },
        { -0.018532f, -0.012842f, -0.0087756f },
        { -0.0080533f, -0.010771f, -0.017536f },
        { -0.0039009f, 0.01615f, 0.033359f },
        { -0.0074554f, -0.014154f, -0.006191f },
        { 0.0034734f, -0.01137f, -0.010581f },
        { 0.011476f, 0.0039281f, 0.0028231f },
        { 0.0071639f, -0.0014741f, -0.0038066f },
        { 0.002225f, -0.0087552f, -0.0095719f },
        { 0.024146f, 0.021696f, 0.028056f },
        { -0.0054365f, -0.024291f, -0.017802f },
        { 0.0074263f, 0.01051f, 0.012705f },
        { 0.0062669f, 0.0062658f, 0.019211f },
        { 0.016378f, 0.0094933f, 0.0066971f },
        { 0.017173f, 0.023601f, 0.023296f },
        { -0.014568f, -0.0098279f, -0.011556f },
        { 0.014431f, 0.01443f, 0.0066362f },
        { -0.006823f, 0.018863f, 0.014555f },
        { 0.0061156f, 0.00347f, -0.0026662f },
        { -0.0026983f, -0.0059402f, -0.0092276f },
        { 0.010235f, 0.0074173f, -0.0076243f },
        { -0.013255f, 0.019322f, -0.00092153f },
        { 0.0024222f, -0.0048039f, -0.015759f },
        { 0.026244f, 0.025951f, 0.020249f },
        { 0.015711f, 0.018498f, 0.0027407f },
        { -0.0021714f, 0.0047214f, -0.022443f },
        { -0.0074747f, 0.0074166f, 0.01443f },
        { -0.0083906f, -0.0079776f, 0.0097927f },
        { 0.038321f, 0.0096622f, -0.019268f },
        { -0.014605f, -0.0067032f, 0.0039675f },
    };
    private static readonly float[] _ltxBias = [-0.0571f, -0.1657f, -0.2512f];
}
