using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Converts between the DiT's packed row layout and the VAEs' latent tensors. Video rows are patchified
/// <c>[t,h,w] -> C*pt*ph*pw</c>; audio rows are channel-major stereo (<c>ch0 t0..T-1, ch1 t0..T-1</c>) — getting
/// either ordering wrong decodes to plausible-looking garbage rather than an error.</summary>
public static unsafe class MiniMaxH3Latents
{
    /// <summary>Packed video rows <c>[t*h*w, C*pt*ph*pw]</c> to the VAE latent <c>[1, C, T, H, W]</c>.</summary>
    public static Tensor UnpackVideo(Tensor rows, int latentT, int latentH, int latentW, MiniMaxH3Config config)
    {
        ArgumentNullException.ThrowIfNull(rows);
        ArgumentNullException.ThrowIfNull(config);
        int c = config.LatentsDim, pt = config.PatchT, ph = config.PatchH, pw = config.PatchW;
        int th = latentH / ph, tw = latentW / pw;
        Tensor outT = new Tensor(new TensorShape([1L, c, (long)latentT * pt, (long)th * ph, (long)tw * pw]), DType.F32);
        float* src = (float*)rows.DataPointer;
        float* dst = (float*)outT.DataPointer;
        long H = (long)th * ph, W = (long)tw * pw, T = (long)latentT * pt;
        int rowStride = c * pt * ph * pw;
        for (int t = 0; t < latentT; t++)
        {
            for (int y = 0; y < th; y++)
            {
                for (int x = 0; x < tw; x++)
                {
                    float* row = src + ((long)(t * th + y) * tw + x) * rowStride;
                    for (int ci = 0; ci < c; ci++)
                    {
                        for (int rt = 0; rt < pt; rt++)
                        {
                            for (int py = 0; py < ph; py++)
                            {
                                for (int px = 0; px < pw; px++)
                                {
                                    long di = ((((long)ci * T + t * pt + rt) * H) + y * ph + py) * W + x * pw + px;
                                    dst[di] = row[((ci * pt + rt) * ph + py) * pw + px];
                                }
                            }
                        }
                    }
                }
            }
        }
        return outT;
    }

    /// <summary>The VAE latent <c>[1, C, T, H, W]</c> to packed video rows <c>[t*h*w, C*pt*ph*pw]</c> — the inverse of
    /// <see cref="UnpackVideo"/>, used to turn an encoded keyframe or reference clip into conditioning rows. Axes
    /// that do not fill the final patch are circular-padded, matching the target-latent patchifier.</summary>
    public static Tensor PackVideo(Tensor latent, MiniMaxH3Config config)
    {
        ArgumentNullException.ThrowIfNull(latent);
        ArgumentNullException.ThrowIfNull(config);
        int c = config.LatentsDim, pt = config.PatchT, ph = config.PatchH, pw = config.PatchW;
        if (latent.DType != DType.F32 || latent.Shape.Rank != 5 || latent.Shape[0] != 1)
        {
            throw new ArgumentException(
                $"MiniMax-H3 video conditioning must be F32 [1,C,T,H,W]; got {latent.DType} {latent.Shape}.",
                nameof(latent));
        }
        int sourceT = checked((int)latent.Shape[2]);
        int sourceH = checked((int)latent.Shape[3]);
        int sourceW = checked((int)latent.Shape[4]);
        if ((int)latent.Shape[1] != c || sourceT <= 0 || sourceH <= 0 || sourceW <= 0)
        {
            throw new ArgumentException(
                $"MiniMax-H3 latent must have {c} channels and positive T/H/W; got {latent.Shape}.", nameof(latent));
        }
        int latentT = MiniMaxH3Geometry.DivideRoundUp(sourceT, pt);
        int th = MiniMaxH3Geometry.DivideRoundUp(sourceH, ph);
        int tw = MiniMaxH3Geometry.DivideRoundUp(sourceW, pw);
        int rowStride = c * pt * ph * pw;
        Tensor rows = new Tensor(new TensorShape((long)latentT * th * tw, rowStride), DType.F32);
        float* src = (float*)latent.DataPointer;
        float* dst = (float*)rows.DataPointer;
        for (int t = 0; t < latentT; t++)
        {
            for (int y = 0; y < th; y++)
            {
                for (int x = 0; x < tw; x++)
                {
                    float* row = dst + ((long)(t * th + y) * tw + x) * rowStride;
                    for (int ci = 0; ci < c; ci++)
                    {
                        for (int rt = 0; rt < pt; rt++)
                        {
                            for (int py = 0; py < ph; py++)
                            {
                                for (int px = 0; px < pw; px++)
                                {
                                    int sourceFrame = (t * pt + rt) % sourceT;
                                    int sourceY = (y * ph + py) % sourceH;
                                    int sourceX = (x * pw + px) % sourceW;
                                    long si = ((((long)ci * sourceT + sourceFrame) * sourceH) + sourceY)
                                        * sourceW + sourceX;
                                    row[((ci * pt + rt) * ph + py) * pw + px] = src[si];
                                }
                            }
                        }
                    }
                }
            }
        }
        return rows;
    }

    /// <summary>The audio VAE latent <c>[1, C, ch, T]</c> to packed channel-major rows <c>[ch*T, C]</c> — the inverse
    /// of <see cref="UnpackAudio"/>, used to turn encoded reference audio into conditioning rows.</summary>
    public static Tensor PackAudio(Tensor latent, MiniMaxH3Config config)
    {
        ArgumentNullException.ThrowIfNull(latent);
        ArgumentNullException.ThrowIfNull(config);
        int c = config.AudioLatentsDim;
        int channels = (int)latent.Shape[2], audioT = (int)latent.Shape[3];
        if ((int)latent.Shape[1] != c)
        {
            throw new ArgumentException(
                $"MiniMax-H3 audio latent has {latent.Shape[1]} channels, expected {c}.", nameof(latent));
        }
        Tensor rows = new Tensor(new TensorShape((long)channels * audioT, c), DType.F32);
        float* src = (float*)latent.DataPointer;
        float* dst = (float*)rows.DataPointer;
        for (int ch = 0; ch < channels; ch++)
        {
            for (int t = 0; t < audioT; t++)
            {
                float* row = dst + ((long)ch * audioT + t) * c;
                for (int ci = 0; ci < c; ci++)
                {
                    row[ci] = src[(((long)ci * channels) + ch) * audioT + t];
                }
            }
        }
        return rows;
    }

    /// <summary>Packed audio rows <c>[ch*T, C]</c> (channel-major) to the audio VAE latent <c>[1, C, ch, T]</c>.</summary>
    public static Tensor UnpackAudio(Tensor rows, int audioT, MiniMaxH3Config config)
    {
        ArgumentNullException.ThrowIfNull(rows);
        ArgumentNullException.ThrowIfNull(config);
        int c = config.AudioLatentsDim, channels = (int)rows.Shape[0] / Math.Max(1, audioT);
        Tensor outT = new Tensor(new TensorShape(1, c, channels, audioT), DType.F32);
        float* src = (float*)rows.DataPointer;
        float* dst = (float*)outT.DataPointer;
        for (int ch = 0; ch < channels; ch++)
        {
            for (int t = 0; t < audioT; t++)
            {
                float* row = src + ((long)ch * audioT + t) * c;
                for (int ci = 0; ci < c; ci++)
                {
                    dst[(((long)ci * channels) + ch) * audioT + t] = row[ci];
                }
            }
        }
        return outT;
    }
}
