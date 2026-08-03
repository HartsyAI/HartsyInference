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
