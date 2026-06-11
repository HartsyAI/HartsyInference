using SharpInference.Core.Tensors;

namespace SharpInference.Video;

/// <summary>Shared frame extraction for the video pipelines: one frame of a decoded RGB clip tensor
/// <c>[1, C, F, H, W]</c> in [-1, 1] → interleaved RGB24 bytes. Hoisted from the per-pipeline copies
/// (Lance/LTX/Wan/Matrix-Game all decode to the same layout).</summary>
public static unsafe class VideoRgbFrames
{
    /// <summary>Extracts frame <paramref name="frameIndex"/> as interleaved RGB bytes (channels beyond the tensor's C fill 0).</summary>
    public static byte[] ExtractFrame(Tensor rgb, int frameIndex)
    {
        int c = (int)rgb.Shape[1], f = (int)rgb.Shape[2], h = (int)rgb.Shape[3], w = (int)rgb.Shape[4];
        long frame = (long)h * w;
        byte[] outB = new byte[h * w * 3];
        float* p = (float*)rgb.DataPointer;
        for (long pix = 0; pix < frame; pix++)
            for (int ci = 0; ci < 3; ci++)
            {
                float v = ci < c ? p[((long)ci * f + frameIndex) * frame + pix] : 0f;
                int b = (int)MathF.Round((v + 1.0f) * 127.5f);
                outB[pix * 3 + ci] = (byte)Math.Clamp(b, 0, 255);
            }
        return outB;
    }
}
