using HartsyInference.Core.Tensors;

namespace HartsyInference.Video;

/// <summary>Shared frame extraction for the video pipelines: one frame of a decoded RGB clip tensor
/// <c>[1, C, F, H, W]</c> in [-1, 1] → interleaved RGB24 bytes. Hoisted from the per-pipeline copies
/// (Lance/LTX/Wan/Matrix-Game all decode to the same layout).</summary>
public static unsafe class VideoRgbFrames
{
    /// <summary>Every frame of the clip. Identical output to calling <see cref="ExtractFrame"/> per frame — the
    /// frames are independent, and at LTX-2.5's 145×1280×736 the scalar loop is ~410 M iterations of pure host
    /// work sitting in the generation's wall clock.</summary>
    /// <remarks><paramref name="rgb"/> is materialized to host BEFORE the workers start: the first
    /// <c>DataPointer</c> read fires the lazy device sync, which several threads must not race.</remarks>
    public static byte[][] ExtractAllFrames(Tensor rgb)
    {
        _ = rgb.DataPointer;
        byte[][] frames = new byte[(int)rgb.Shape[2]][];
        System.Threading.Tasks.Parallel.For(0, frames.Length, i => frames[i] = ExtractFrame(rgb, i));
        return frames;
    }

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
