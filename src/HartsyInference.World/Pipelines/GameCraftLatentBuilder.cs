using HartsyInference.Core.Tensors;

namespace HartsyInference.World.Pipelines;

/// <summary>Assembles Hunyuan-GameCraft's 33-channel composite DiT input from the noisy target latent, the
/// reference/history latent, and a per-frame binary mask: channels <c>[0..15]</c> = noisy, <c>[16..31]</c> =
/// history, channel <c>32</c> = mask (broadcast across H×W). Chunk 0 uses reference-frame mode (mask=1 on
/// frame 0 only); later chunks use single-historical-clip mode (mask=1 on the history frames).</summary>
public static unsafe class GameCraftLatentBuilder
{
    /// <summary>Builds <c>[B, 33, T, H, W]</c> from <paramref name="noisy"/> and <paramref name="history"/>
    /// (both <c>[B, 16, T, H, W]</c>) plus <paramref name="maskPerFrame"/> (length T, values 0/1).</summary>
    public static Tensor Build(Tensor noisy, Tensor history, ReadOnlySpan<float> maskPerFrame)
    {
        if (noisy.Shape.Rank != 5 || history.Shape.Rank != 5)
            throw new ArgumentException("noisy/history must be [B,16,T,H,W].");
        int b = (int)noisy.Shape[0], c = (int)noisy.Shape[1], t = (int)noisy.Shape[2], h = (int)noisy.Shape[3], w = (int)noisy.Shape[4];
        if (history.Shape[1] != c || history.Shape[2] != t || history.Shape[3] != h || history.Shape[4] != w)
            throw new ArgumentException("noisy and history shapes must match.");
        if (maskPerFrame.Length != t) throw new ArgumentException($"maskPerFrame must have {t} entries.", nameof(maskPerFrame));

        int outC = 2 * c + 1;
        Tensor outT = new(new TensorShape([(long)b, outC, t, h, w]), DType.F32);
        float* np = (float*)noisy.DataPointer, hp = (float*)history.DataPointer, op = (float*)outT.DataPointer;
        long frame = (long)h * w;
        long perChannel = (long)t * frame;

        for (int bb = 0; bb < b; bb++)
        {
            // Channels 0..c-1: noisy; c..2c-1: history.
            for (int ci = 0; ci < c; ci++)
            {
                long src = ((long)bb * c + ci) * perChannel;
                Buffer.MemoryCopy(np + src, op + ((long)bb * outC + ci) * perChannel, perChannel * 4, perChannel * 4);
                Buffer.MemoryCopy(hp + src, op + ((long)bb * outC + c + ci) * perChannel, perChannel * 4, perChannel * 4);
            }
            // Channel 2c: mask, broadcast per-frame value across H×W.
            long maskBase = ((long)bb * outC + 2 * c) * perChannel;
            for (int ti = 0; ti < t; ti++)
            {
                float mv = maskPerFrame[ti];
                float* dst = op + maskBase + ti * frame;
                for (long pix = 0; pix < frame; pix++) dst[pix] = mv;
            }
        }
        return outT;
    }
}
