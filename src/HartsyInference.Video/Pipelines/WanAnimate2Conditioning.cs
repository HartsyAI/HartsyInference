using HartsyInference.Core.Tensors;

namespace HartsyInference.Video.Pipelines;

/// <summary>Builds Wan-Animate-2's generation-stream conditioning: the 4 i2v mask channels and the 16 conditioning
/// latent channels that sit behind the noise in the DiT's 36 inputs, ordered
/// <c>latent(16) | mask(4) | cond(16)</c>. The driving stream's own 36 channels are assembled inside
/// <c>WanAnimate2Transformer</c>, which needs the latent only once and duplicates it there.
///
/// <para>Mask polarity is Wan2.1 i2v's: <b>1 = known/given, 0 = generate</b>. ComfyUI stores the inverse and flips it
/// in <c>concat_cond</c>, landing on the same tensor — do not read its node as a polarity reference.</para></summary>
public static unsafe class WanAnimate2Conditioning
{
    /// <summary>Mask channels in the i2v conditioning block, which is also the VAE's temporal compression.</summary>
    public const int MaskChannels = 4;

    /// <summary>Upstream's <c>get_i2v_mask</c> (<c>wanxiang/eval_i2v.py</c>) as a <c>[4, latentFrames, h, w]</c>
    /// tensor. A pixel-frame indicator of length <c>(latentFrames-1)·4 + 1</c> is set to 1 over the first
    /// <paramref name="maskLen"/> entries, element 0 is repeated 3 extra times, and the result is reshaped
    /// <c>(latentFrames, 4)</c> and transposed. <paramref name="maskLen"/> counts <b>pixel</b> frames, not latent
    /// ones — the whole point of the leading repeat is that pixel frame 0 alone marks latent frame 0 known.</summary>
    public static Tensor GetI2vMask(int latentFrames, int h, int w, int maskLen)
    {
        if (latentFrames < 1) throw new ArgumentOutOfRangeException(nameof(latentFrames));
        if (h < 1 || w < 1) throw new ArgumentOutOfRangeException(nameof(h));
        int pixelFrames = (latentFrames - 1) * MaskChannels + 1;
        if (maskLen < 0 || maskLen > pixelFrames)
            throw new ArgumentOutOfRangeException(nameof(maskLen), $"maskLen must be within [0, {pixelFrames}] pixel frames.");
        Tensor mask = new Tensor(new TensorShape([(long)MaskChannels, latentFrames, h, w]), DType.F32);
        float* p = (float*)mask.DataPointer;
        long frame = (long)h * w;
        long perChannel = latentFrames * frame;
        for (int c = 0; c < MaskChannels; c++)
            for (int t = 0; t < latentFrames; t++)
            {
                // The 3 prepended copies of element 0 shift every later element by 3, so cell (c, t) reads pixel
                // frame t·4 + c − 3, and the whole of latent frame 0 reads pixel frame 0.
                int pixel = t == 0 ? 0 : t * MaskChannels + c - 3;
                float value = pixel < maskLen ? 1f : 0f;
                new Span<float>(p + (long)c * perChannel + t * frame, (int)frame).Fill(value);
            }
        return mask;
    }

    /// <summary>The generation stream's <c>[1, 20, 1 + videoFrames, h, w]</c> conditioning block: mask channels then
    /// conditioning latent, with <paramref name="referenceLatent"/> at latent frame 0 (mask all ones — the character
    /// image is given) and <paramref name="videoLatent"/> behind it. <paramref name="continuationChunk"/> marks
    /// latent frame 1 known as well, which is how a chunk past the first is told its carried frame is real rather
    /// than the mid-grey the rest of the encode is.</summary>
    /// <param name="referenceLatent"><c>[1, z, 1, h, w]</c> — a single-frame VAE encode of the reference image.</param>
    /// <param name="videoLatent"><c>[1, z, videoFrames, h, w]</c> — one causal encode of <c>[carried] ++ [grey]</c>.</param>
    public static Tensor BuildGenerationChannels(Tensor referenceLatent, Tensor videoLatent, bool continuationChunk)
    {
        ArgumentNullException.ThrowIfNull(referenceLatent);
        ArgumentNullException.ThrowIfNull(videoLatent);
        if (referenceLatent.Shape.Rank != 5 || referenceLatent.Shape[2] != 1)
            throw new ArgumentException($"referenceLatent must be [1, z, 1, h, w]; got {referenceLatent.Shape}.", nameof(referenceLatent));
        if (videoLatent.Shape.Rank != 5)
            throw new ArgumentException($"videoLatent must be [1, z, T, h, w]; got {videoLatent.Shape}.", nameof(videoLatent));
        int z = (int)referenceLatent.Shape[1];
        int h = (int)referenceLatent.Shape[3], w = (int)referenceLatent.Shape[4];
        if (videoLatent.Shape[1] != z || videoLatent.Shape[3] != h || videoLatent.Shape[4] != w)
            throw new ArgumentException($"videoLatent {videoLatent.Shape} does not match referenceLatent {referenceLatent.Shape}.", nameof(videoLatent));
        int videoFrames = (int)videoLatent.Shape[2];
        int total = 1 + videoFrames;

        Tensor conditioning = new Tensor(new TensorShape([1L, MaskChannels + z, total, h, w]), DType.F32);
        float* dst = (float*)conditioning.DataPointer;
        long frame = (long)h * w;
        long perChannel = (long)total * frame;

        // The reference slot is its own all-ones single-frame mask; the video slot's mask covers the remaining
        // frames and is built by the same get_i2v_mask the reference implementation calls for both.
        using Tensor videoMask = GetI2vMask(videoFrames, h, w, continuationChunk ? 1 : 0);
        float* vm = (float*)videoMask.DataPointer;
        long videoMaskPerChannel = (long)videoFrames * frame;
        for (int c = 0; c < MaskChannels; c++)
        {
            new Span<float>(dst + (long)c * perChannel, (int)frame).Fill(1f);
            long bytes = videoMaskPerChannel * sizeof(float);
            Buffer.MemoryCopy(vm + (long)c * videoMaskPerChannel, dst + (long)c * perChannel + frame, bytes, bytes);
        }

        float* refSrc = (float*)referenceLatent.DataPointer;
        float* vidSrc = (float*)videoLatent.DataPointer;
        for (int c = 0; c < z; c++)
        {
            long dstBase = (long)(MaskChannels + c) * perChannel;
            Buffer.MemoryCopy(refSrc + (long)c * frame, dst + dstBase, frame * sizeof(float), frame * sizeof(float));
            long bytes = videoMaskPerChannel * sizeof(float);
            Buffer.MemoryCopy(vidSrc + (long)c * videoMaskPerChannel, dst + dstBase + frame, bytes, bytes);
        }
        return conditioning;
    }

    /// <summary>Channel-concatenates the noise and the conditioning block into the DiT's <c>[1, 36, T, h, w]</c>
    /// input. The channel order is <c>x</c> first then <c>y</c>, and <c>y</c> is mask-first.</summary>
    public static Tensor ConcatChannels(Tensor noise, Tensor conditioning)
    {
        ArgumentNullException.ThrowIfNull(noise);
        ArgumentNullException.ThrowIfNull(conditioning);
        int ca = (int)noise.Shape[1], cb = (int)conditioning.Shape[1];
        int t = (int)noise.Shape[2], h = (int)noise.Shape[3], w = (int)noise.Shape[4];
        if (conditioning.Shape[2] != t || conditioning.Shape[3] != h || conditioning.Shape[4] != w)
            throw new ArgumentException($"conditioning {conditioning.Shape} does not match noise {noise.Shape}.", nameof(conditioning));
        long perChannel = (long)t * h * w;
        Tensor o = new Tensor(new TensorShape([1L, ca + cb, t, h, w]), DType.F32);
        float* op = (float*)o.DataPointer;
        long aBytes = ca * perChannel * sizeof(float), bBytes = cb * perChannel * sizeof(float);
        Buffer.MemoryCopy((float*)noise.DataPointer, op, aBytes, aBytes);
        Buffer.MemoryCopy((float*)conditioning.DataPointer, op + ca * perChannel, bBytes, bBytes);
        return o;
    }

    /// <summary>Drops the leading reference latent frame before decode — upstream's <c>x0[0][:, 1:]</c>. Frame 0 is
    /// genuinely denoised (the noise covers it) and then thrown away; it is not a frozen slot.</summary>
    public static Tensor TrimReferenceFrame(Tensor latents)
    {
        ArgumentNullException.ThrowIfNull(latents);
        int c = (int)latents.Shape[1], t = (int)latents.Shape[2];
        int h = (int)latents.Shape[3], w = (int)latents.Shape[4];
        if (t < 2) throw new ArgumentException($"latents must carry the reference frame plus at least one generated frame; got T={t}.", nameof(latents));
        Tensor o = new Tensor(new TensorShape([1L, c, t - 1, h, w]), DType.F32);
        float* src = (float*)latents.DataPointer, dst = (float*)o.DataPointer;
        long frame = (long)h * w;
        long bytes = (long)(t - 1) * frame * sizeof(float);
        for (int ci = 0; ci < c; ci++)
        {
            Buffer.MemoryCopy(src + ((long)ci * t + 1) * frame, dst + (long)ci * (t - 1) * frame, bytes, bytes);
        }
        return o;
    }
}
