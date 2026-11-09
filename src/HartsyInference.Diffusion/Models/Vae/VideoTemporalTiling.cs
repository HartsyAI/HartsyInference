using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>Temporal tiling for video VAE decodes. Each window of latent frames is decoded alone and the decoded frames
/// are cross-faded, so peak memory is one window plus the output.
///
/// <para>Window k starts at latent frame <c>k·S</c> and spans <c>min(n, T − k·S)</c> latent frames. Its decoded frame j
/// lands at output frame <c>C·k·S + j</c>, so a decoder that is local in time reproduces the full decode. Each overlap is
/// a linear ramp of <c>B = C·blend</c> frames: the incoming window weighs <c>min(j, B)/B</c> and the outgoing window the
/// complement, so the weights sum to 1 at every output frame. A short final window is decoded at its own length and
/// keeps the full overlap with its predecessor.</para>
///
/// <para>Latents are not normalized here. Pass them in the form the decode delegate expects; for LTX-2 that is the raw
/// latent, because <c>LtxVideo2VaeDecoder.Decode</c> un-normalizes internally.</para></summary>
public static unsafe class VideoTemporalTiling
{
    /// <summary>Decodes a host F32 latent <c>[B, C, T, H, W]</c> window by window into <c>[B, Cout, C·(T−1)+1, Ho, Wo]</c>.
    /// The delegate borrows each window, which the helper disposes. It returns a new F32 tensor that the helper disposes
    /// after blending; the caller owns the result.</summary>
    public static Tensor Decode(Tensor latent, VideoTemporalTilingOptions options, Func<Tensor, Tensor> decodeWindow)
    {
        ArgumentNullException.ThrowIfNull(latent);
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(decodeWindow);
        ValidateLatent(latent);
        ValidateOptions(options);

        int batch = (int)latent.Shape[0];
        int frames = (int)latent.Shape[2];
        int tile = options.TileLatentFrames;
        int stride = options.StrideLatentFrames;
        int compression = options.CompressionRatio;
        int outFrames = compression * (frames - 1) + 1;

        if (frames <= tile)
        {
            // One window covers the whole latent: pass it through untouched, with no copy and no blend.
            Tensor whole = decodeWindow(latent);
            bool accepted = false;
            try
            {
                RequireDecoded(whole, latent, batch, outFrames);
                accepted = true;
                return whole;
            }
            finally
            {
                if (!accepted && !ReferenceEquals(whole, latent))
                {
                    whole.Dispose();
                }
            }
        }

        int ramp = compression * options.BlendLatentFrames;
        int nextOffset = compression * stride;
        int windowCount = (frames - tile + stride - 1) / stride + 1;
        Tensor? output = null;
        try
        {
            for (int k = 0; k < windowCount; k++)
            {
                int start = k * stride;
                int length = Math.Min(tile, frames - start);
                using Tensor window = ExtractWindow(latent, start, length);
                using Tensor decoded = decodeWindow(window);
                RequireDecoded(decoded, window, batch, compression * (length - 1) + 1);
                output ??= AllocateOutput(decoded, batch, outFrames);
                RequireSameLayout(output, decoded);
                Accumulate(output, decoded, compression * start, ramp, k == 0, k == windowCount - 1, nextOffset);
            }

            Tensor result = output!;
            output = null;
            return result;
        }
        finally
        {
            output?.Dispose();
        }
    }

    /// <summary>Copies latent frames <c>[start, start + length)</c> of every (batch, channel) into a new contiguous window.</summary>
    private static Tensor ExtractWindow(Tensor latent, int start, int length)
    {
        long groups = latent.Shape[0] * latent.Shape[1];
        long frames = latent.Shape[2];
        long plane = latent.Shape[3] * latent.Shape[4];
        Tensor window = new Tensor(new TensorShape([latent.Shape[0], latent.Shape[1], length, latent.Shape[3], latent.Shape[4]]), DType.F32);

        float* src = (float*)latent.DataPointer;
        float* dst = (float*)window.DataPointer;
        long bytes = length * plane * sizeof(float);
        for (long g = 0; g < groups; g++)
        {
            Buffer.MemoryCopy(src + (g * frames + start) * plane, dst + g * length * plane, bytes, bytes);
        }
        return window;
    }

    private static Tensor AllocateOutput(Tensor decoded, int batch, int outFrames)
    {
        return new Tensor(new TensorShape([batch, decoded.Shape[1], outFrames, decoded.Shape[3], decoded.Shape[4]]), DType.F32);
    }

    /// <summary>Adds one window's decoded frames into the output at their global frame indices, weighted by the blend.
    /// The output buffer is zeroed on first access, so frames start at zero and only the weighted sum accumulates.</summary>
    private static void Accumulate(Tensor output, Tensor decoded, int startFrame, int ramp, bool first, bool last, int nextOffset)
    {
        long groups = decoded.Shape[0] * decoded.Shape[1];
        int frames = (int)decoded.Shape[2];
        int plane = checked((int)(decoded.Shape[3] * decoded.Shape[4]));
        long outFrames = output.Shape[2];

        float* src = (float*)decoded.DataPointer;
        float* dst = (float*)output.DataPointer;
        for (int j = 0; j < frames; j++)
        {
            float weight = FrameWeight(j, ramp, first, last, nextOffset);
            if (weight == 0f)
            {
                continue;
            }

            long outFrame = startFrame + j;
            for (long g = 0; g < groups; g++)
            {
                float* s = src + (g * frames + j) * plane;
                float* d = dst + (g * outFrames + outFrame) * plane;
                for (int p = 0; p < plane; p++)
                {
                    d[p] += weight * s[p];
                }
            }
        }
    }

    /// <summary>Blend weight of window-local frame <paramref name="j"/>: the incoming ramp over this window's first
    /// <paramref name="ramp"/> frames times the outgoing ramp over the next window's first <paramref name="ramp"/> frames.
    /// The two windows in an overlap carry complementary ramps, so the weights sum to one.</summary>
    private static float FrameWeight(int j, int ramp, bool first, bool last, int nextOffset)
    {
        float incoming = first ? 1f : Math.Min(j, ramp) / (float)ramp;
        if (last)
        {
            return incoming;
        }

        float outgoing = 1f - Math.Min(Math.Max(j - nextOffset, 0), ramp) / (float)ramp;
        return incoming * outgoing;
    }

    private static void ValidateLatent(Tensor latent)
    {
        if (latent.Shape.Rank != 5)
        {
            throw new ArgumentException($"Latent must be [B, C, T, H, W]; got rank {latent.Shape.Rank}.", nameof(latent));
        }
        if (latent.DType != DType.F32)
        {
            throw new ArgumentException($"Latent must be F32 host memory; got {latent.DType.Name}.", nameof(latent));
        }
        for (int d = 0; d < 5; d++)
        {
            if (latent.Shape[d] < 1)
            {
                throw new ArgumentException($"Latent dimension {d} must be at least 1.", nameof(latent));
            }
        }
    }

    private static void ValidateOptions(VideoTemporalTilingOptions options)
    {
        if (options.CompressionRatio < 1)
        {
            throw new ArgumentException("Compression ratio must be at least 1.", nameof(options));
        }
        if (options.StrideLatentFrames < 1)
        {
            throw new ArgumentException("Stride must be at least 1 latent frame.", nameof(options));
        }

        // Consecutive windows share this many latent intervals; the blend must fit inside it and inside one stride,
        // so that no frame is ramped by more than two windows.
        int overlap = options.TileLatentFrames - 1 - options.StrideLatentFrames;
        int maxBlend = Math.Min(options.StrideLatentFrames, overlap);
        if (maxBlend < 1)
        {
            throw new ArgumentException(
                $"Tile {options.TileLatentFrames} with stride {options.StrideLatentFrames} leaves no overlap to blend across.", nameof(options));
        }
        if (options.BlendLatentFrames < 1 || options.BlendLatentFrames > maxBlend)
        {
            throw new ArgumentException($"Blend {options.BlendLatentFrames} must be in [1, {maxBlend}].", nameof(options));
        }
    }

    /// <summary>Checks a delegate's result. Throws without disposing it, so the caller releases it.</summary>
    private static void RequireDecoded(Tensor decoded, Tensor source, int batch, int expectedFrames)
    {
        if (ReferenceEquals(decoded, source))
        {
            throw new InvalidOperationException("Decode delegate returned its input; it must return a new tensor.");
        }
        if (decoded.Shape.Rank != 5 || decoded.DType != DType.F32)
        {
            throw new InvalidOperationException("Decode delegate must return a 5-D F32 tensor.");
        }
        if (decoded.Shape[0] != batch || decoded.Shape[2] != expectedFrames)
        {
            throw new InvalidOperationException(
                $"Decode delegate returned batch {decoded.Shape[0]} and {decoded.Shape[2]} frames; expected batch {batch} and {expectedFrames} frames.");
        }
    }

    private static void RequireSameLayout(Tensor output, Tensor decoded)
    {
        if (decoded.Shape[1] != output.Shape[1] || decoded.Shape[3] != output.Shape[3] || decoded.Shape[4] != output.Shape[4])
        {
            throw new InvalidOperationException("Decode delegate changed channels or spatial size between windows.");
        }
    }
}
