using HartsyInference.Core.Numerics;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Pure geometry and resampling helpers shared by H3 guide and continuous AV-mask preparation.</summary>
public static class MiniMaxH3Masking
{
    private const double FrameRescale = 5.0 / 3.0;

    /// <summary>Resolves a signed target-frame anchor after the generation length has been aligned. <c>-1</c> is the
    /// final aligned frame.</summary>
    public static int ResolveFrameIndex(int frameIndex, int alignedFrameCount)
    {
        if (alignedFrameCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(alignedFrameCount), alignedFrameCount,
                "An H3 guide needs a positive aligned target length.");
        }
        int resolved = frameIndex < 0 ? checked(alignedFrameCount + frameIndex) : frameIndex;
        if (resolved < 0 || resolved >= alignedFrameCount)
        {
            throw new ArgumentOutOfRangeException(nameof(frameIndex), frameIndex,
                $"H3 guide frame {frameIndex} resolves to {resolved}; expected [0,{alignedFrameCount}).");
        }
        return resolved;
    }

    /// <summary>Number of visual frames an arbitrary guide clip keeps. Clips shorter than five become a still;
    /// longer clips truncate down to H3's <c>17n+5</c> grid.</summary>
    public static int GuideFrameCount(int decodedFrameCount)
    {
        if (decodedFrameCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(decodedFrameCount), decodedFrameCount,
                "An H3 visual guide video must contain at least one frame.");
        }
        return decodedFrameCount < 5 ? 1 : MiniMaxH3Geometry.SnapFrameCountDown(decodedFrameCount);
    }

    /// <summary>Requires a normalized visual guide clip to fit completely inside the aligned target after its
    /// signed anchor has been resolved. H3 defines no implicit visual crop at the target end.</summary>
    public static void ValidateGuideFrameSpan(int resolvedFrameIndex, int guideFrameCount, int alignedFrameCount)
    {
        if (alignedFrameCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(alignedFrameCount), alignedFrameCount,
                "An H3 guide needs a positive aligned target length.");
        }
        if (resolvedFrameIndex < 0 || resolvedFrameIndex >= alignedFrameCount)
        {
            throw new ArgumentOutOfRangeException(nameof(resolvedFrameIndex), resolvedFrameIndex,
                $"H3 guide anchor must be in [0,{alignedFrameCount}).");
        }
        if (guideFrameCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(guideFrameCount), guideFrameCount,
                "An H3 visual guide must keep at least one frame.");
        }
        if ((long)resolvedFrameIndex + guideFrameCount > alignedFrameCount)
        {
            throw new ArgumentOutOfRangeException(nameof(guideFrameCount), guideFrameCount,
                $"H3 visual guide [{resolvedFrameIndex},{(long)resolvedFrameIndex + guideFrameCount}) exceeds "
                + $"the aligned target range [0,{alignedFrameCount}).");
        }
    }

    /// <summary>Maximum native 40-Hz audio-latent rows per channel remaining after a target-frame anchor.</summary>
    public static int GuideAudioLatentFrames(int targetAudioLatentFrames, int resolvedFrameIndex)
    {
        if (targetAudioLatentFrames <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(targetAudioLatentFrames));
        }
        if (resolvedFrameIndex < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(resolvedFrameIndex));
        }
        return Math.Max(0, (int)Math.Floor(targetAudioLatentFrames - FrameRescale * resolvedFrameIndex));
    }

    /// <summary>Resamples a continuous latent-space <c>[T,H,W]</c> video mask into patch rows with spatial
    /// <c>amax</c>. Values remain continuous; only the 2-D patch reduction changes them. Returns null for an
    /// all-white result so the caller can preserve the exact unmasked execution path.</summary>
    public static float[]? PackVideoMaskRows(ReadOnlySpan<float> latentMask, int latentFrames, int latentHeight,
        int latentWidth, int patchHeight = 2, int patchWidth = 2)
    {
        if (latentFrames <= 0 || latentHeight <= 0 || latentWidth <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(latentFrames),
                $"H3 video mask geometry must be positive; got {latentFrames}x{latentHeight}x{latentWidth}.");
        }
        if (patchHeight <= 0 || patchWidth <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(patchHeight), "H3 video-mask patches must be positive.");
        }
        int expected = checked(latentFrames * latentHeight * latentWidth);
        if (latentMask.Length != expected)
        {
            throw new ArgumentException(
                $"H3 latent video mask has {latentMask.Length} values, expected {expected}.", nameof(latentMask));
        }
        ValidateValues(latentMask, nameof(latentMask));

        int patchRows = MiniMaxH3Geometry.DivideRoundUp(latentHeight, patchHeight);
        int patchColumns = MiniMaxH3Geometry.DivideRoundUp(latentWidth, patchWidth);
        float[] rows = new float[checked(latentFrames * patchRows * patchColumns)];
        bool allWhite = true;
        int output = 0;
        for (int frame = 0; frame < latentFrames; frame++)
        {
            int frameOffset = frame * latentHeight * latentWidth;
            for (int patchY = 0; patchY < patchRows; patchY++)
            {
                for (int patchX = 0; patchX < patchColumns; patchX++)
                {
                    float maximum = float.NegativeInfinity;
                    for (int y = 0; y < patchHeight; y++)
                    {
                        int sourceY = (patchY * patchHeight + y) % latentHeight;
                        for (int x = 0; x < patchWidth; x++)
                        {
                            int sourceX = (patchX * patchWidth + x) % latentWidth;
                            maximum = Math.Max(maximum,
                                latentMask[frameOffset + sourceY * latentWidth + sourceX]);
                        }
                    }
                    rows[output++] = maximum;
                    allWhite &= maximum == 1f;
                }
            }
        }
        return allWhite ? null : rows;
    }

    /// <summary>Linearly resamples mask values from their declared cadence to H3's 40-Hz audio rows, then repeats
    /// the result in channel-major order. Values beyond the supplied duration hold the final sample. Returns null
    /// for all-white output.</summary>
    public static float[]? ResampleAudioMask(IReadOnlyList<float> values, float sourceRate,
        int targetAudioLatentFrames, int channels = 2)
    {
        ArgumentNullException.ThrowIfNull(values);
        if (values.Count == 0)
        {
            throw new ArgumentException("An H3 audio mask must contain at least one value.", nameof(values));
        }
        if (!float.IsFinite(sourceRate) || sourceRate <= 0f)
        {
            throw new ArgumentOutOfRangeException(nameof(sourceRate), sourceRate,
                "An H3 audio-mask cadence must be finite and positive.");
        }
        if (targetAudioLatentFrames <= 0 || channels <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(targetAudioLatentFrames),
                "H3 audio-mask output geometry must be positive.");
        }
        for (int i = 0; i < values.Count; i++)
        {
            ValidateValue(values[i], nameof(values), i);
        }

        float[] mono = new float[targetAudioLatentFrames];
        bool allWhite = true;
        for (int i = 0; i < mono.Length; i++)
        {
            double position = i * (double)sourceRate / MiniMaxH3Geometry.AudioLatentFps;
            int left = Math.Min(values.Count - 1, (int)Math.Floor(position));
            int right = Math.Min(values.Count - 1, left + 1);
            float fraction = (float)Math.Clamp(position - left, 0.0, 1.0);
            mono[i] = values[left] + (values[right] - values[left]) * fraction;
            allWhite &= mono[i] == 1f;
        }
        if (allWhite)
        {
            return null;
        }

        float[] rows = new float[checked(channels * targetAudioLatentFrames)];
        for (int channel = 0; channel < channels; channel++)
        {
            Array.Copy(mono, 0, rows, channel * targetAudioLatentFrames, mono.Length);
        }
        return rows;
    }

    private static void ValidateValues(ReadOnlySpan<float> values, string parameterName)
    {
        for (int i = 0; i < values.Length; i++)
        {
            ValidateValue(values[i], parameterName, i);
        }
    }

    private static void ValidateValue(float value, string parameterName, int index)
    {
        if (!UnitInterval.Contains(value))
        {
            throw new ArgumentOutOfRangeException(parameterName, value,
                $"H3 mask values must be finite and in [0,1]; index {index} was {value}.");
        }
    }
}
