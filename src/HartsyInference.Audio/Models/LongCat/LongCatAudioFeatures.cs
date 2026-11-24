using HartsyInference.Core.Numerics;

namespace HartsyInference.Audio.Models.LongCat;

/// <summary>Host math between a speech encoder's hidden states and LongCat-Video-Avatar's <c>AudioProjModel</c> input, as
/// <c>pipeline_longcat_video_avatar.py</c> and the demo scripts do it. Every tensor is row-major F32; a feature sequence
/// is <c>[frames, blocks, channels]</c>, a window stack <c>[frames, window, blocks, channels]</c>.</summary>
public static class LongCatAudioFeatures
{
    /// <summary>Whisper encoder frames per second (a 30 s window is 1500 frames).</summary>
    public const int WhisperEncoderFps = 50;

    /// <summary>Hidden states Whisper-large-v3 returns: the stem output plus one per layer, the last one normed.</summary>
    public const int WhisperStateCount = 33;

    /// <summary>Whisper blocks per frame after pooling.</summary>
    public const int WhisperBlockCount = 5;

    /// <summary>Mean of <c>[0,8)</c>, <c>[8,16)</c>, <c>[16,24)</c> and <c>[24,32)</c> of the 33 hidden states, then state
    /// 32 alone (the post-LayerNorm output), per encoder frame. <paramref name="states"/> is
    /// <c>[frames, 33, channels]</c>, <paramref name="pooled"/> <c>[frames, 5, channels]</c>.</summary>
    public static void PoolWhisperStates(ReadOnlySpan<float> states, int frames, int channels, Span<float> pooled)
    {
        if (frames < 0) throw new ArgumentOutOfRangeException(nameof(frames), frames, "Frame count is negative.");
        if (channels < 1) throw new ArgumentOutOfRangeException(nameof(channels), channels, "Needs at least one channel.");
        long expectedIn = (long)frames * WhisperStateCount * channels;
        if (states.Length != expectedIn)
            throw new ArgumentException($"states has {states.Length} values, expected {expectedIn} for [{frames}, {WhisperStateCount}, {channels}].", nameof(states));
        long expectedOut = (long)frames * WhisperBlockCount * channels;
        if (pooled.Length != expectedOut)
            throw new ArgumentException($"pooled has {pooled.Length} values, expected {expectedOut}.", nameof(pooled));

        for (int t = 0; t < frames; t++)
        {
            ReadOnlySpan<float> frame = states.Slice(t * WhisperStateCount * channels, WhisperStateCount * channels);
            Span<float> output = pooled.Slice(t * WhisperBlockCount * channels, WhisperBlockCount * channels);
            for (int block = 0; block < WhisperBlockCount - 1; block++)
            {
                Span<float> target = output.Slice(block * channels, channels);
                for (int c = 0; c < channels; c++)
                {
                    float sum = 0f;
                    for (int s = block * 8; s < block * 8 + 8; s++) sum += frame[s * channels + c];
                    target[c] = sum / 8f;
                }
            }
            frame.Slice(32 * channels, channels).CopyTo(output.Slice((WhisperBlockCount - 1) * channels, channels));
        }
    }

    /// <summary>Avatar 1.0's wav2vec2 features: the transformer-layer states <c>hidden_states[1:]</c>, dropping the encoder's
    /// first (normed-input) state. <paramref name="states"/> is <c>[frames, stateCount, channels]</c>, the result
    /// <c>[frames, stateCount - 1, channels]</c>.</summary>
    public static float[] SkipEmbeddingState(ReadOnlySpan<float> states, int frames, int stateCount, int channels)
    {
        if (frames < 0 || stateCount < 2 || channels < 1) throw new ArgumentException("Needs frames >= 0, at least two states and one channel.");
        if (states.Length != (long)frames * stateCount * channels)
            throw new ArgumentException($"states has {states.Length} values, expected {(long)frames * stateCount * channels}.", nameof(states));
        int kept = (stateCount - 1) * channels;
        float[] layers = new float[(long)frames * kept];
        for (int t = 0; t < frames; t++)
            states.Slice(t * stateCount * channels + channels, kept).CopyTo(layers.AsSpan(t * kept, kept));
        return layers;
    }

    /// <summary>The video frame count of a clip: <c>int(samples / sampleRate * fps)</c> in F64, as the reference
    /// computes it before it truncates the encoder output and resamples.</summary>
    public static int VideoFrames(int sampleCount, int sampleRate, double fps) => LinearResample.FramesAtRate(sampleCount, sampleRate, fps);

    /// <summary>Resamples a <c>[frames, blocks, channels]</c> sequence to <paramref name="destinationFrames"/> frames with
    /// <c>align_corners=True</c> linear interpolation, each block on its own as the reference does (blocks and channels
    /// are independent, so one pass over their product is the same).</summary>
    public static float[] ResampleFrames(ReadOnlySpan<float> features, int frames, int blocks, int channels, int destinationFrames)
    {
        float[] resampled = new float[(long)destinationFrames * blocks * channels];
        LinearResample.AlignCorners(features, frames, blocks * channels, resampled, destinationFrames);
        return resampled;
    }

    /// <summary>Cuts each video frame's window out of <paramref name="features"/>. Window <c>i</c> of the segment is centred
    /// on feature frame <c>startFrame + i * stride</c> and spans <c>Window</c> frames around it, indices clamped to the clip.
    /// Result <c>[count, Window, Blocks, Channels]</c>.</summary>
    public static float[] BuildWindows(ReadOnlySpan<float> features, int frames, LongCatAudioLayout layout, int startFrame, int count)
    {
        ValidateLayout(layout);
        if (frames < 1) throw new ArgumentOutOfRangeException(nameof(frames), frames, "Needs at least one feature frame.");
        if (count < 0) throw new ArgumentOutOfRangeException(nameof(count), count, "Window count is negative.");
        int frameValues = layout.Blocks * layout.Channels;
        if (features.Length != (long)frames * frameValues)
            throw new ArgumentException($"features has {features.Length} values, expected {(long)frames * frameValues}.", nameof(features));

        int half = layout.Window / 2;
        float[] windows = new float[(long)count * layout.WindowValues];
        for (int i = 0; i < count; i++)
        {
            int center = startFrame + i * layout.AudioStride;
            for (int w = 0; w < layout.Window; w++)
            {
                int source = Math.Clamp(center + w - half, 0, frames - 1);
                features.Slice(source * frameValues, frameValues).CopyTo(windows.AsSpan((i * layout.Window + w) * frameValues, frameValues));
            }
        }
        return windows;
    }

    /// <summary>Feature frame a segment's windows start at: <c>segment * stride * (framesPerSegment - overlapFrames)</c>.</summary>
    public static int SegmentStartFrame(int segment, LongCatAudioLayout layout, int framesPerSegment = 93, int overlapFrames = 13)
    {
        if (segment < 0) throw new ArgumentOutOfRangeException(nameof(segment), segment, "Segment index is negative.");
        if (overlapFrames < 0 || overlapFrames >= framesPerSegment)
            throw new ArgumentOutOfRangeException(nameof(overlapFrames), overlapFrames, "Overlap must be below the segment length.");
        return segment * layout.AudioStride * (framesPerSegment - overlapFrames);
    }

    /// <summary>Splits a segment's windows into the two <c>AudioProjModel</c> inputs. The first video frame gives its whole
    /// window (<c>[Window * Blocks * Channels]</c>, <c>proj1</c>'s input). Each following group of <c>VaeScale</c> frames gives
    /// the first frame's windows up to the centre, the middle frames' centre windows and the last frame's windows from the
    /// centre on (<c>[groups, GroupWindow * Blocks * Channels]</c>, <c>proj1_vf</c>'s input).</summary>
    public static (float[] First, float[] Later) GroupForAudioProj(ReadOnlySpan<float> windows, int frameCount, LongCatAudioLayout layout)
    {
        ValidateLayout(layout);
        if (frameCount < 1 || (frameCount - 1) % layout.VaeScale != 0)
            throw new ArgumentException($"A segment has 1 + {layout.VaeScale} * n frames; got {frameCount}.", nameof(frameCount));
        int sliceValues = layout.Blocks * layout.Channels;
        int windowValues = layout.WindowValues;
        if (windows.Length != (long)frameCount * windowValues)
            throw new ArgumentException($"windows has {windows.Length} values, expected {(long)frameCount * windowValues}.", nameof(windows));

        float[] first = windows[..windowValues].ToArray();
        int groups = (frameCount - 1) / layout.VaeScale;
        int half = layout.Window / 2;
        float[] later = new float[(long)groups * layout.GroupValues];
        for (int g = 0; g < groups; g++)
        {
            int baseFrame = 1 + g * layout.VaeScale;
            int slot = 0;
            for (int n = 0; n < layout.VaeScale; n++)
            {
                int from = n == 0 ? 0 : half;
                int to = n == layout.VaeScale - 1 ? layout.Window - 1 : half;
                for (int w = from; w <= to; w++)
                {
                    windows.Slice((baseFrame + n) * windowValues + w * sliceValues, sliceValues)
                        .CopyTo(later.AsSpan((g * layout.GroupWindow + slot) * sliceValues, sliceValues));
                    slot++;
                }
            }
        }
        return (first, later);
    }

    private static void ValidateLayout(LongCatAudioLayout layout)
    {
        ArgumentNullException.ThrowIfNull(layout);
        if (layout.Window < 1 || layout.Window % 2 == 0) throw new ArgumentException($"Window must be odd; got {layout.Window}.", nameof(layout));
        if (layout.VaeScale < 2) throw new ArgumentException($"VaeScale must be at least 2; got {layout.VaeScale}.", nameof(layout));
        if (layout.Blocks < 1 || layout.Channels < 1 || layout.AudioStride < 1 || layout.VideoFps < 1)
            throw new ArgumentException("Blocks, channels, stride and fps must be positive.", nameof(layout));
    }
}
