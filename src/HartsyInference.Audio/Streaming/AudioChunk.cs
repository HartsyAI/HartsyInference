namespace HartsyInference.Audio.Streaming;

/// <summary>One emitted slice of generated audio. Carries float32 PCM samples in
/// <c>[-1, 1]</c>, the sample rate they're at, the channel count, and a sample-rate-relative
/// timestamp marking where in the overall output this chunk starts.
///
/// <para>Used as the element type for all streaming TTS / music output. Streaming STT
/// emits text tokens, not audio chunks — it consumes <see cref="AudioRingBuffer"/> on the
/// input side instead.</para>
///
/// <para>The <see cref="Samples"/> array is owned by the producer; callers must consume
/// or copy it before the next chunk arrives. Producers re-using a backing buffer should
/// allocate a fresh array per chunk.</para></summary>
public readonly record struct AudioChunk(float[] Samples, int SampleRate, int Channels, long StartSampleOffset)
{
    /// <summary>Number of frames in the chunk (= samples per channel).</summary>
    public int FrameCount => Samples.Length / Channels;

    /// <summary>Chunk duration in seconds.</summary>
    public double DurationSeconds => (double)FrameCount / SampleRate;
}
