using HartsyInference.Audio.Models.Wake;
using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Endpointing driven by a scripted <see cref="IVadModel"/>, so the hysteresis, the minimum-silence and
/// minimum-speech debounces and the padding are pinned without Silero's weights. The real model's scores are
/// covered by <c>SileroVadParityTests</c>; this is the logic on top of them.</summary>
public sealed class SileroVadStreamScriptedTests
{
    private const int Window = 512;
    private const int SampleRate = 16_000;

    /// <summary>Returns a scripted probability per call; counts resets.</summary>
    private sealed class ScriptedVad(IReadOnlyList<float> script) : IVadModel
    {
        private int _next;

        public int Resets { get; private set; }
        public int Calls => _next;
        public int WindowSamples => Window;

        public float Process(IBackend backend, ReadOnlySpan<float> chunk)
        {
            Assert.Equal(Window, chunk.Length);
            return _next < script.Count ? script[_next++] : script[^1];
        }

        public void Reset() => Resets++;
    }

    private static float[] Script(int speechChunks, int silenceChunks, float speech = 0.9f, float silence = 0.1f)
    {
        float[] script = new float[speechChunks + silenceChunks];
        Array.Fill(script, speech, 0, speechChunks);
        Array.Fill(script, silence, speechChunks, silenceChunks);
        return script;
    }

    [Fact]
    public void SpeechThenSilence_ClosesOneSegmentAfterTheMinimumSilence()
    {
        // 10 chunks of speech (320 ms) then silence. Minimum silence 100 ms = 1600 samples, so the segment closes on
        // the first silent chunk that starts 1600+ samples after silence began: chunk 14 (4 × 512 = 2048 ≥ 1600).
        ScriptedVad vad = new(Script(10, 10));
        SileroVadStream stream = new(vad);
        using CpuBackend backend = new();
        float[] chunk = new float[Window];

        List<(int Index, SileroVadSegment Segment)> closed = [];
        for (int i = 0; i < 20; i++)
        {
            if (stream.Push(backend, chunk, out SileroVadSegment segment))
            {
                closed.Add((i, segment));
            }
            if (i < 10)
            {
                Assert.True(stream.InSpeech);
                Assert.True(stream.LastChunkWasSpeech);
            }
        }

        (int index, SileroVadSegment seg) = Assert.Single(closed);
        Assert.Equal(14, index);
        Assert.Equal(0, seg.StartSample);
        Assert.Equal(10 * Window + 30 * SampleRate / 1000, seg.EndSample);
        Assert.False(stream.InSpeech);
        Assert.Equal(20L * Window, stream.ConsumedSamples);
    }

    [Fact]
    public void SpeechStartingMidStream_IsPaddedBackwards()
    {
        ScriptedVad vad = new([0.1f, 0.1f, 0.1f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f]);
        SileroVadStream stream = new(vad);
        using CpuBackend backend = new();
        float[] chunk = new float[Window];
        SileroVadSegment found = default;
        for (int i = 0; i < 17; i++)
        {
            if (stream.Push(backend, chunk, out SileroVadSegment segment))
            {
                found = segment;
            }
            if (i == 3)
            {
                Assert.Equal(3 * Window - 480, stream.SpeechStartSample);
            }
        }
        Assert.Equal(3 * Window - 480, found.StartSample);
        Assert.Equal(12 * Window + 480, found.EndSample);
    }

    [Fact]
    public void ADipIntoTheHysteresisBand_DoesNotCloseTheSegment()
    {
        // 0.4 sits between exit (0.35) and enter (0.5): neither opens nor closes.
        float[] script = [0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.9f, 0.9f];
        ScriptedVad vad = new(script);
        SileroVadStream stream = new(vad);
        using CpuBackend backend = new();
        float[] chunk = new float[Window];
        for (int i = 0; i < script.Length; i++)
        {
            Assert.False(stream.Push(backend, chunk, out _));
            Assert.True(stream.InSpeech);
        }
        Assert.True(stream.Flush(out SileroVadSegment flushed));
        Assert.Equal(0, flushed.StartSample);
    }

    [Fact]
    public void ABriefReturnAboveEnter_CancelsAPendingClose()
    {
        // Two silent chunks (64 ms) is under the 100 ms minimum; speech resumes, so no segment closes.
        float[] script = [0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.1f, 0.1f, 0.9f, 0.9f, 0.9f, 0.9f];
        ScriptedVad vad = new(script);
        SileroVadStream stream = new(vad);
        using CpuBackend backend = new();
        float[] chunk = new float[Window];
        for (int i = 0; i < script.Length; i++)
        {
            Assert.False(stream.Push(backend, chunk, out _));
        }
        Assert.True(stream.InSpeech);
    }

    [Fact]
    public void SpeechShorterThanTheMinimum_IsDropped()
    {
        // 2 chunks = 64 ms of speech, under the 250 ms minimum.
        ScriptedVad vad = new(Script(2, 10));
        SileroVadStream stream = new(vad);
        using CpuBackend backend = new();
        float[] chunk = new float[Window];
        for (int i = 0; i < 12; i++)
        {
            Assert.False(stream.Push(backend, chunk, out _));
        }
        Assert.False(stream.InSpeech);
        Assert.False(stream.Flush(out _));
    }

    [Fact]
    public void Flush_ClosesAnOpenSegmentAtTheEndOfTheAudio()
    {
        ScriptedVad vad = new(Script(10, 0));
        SileroVadStream stream = new(vad);
        using CpuBackend backend = new();
        float[] chunk = new float[Window];
        for (int i = 0; i < 10; i++)
        {
            stream.Push(backend, chunk, out _);
        }
        Assert.True(stream.Flush(out SileroVadSegment segment));
        Assert.Equal(0, segment.StartSample);
        Assert.Equal(10 * Window, segment.EndSample);
        Assert.False(stream.InSpeech);
    }

    [Fact]
    public void Reset_ClearsTheStreamAndTheModel()
    {
        ScriptedVad vad = new(Script(10, 0));
        SileroVadStream stream = new(vad);
        using CpuBackend backend = new();
        float[] chunk = new float[Window];
        for (int i = 0; i < 5; i++)
        {
            stream.Push(backend, chunk, out _);
        }
        stream.Reset();
        Assert.Equal(1, vad.Resets);
        Assert.Equal(0, stream.ConsumedSamples);
        Assert.False(stream.InSpeech);
        Assert.Equal(0f, stream.LastProbability);
        Assert.Equal(-1, stream.SpeechStartSample);
    }

    [Fact]
    public void ModelProperties_ReportWhatDrivesTheStream()
    {
        ScriptedVad vad = new(Script(1, 0));
        SileroVadStream stream = new(vad);
        Assert.Same(vad, stream.VadModel);
        Assert.Throws<InvalidOperationException>(() => stream.Model);
        Assert.Throws<ArgumentNullException>(() => new SileroVadStream((IVadModel)null!));
    }
}
