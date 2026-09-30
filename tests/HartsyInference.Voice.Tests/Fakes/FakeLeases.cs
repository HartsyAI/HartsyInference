using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.Voice.Tests.Fakes;

/// <summary>An engine synthesizer lease without weights: returns one sample holding its generation, throws
/// <see cref="ObjectDisposedException"/> once <see cref="Revoked"/>, as an engine release makes a real lease do, and
/// records the options it was called with.</summary>
internal sealed class FakeSynthesizerLease(int generation, bool throwOnDispose = false) : ISynthesizerLease
{
    public int Generation { get; } = generation;

    public int SampleRate => 24_000;

    public bool Revoked { get; set; }

    public bool Disposed { get; private set; }

    public SpeechRequest? LastOptions { get; private set; }

    public float[] Synthesize(string text, SpeechRequest options)
    {
        ObjectDisposedException.ThrowIf(Revoked, this);
        LastOptions = options;
        return [Generation];
    }

    public void Dispose()
    {
        Disposed = true;
        if (throwOnDispose)
        {
            throw new InvalidOperationException("closing the revoked lease failed");
        }
    }
}

/// <summary>An engine transcriber lease without weights: answers with its generation, throws
/// <see cref="ObjectDisposedException"/> once <see cref="Revoked"/>, and records the options it was called with.</summary>
internal sealed class FakeTranscriberLease(int generation) : ITranscriberLease
{
    public int Generation { get; } = generation;

    public bool Revoked { get; set; }

    public bool Disposed { get; private set; }

    public AudioRequest? LastOptions { get; private set; }

    public int LastSampleRate { get; private set; }

    public string Transcribe(ReadOnlySpan<float> pcm, int sampleRate, AudioRequest options)
    {
        ObjectDisposedException.ThrowIf(Revoked, this);
        LastOptions = options;
        LastSampleRate = sampleRate;
        return "generation " + Generation;
    }

    public void Dispose() => Disposed = true;
}
