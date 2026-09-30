using System.Collections.Concurrent;

namespace HartsyInference.Voice.Tests.Fakes;

/// <summary>Speech models without weights. Recognition returns scripted transcripts in order; synthesis returns
/// <see cref="SamplesPerSentence"/> samples of a constant marker, <c>n / 1000</c> for the n-th sentence synthesized, so
/// the played audio shows which sentence each sample came from. Records the thread every call ran on.</summary>
internal sealed class FakeSpeech(int sampleRate = 24_000) : IVoiceSpeech
{
    private int _sentences;

    public int SynthesisSampleRate { get; } = sampleRate;

    public int SamplesPerSentence { get; init; } = 2_400;

    /// <summary>Transcripts handed out by <see cref="Transcribe"/>, in order; an empty queue transcribes to "".</summary>
    public ConcurrentQueue<string> Transcripts { get; } = new();

    /// <summary>Lengths of the utterances recognized.</summary>
    public ConcurrentQueue<int> TranscribedSamples { get; } = new();

    /// <summary>Sentences synthesized, in order.</summary>
    public ConcurrentQueue<string> Synthesized { get; } = new();

    /// <summary>Managed thread ids of every call.</summary>
    public ConcurrentQueue<int> Threads { get; } = new();

    /// <summary>When set, synthesis waits for it, so a test controls how fast replies are produced.</summary>
    public ManualResetEventSlim? HoldSynthesis { get; init; }

    public bool Disposed { get; private set; }

    /// <summary>Thread the models were disposed on.</summary>
    public int DisposedOnThread { get; private set; }

    /// <summary>Set to make the next call throw <see cref="ObjectDisposedException"/>, as a revoked engine lease does,
    /// until <see cref="Reopen"/> runs.</summary>
    public bool Revoked { get; set; }

    /// <summary>When set, <see cref="Reopen"/> fails with it instead of restoring the models.</summary>
    public Exception? ReopenFailure { get; set; }

    public int Reopens { get; private set; }

    public string Transcribe(float[] audio)
    {
        ThrowIfRevoked();
        Threads.Enqueue(Environment.CurrentManagedThreadId);
        TranscribedSamples.Enqueue(audio.Length);
        return Transcripts.TryDequeue(out string? text) ? text : "";
    }

    public float[] Synthesize(string text)
    {
        ThrowIfRevoked();
        Threads.Enqueue(Environment.CurrentManagedThreadId);
        HoldSynthesis?.Wait(TimeSpan.FromSeconds(30));
        Synthesized.Enqueue(text);
        float[] samples = new float[SamplesPerSentence];
        Array.Fill(samples, Marker(Interlocked.Increment(ref _sentences)));
        return samples;
    }

    /// <summary>The value every sample of the <paramref name="sentence"/>-th synthesized sentence carries (1-based).</summary>
    public static float Marker(int sentence) => sentence / 1000f;

    public void Reopen()
    {
        Reopens++;
        if (ReopenFailure is not null)
        {
            throw ReopenFailure;
        }
        Revoked = false;
    }

    public void Dispose()
    {
        Disposed = true;
        DisposedOnThread = Environment.CurrentManagedThreadId;
    }

    private void ThrowIfRevoked()
    {
        if (Revoked)
        {
            throw new ObjectDisposedException("fake lease", "The engine released this lease.");
        }
    }
}
