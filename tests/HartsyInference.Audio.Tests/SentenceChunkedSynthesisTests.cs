using System.Threading.Channels;
using HartsyInference.Audio.Frontends;
using HartsyInference.Audio.Streaming;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>The generic sentence-streaming loop with a fake synthesizer: order, offsets, what happens when the
/// consumer cancels, breaks or the synth throws, and that a bounded channel really holds the producer back. All
/// of it would fail silently in production — a hung stream or a lost sentence — which is what earns it a test.</summary>
public sealed class SentenceChunkedSynthesisTests
{
    private const int Rate = 24_000;
    private const string ThreeSentences =
        "The weather tomorrow is clear and mild. Rain arrives on Thursday evening. Bring a coat with you.";

    private static readonly Func<Func<float[]>, CancellationToken, Task<float[]>> RunOnPool =
        static (work, ct) => Task.Run(work, ct);

    /// <summary>Samples that identify their sentence: length is four per character, first sample its length.</summary>
    private static float[] FakeSynth(string sentence)
    {
        float[] samples = new float[sentence.Length * 4];
        samples[0] = sentence.Length;
        return samples;
    }

    [Fact]
    public async Task StreamBySentence_EmitsEachSentenceInOrderWithCumulativeOffsets()
    {
        List<string> synthesized = [];
        List<AudioChunk> chunks = [];
        await foreach (AudioChunk chunk in SentenceChunkedSynthesis.StreamBySentence(ThreeSentences, Rate,
            (s, _) => { lock (synthesized) synthesized.Add(s); return FakeSynth(s); }, RunOnPool,
            SentenceSplitter.MinSentenceLength, SentenceChunkedSynthesis.NoClauseLimit, CancellationToken.None))
        {
            chunks.Add(chunk);
        }

        IReadOnlyList<string> expected = SentenceSplitter.Split(ThreeSentences);
        Assert.Equal(3, expected.Count);
        Assert.Equal(expected, synthesized);
        Assert.Equal(3, chunks.Count);
        long offset = 0;
        for (int i = 0; i < chunks.Count; i++)
        {
            Assert.Equal(Rate, chunks[i].SampleRate);
            Assert.Equal(1, chunks[i].Channels);
            Assert.Equal(offset, chunks[i].StartSampleOffset);
            Assert.Equal(expected[i].Length, chunks[i].Samples[0]);
            Assert.Equal(expected[i].Length * 4, chunks[i].Samples.Length);
            offset += chunks[i].Samples.Length;
        }
    }

    [Fact]
    public async Task StreamBySentence_CancelledBetweenSentences_StopsBeforeTheNextJob()
    {
        IReadOnlyList<string> sentences = SentenceSplitter.Split(ThreeSentences);
        using CancellationTokenSource cts = new();
        using ManualResetEventSlim secondJobMayFinish = new(false);
        List<string> synthesized = [];
        float[] Synth(string s, CancellationToken ct)
        {
            lock (synthesized) synthesized.Add(s);
            if (s == sentences[1])
            {
                // Hold the second job until the consumer has cancelled, so the check after it is what stops us.
                secondJobMayFinish.Wait(TimeSpan.FromSeconds(10));
            }
            return FakeSynth(s);
        }

        int received = 0;
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
        {
            await foreach (AudioChunk _ in SentenceChunkedSynthesis.StreamBySentence(ThreeSentences, Rate, Synth, RunOnPool,
                SentenceSplitter.MinSentenceLength, SentenceChunkedSynthesis.NoClauseLimit, cts.Token))
            {
                received++;
                cts.Cancel();
                secondJobMayFinish.Set();
            }
        });
        Assert.Equal(1, received);
        Assert.DoesNotContain(sentences[2], synthesized);
    }

    [Fact]
    public async Task StreamBySentence_ProducerFault_SurfacesToTheConsumer()
    {
        IReadOnlyList<string> sentences = SentenceSplitter.Split(ThreeSentences);
        int received = 0;
        InvalidOperationException fault = await Assert.ThrowsAsync<InvalidOperationException>(async () =>
        {
            await foreach (AudioChunk _ in SentenceChunkedSynthesis.StreamBySentence(ThreeSentences, Rate,
                (s, _) => s == sentences[1] ? throw new InvalidOperationException("vocoder exploded") : FakeSynth(s), RunOnPool,
                SentenceSplitter.MinSentenceLength, SentenceChunkedSynthesis.NoClauseLimit, CancellationToken.None))
            {
                received++;
            }
        });
        Assert.Equal("vocoder exploded", fault.Message);
        Assert.Equal(1, received);
    }

    [Fact]
    public async Task StreamBySentence_AJobsOwnCancellation_IsNotMistakenForAbandonment()
    {
        // The synth cancels itself (an internal timeout, say) with the caller's token untouched. The consumer
        // drains what was produced and must then see that cancellation, not a clean, short stream.
        IReadOnlyList<string> sentences = SentenceSplitter.Split(ThreeSentences);
        int received = 0;
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
        {
            await foreach (AudioChunk _ in SentenceChunkedSynthesis.StreamBySentence(ThreeSentences, Rate,
                (s, _) => s == sentences[1] ? throw new OperationCanceledException("synth timed out") : FakeSynth(s), RunOnPool,
                SentenceSplitter.MinSentenceLength, SentenceChunkedSynthesis.NoClauseLimit, CancellationToken.None))
            {
                received++;
            }
        });
        Assert.Equal(1, received);
    }

    [Fact]
    public async Task StreamBySentence_ConsumerBreakingWithoutCancelling_DoesNotHang()
    {
        const string Many = "One sentence that is long enough to stand alone. Two sentences that are long enough to stand alone. "
            + "Three sentences that are long enough to stand alone. Four sentences that are long enough to stand alone. "
            + "Five sentences that are long enough to stand alone. Six sentences that are long enough to stand alone.";
        int calls = 0;
        Task consume = Task.Run(async () =>
        {
            await foreach (AudioChunk _ in SentenceChunkedSynthesis.StreamBySentence(Many, Rate,
                (s, _) => { Interlocked.Increment(ref calls); return FakeSynth(s); }, RunOnPool,
                SentenceSplitter.MinSentenceLength, SentenceChunkedSynthesis.NoClauseLimit, CancellationToken.None))
            {
                break;
            }
        });
        Task finished = await Task.WhenAny(consume, Task.Delay(TimeSpan.FromSeconds(10)));
        Assert.Same(consume, finished);
        await consume;
        Assert.True(calls < 6, $"the producer kept synthesizing after the consumer left ({calls} jobs)");
    }

    [Fact]
    public async Task StreamFromDeltas_SpeaksTheFirstSentenceBeforeTheTextIsFinished()
    {
        Channel<string> deltas = Channel.CreateUnbounded<string>();
        List<string> synthesized = [];
        IAsyncEnumerator<AudioChunk> stream = SentenceChunkedSynthesis.StreamFromDeltas(deltas.Reader.ReadAllAsync(), Rate,
            (s, _) => { lock (synthesized) synthesized.Add(s); return FakeSynth(s); }, RunOnPool,
            firstSentenceMinChars: 1, maxChars: SentenceChunkedSynthesis.NoClauseLimit, maxInFlight: 2, CancellationToken.None)
            .GetAsyncEnumerator();
        try
        {
            foreach (string delta in new[] { "Hello there", ", my go", "od friend. ", "That is all I" })
            {
                deltas.Writer.TryWrite(delta);
            }
            // The channel is still open: the first sentence must arrive regardless.
            ValueTask<bool> first = stream.MoveNextAsync();
            Task<bool> firstTask = first.AsTask();
            Assert.Same(firstTask, await Task.WhenAny(firstTask, Task.Delay(TimeSpan.FromSeconds(10))));
            Assert.True(await firstTask);
            Assert.Equal("Hello there, my good friend.".Length, stream.Current.Samples[0]);
            Assert.Equal(0, stream.Current.StartSampleOffset);

            deltas.Writer.TryWrite(" have to say.");
            deltas.Writer.Complete();
            Assert.True(await stream.MoveNextAsync());
            Assert.Equal("That is all I have to say.".Length, stream.Current.Samples[0]);
            Assert.Equal("Hello there, my good friend.".Length * 4, stream.Current.StartSampleOffset);
            Assert.False(await stream.MoveNextAsync());
        }
        finally
        {
            await stream.DisposeAsync();
        }
        Assert.Equal(["Hello there, my good friend.", "That is all I have to say."], synthesized);
    }

    [Fact]
    public void Arguments_AreValidated()
    {
        Func<string, CancellationToken, float[]> synth = (s, _) => FakeSynth(s);
        Assert.Throws<ArgumentOutOfRangeException>(() => SentenceChunkedSynthesis.StreamBySentence("x", 0, synth, RunOnPool, 1, 10, CancellationToken.None));
        Assert.Throws<ArgumentOutOfRangeException>(() => SentenceChunkedSynthesis.StreamBySentence("x", Rate, synth, RunOnPool, 1, 0, CancellationToken.None));
        Assert.Throws<ArgumentNullException>(() => SentenceChunkedSynthesis.StreamBySentence("x", Rate, null!, RunOnPool, 1, 10, CancellationToken.None));
        Assert.Throws<ArgumentNullException>(() => SentenceChunkedSynthesis.StreamFromDeltas(null!, Rate, synth, RunOnPool, 1, 10, 1, CancellationToken.None));
        Assert.Throws<ArgumentOutOfRangeException>(() => SentenceChunkedSynthesis.StreamFromDeltas(Deltas([]), Rate, synth, RunOnPool, 1, 10, 0, CancellationToken.None));
    }

    private static async IAsyncEnumerable<string> Deltas(IEnumerable<string> pieces)
    {
        foreach (string piece in pieces)
        {
            yield return piece;
        }
        await Task.CompletedTask;
    }
}
