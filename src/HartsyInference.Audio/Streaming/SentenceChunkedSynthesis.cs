using System.Runtime.CompilerServices;
using HartsyInference.Audio.Frontends;
using HartsyInference.Core.Logging;

namespace HartsyInference.Audio.Streaming;

/// <summary>Turns any whole-utterance synthesizer into a sentence-at-a-time stream.</summary>
/// <remarks>A model with no incremental decode loop — Piper, Kokoro — returns nothing until its whole input is
/// done, so the unit of streaming is the sentence: the listener waits for one sentence's synthesis instead of the
/// passage's. Cutting a sentence in two changes its prosody, so the split is <see cref="SentenceSplitter"/>'s
/// conservative one, with <see cref="SentenceSplitter.SplitClauses"/> only for sentences longer than the model
/// can take. The output is not sample-identical to a single whole-text call for the same reason; it is identical
/// to synthesizing each sentence on its own.
/// <para>The shape is the producer/consumer of the streaming codec models: one producer task synthesizes and
/// enqueues, the caller drains an <see cref="AudioStreamer"/> bounded to <c>maxInFlight</c> chunks, the producer's
/// <see cref="AudioStreamer.Complete"/> runs in a <c>finally</c> so a faulted or cancelled producer always unblocks
/// the consumer, and the consumer's <c>finally</c> always awaits the producer so its exception surfaces even when
/// enumeration stopped early. A consumer that stops without cancelling cancels the producer itself, so a full
/// channel can never hold it forever. <c>run</c> is the scheduler seam: it is handed each synthesis job and decides
/// where it executes — <see cref="Task.Run(Func{Task})"/> for a CPU model, a post to a dedicated device thread for
/// a session that owns one. Cancellation is checked before every job, inside every job and before every
/// enqueue.</para></remarks>
public static class SentenceChunkedSynthesis
{
    /// <summary>Pass as <c>maxChars</c> to disable clause splitting.</summary>
    public const int NoClauseLimit = int.MaxValue;

    /// <summary>Chunks <see cref="StreamBySentence"/> lets the producer run ahead of the consumer: one being played, one ready.</summary>
    public const int DefaultMaxInFlight = 2;

    /// <summary>Streams <paramref name="text"/> one sentence at a time.</summary>
    /// <param name="text">The whole passage; split up front with <see cref="SentenceSplitter.Split"/>.</param>
    /// <param name="sampleRate">Rate of the samples <paramref name="synth"/> returns.</param>
    /// <param name="synth">Synthesizes one sentence to mono samples; runs wherever <paramref name="run"/> puts it.</param>
    /// <param name="run">Executes one synthesis job and returns its samples; the scheduler seam.</param>
    /// <param name="minChars">Shortest sentence emitted alone; shorter ones merge forward.</param>
    /// <param name="maxChars">Longest sentence synthesized whole; longer ones are clause-split. <see cref="NoClauseLimit"/> disables it.</param>
    /// <param name="ct">Stops synthesis between and inside jobs.</param>
    public static IAsyncEnumerable<AudioChunk> StreamBySentence(string text, int sampleRate,
        Func<string, CancellationToken, float[]> synth, Func<Func<float[]>, CancellationToken, Task<float[]>> run,
        int minChars, int maxChars, CancellationToken ct)
    {
        Validate(sampleRate, synth, run, maxChars, DefaultMaxInFlight);
        if (minChars < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(minChars), minChars, "minChars must be non-negative");
        }
        IReadOnlyList<string> sentences = SentenceSplitter.Split(text, minChars);
        return Stream(Enumerate(sentences), sentences.Count, sampleRate, synth, run, maxChars, DefaultMaxInFlight, ct);
    }

    /// <summary>Streams text that is still being written, sentence by sentence as each one completes.</summary>
    /// <param name="deltas">Text fragments in order, as a language model emits them.</param>
    /// <param name="sampleRate">Rate of the samples <paramref name="synth"/> returns.</param>
    /// <param name="synth">Synthesizes one sentence to mono samples; runs wherever <paramref name="run"/> puts it.</param>
    /// <param name="run">Executes one synthesis job and returns its samples; the scheduler seam.</param>
    /// <param name="firstSentenceMinChars">Shortest first sentence, so the opening words are spoken early; later
    /// sentences use <see cref="SentenceSplitter.MinSentenceLength"/>.</param>
    /// <param name="maxChars">Longest sentence synthesized whole; longer ones are clause-split.</param>
    /// <param name="maxInFlight">Chunks the producer may synthesize ahead of the consumer.</param>
    /// <param name="ct">Stops synthesis between and inside jobs, and stops reading <paramref name="deltas"/>.</param>
    public static IAsyncEnumerable<AudioChunk> StreamFromDeltas(IAsyncEnumerable<string> deltas, int sampleRate,
        Func<string, CancellationToken, float[]> synth, Func<Func<float[]>, CancellationToken, Task<float[]>> run,
        int firstSentenceMinChars, int maxChars, int maxInFlight, CancellationToken ct)
    {
        ArgumentNullException.ThrowIfNull(deltas);
        Validate(sampleRate, synth, run, maxChars, maxInFlight);
        if (firstSentenceMinChars < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(firstSentenceMinChars), firstSentenceMinChars, "firstSentenceMinChars must be non-negative");
        }
        return Stream(SentencesFrom(deltas, firstSentenceMinChars, ct), -1, sampleRate, synth, run, maxChars, maxInFlight, ct);
    }

    private static void Validate(int sampleRate, Func<string, CancellationToken, float[]> synth,
        Func<Func<float[]>, CancellationToken, Task<float[]>> run, int maxChars, int maxInFlight)
    {
        ArgumentNullException.ThrowIfNull(synth);
        ArgumentNullException.ThrowIfNull(run);
        if (sampleRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sampleRate), sampleRate, "sampleRate must be positive");
        }
        if (maxChars <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxChars), maxChars, "maxChars must be positive");
        }
        if (maxInFlight <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxInFlight), maxInFlight, "maxInFlight must be positive");
        }
    }

    private static async IAsyncEnumerable<string> Enumerate(IReadOnlyList<string> sentences)
    {
        for (int i = 0; i < sentences.Count; i++)
        {
            yield return sentences[i];
        }
        await Task.CompletedTask.ConfigureAwait(false);
    }

    private static async IAsyncEnumerable<string> SentencesFrom(IAsyncEnumerable<string> deltas, int firstSentenceMinChars,
        [EnumeratorCancellation] CancellationToken ct)
    {
        StreamingSentenceSplitter splitter = new(SentenceSplitter.MinSentenceLength, firstSentenceMinChars);
        await foreach (string delta in deltas.WithCancellation(ct).ConfigureAwait(false))
        {
            IReadOnlyList<string> completed = splitter.Push(delta);
            for (int i = 0; i < completed.Count; i++)
            {
                yield return completed[i];
            }
        }
        string? tail = splitter.Flush();
        if (tail is not null)
        {
            yield return tail;
        }
    }

    private static async IAsyncEnumerable<AudioChunk> Stream(IAsyncEnumerable<string> sentences, int announcedCount,
        int sampleRate, Func<string, CancellationToken, float[]> synth, Func<Func<float[]>, CancellationToken, Task<float[]>> run,
        int maxChars, int maxInFlight, [EnumeratorCancellation] CancellationToken ct)
    {
        using CancellationTokenSource abandon = CancellationTokenSource.CreateLinkedTokenSource(ct);
        CancellationToken token = abandon.Token;
        using AudioStreamer streamer = new(maxInFlight);
        Task producer = Task.Run(() => ProduceAsync(sentences, announcedCount, sampleRate, synth, run, maxChars, streamer, token), token);
        bool drained = false;
        try
        {
            await foreach (AudioChunk chunk in streamer.ReadAllAsync(token).ConfigureAwait(false))
            {
                yield return chunk;
            }
            drained = true;
        }
        finally
        {
            // A consumer that stopped without cancelling must not leave the producer parked on a full channel.
            if (!drained)
            {
                abandon.Cancel();
            }
            try
            {
                await producer.ConfigureAwait(false);
            }
            catch (OperationCanceledException) when (!drained && !ct.IsCancellationRequested)
            {
                // Only the abandonment above is expected here; a job's own cancellation after a drained channel
                // would otherwise pass as a clean, short stream, so it propagates.
                Logs.Debug("[Audio][SentenceStream] Consumer stopped early; the remaining sentences were not synthesized.");
            }
        }
    }

    private static async Task ProduceAsync(IAsyncEnumerable<string> sentences, int announcedCount, int sampleRate,
        Func<string, CancellationToken, float[]> synth, Func<Func<float[]>, CancellationToken, Task<float[]>> run,
        int maxChars, AudioStreamer streamer, CancellationToken token)
    {
        try
        {
            long started = Environment.TickCount64;
            long offset = 0;
            int emitted = 0;
            await foreach (string sentence in sentences.WithCancellation(token).ConfigureAwait(false))
            {
                IReadOnlyList<string> pieces = SentenceSplitter.SplitClauses(sentence, maxChars);
                for (int i = 0; i < pieces.Count; i++)
                {
                    string piece = pieces[i];
                    token.ThrowIfCancellationRequested();
                    float[] samples = await run(() =>
                    {
                        token.ThrowIfCancellationRequested();
                        return synth(piece, token);
                    }, token).ConfigureAwait(false);
                    if (samples is null || samples.Length == 0)
                    {
                        continue;
                    }
                    if (emitted == 0)
                    {
                        string count = announcedCount >= 0 ? $" of {announcedCount}" : "";
                        Logs.Verbose($"[Audio][SentenceStream] First{count} sentence(s) ready in {Environment.TickCount64 - started}ms.");
                    }
                    emitted++;
                    token.ThrowIfCancellationRequested();
                    await streamer.Put(new AudioChunk(samples, sampleRate, Channels: 1, StartSampleOffset: offset), token).ConfigureAwait(false);
                    offset += samples.Length;
                }
            }
        }
        finally
        {
            streamer.Complete();
        }
    }
}
