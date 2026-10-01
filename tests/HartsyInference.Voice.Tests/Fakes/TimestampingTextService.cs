using System.Diagnostics;
using System.Runtime.CompilerServices;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.Voice.Tests.Fakes;

/// <summary>Wraps a real <see cref="ITextService"/> and timestamps every <see cref="TextChunk"/> its
/// <see cref="StreamAsync"/> yields — the session-observed timeline, after the engine's stream filter has already
/// held or forwarded it, as opposed to <see cref="RecordingDiagnostics"/>'s raw per-token view from inside the
/// engine. Comparing the two timelines for the same turn separates "the model is still deciding the next token" from
/// "the filter/parser/channel path is holding text the model already produced". Pass-through otherwise.</summary>
internal sealed class TimestampingTextService(ITextService inner) : ITextService
{
    private readonly List<Entry> _events = [];
    private readonly object _lock = new();
    private readonly Stopwatch _clock = Stopwatch.StartNew();

    /// <summary>One streamed chunk: its kind, the forwarded text (if any), the stop reason (if any), and the
    /// wall-clock moment this decorator's caller observed it.</summary>
    public readonly record struct Entry(TextChunkKind Kind, string? Text, StopReason? Stop, double ElapsedMs);

    /// <summary>Every chunk observed so far, across every <see cref="StreamAsync"/> call, in arrival order.</summary>
    public IReadOnlyList<Entry> Events
    {
        get
        {
            lock (_lock)
            {
                return [.. _events];
            }
        }
    }

    /// <summary>Clears the recorded timeline, e.g. between turns so each turn's events can be read in isolation.</summary>
    public void Clear()
    {
        lock (_lock)
        {
            _events.Clear();
        }
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<TextChunk> StreamAsync(ModelSpec spec, TextRequest request, [EnumeratorCancellation] CancellationToken cancel = default)
    {
        await foreach (TextChunk chunk in inner.StreamAsync(spec, request, cancel).WithCancellation(cancel).ConfigureAwait(false))
        {
            Entry entry = new(chunk.Kind, chunk.Text, chunk.Stop, _clock.Elapsed.TotalMilliseconds);
            lock (_lock)
            {
                _events.Add(entry);
            }
            yield return chunk;
        }
    }

    /// <inheritdoc/>
    public Task<TextResult> GenerateAsync(ModelSpec spec, TextRequest request, CancellationToken cancel = default) =>
        inner.GenerateAsync(spec, request, cancel);

    /// <inheritdoc/>
    public int CountTokens(ModelSpec spec, string text) => inner.CountTokens(spec, text);

    /// <inheritdoc/>
    public bool Unload(string? device = null) => inner.Unload(device);
}
