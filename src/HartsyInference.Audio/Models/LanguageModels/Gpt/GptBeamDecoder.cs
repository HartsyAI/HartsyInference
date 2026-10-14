using HartsyInference.Audio.Sampling;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.Audio.Models.LanguageModels.Gpt;

/// <summary>Per-beam decoding state for a <see cref="GptBackbone"/> under <see cref="BeamSearch"/>: one KV cache per
/// beam, the shared prefix prefilled into the first, and the caches reassigned or copied when beams reorder. The
/// caller supplies how a token becomes the step's input embedding and how the step's hidden state becomes logits.</summary>
/// <remarks>Each step runs one forward per live beam, so beam search costs about <c>beams ×</c> the GPT decode compute
/// of a single sequence, plus a copy of a beam's committed KV rows whenever two beams continue from the same parent.</remarks>
internal sealed unsafe class GptBeamDecoder : IBeamSearchModel, IDisposable
{
    private readonly IBackend _backend;
    private readonly GptBackbone _gpt;
    private readonly Func<int, int, Tensor> _embedStep;
    private readonly Func<Tensor, Tensor> _logits;
    private FixedKvCache[] _caches;
    private FixedKvCache[] _next;
    private readonly bool[] _moved;
    private int _disposed;

    /// <param name="embedStep">(token, step) to the step input <c>[1, 1, hidden]</c> with its position applied.</param>
    /// <param name="logits">Hidden state <c>[1, 1, hidden]</c> to next-token logits <c>[1, 1, vocab]</c>.</param>
    public GptBeamDecoder(IBackend backend, GptBackbone gpt, int beams, Tensor prefix, Func<int, int, Tensor> embedStep,
        Func<Tensor, Tensor> logits)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(gpt);
        ArgumentNullException.ThrowIfNull(prefix);
        ArgumentOutOfRangeException.ThrowIfLessThan(beams, 1);
        _backend = backend;
        _gpt = gpt;
        _embedStep = embedStep;
        _logits = logits;
        _caches = new FixedKvCache[beams];
        _next = new FixedKvCache[beams];
        _moved = new bool[beams];
        try
        {
            for (int b = 0; b < beams; b++)
                _caches[b] = gpt.CreateCache() as FixedKvCache ?? throw new NotSupportedException("Beam search needs a FixedKvCache.");
            gpt.Forward(backend, prefix, nonCausal: false, _caches[0], positionsApplied: true).Dispose();
        }
        catch
        {
            foreach (FixedKvCache? c in _caches) c?.Dispose();
            throw;
        }
    }

    public void Step(int beam, int token, int step, Span<float> logits)
    {
        using Tensor input = _embedStep(token, step);
        using Tensor hidden = _gpt.ForwardStep(_backend, input, _caches[beam], positionsApplied: true);
        using Tensor result = _logits(hidden);
        if (result.ElementCount != logits.Length) throw new InvalidOperationException("Logit width does not match the vocabulary.");
        new ReadOnlySpan<float>((float*)result.DataPointer, logits.Length).CopyTo(logits);
    }

    public void Reorder(ReadOnlySpan<int> parents)
    {
        int n = _caches.Length;
        if (parents.Length != n) throw new ArgumentException("One parent per beam is required.", nameof(parents));
        Array.Clear(_moved);
        Array.Clear(_next);
        // A parent's cache moves to the first slot that continues it; later slots sharing it copy into a cache whose
        // beam died this step.
        for (int b = 0; b < n; b++)
        {
            int p = parents[b];
            if (_moved[p]) continue;
            _moved[p] = true;
            _next[b] = _caches[p];
        }
        int free = 0;
        for (int b = 0; b < n; b++)
        {
            if (_next[b] is not null) continue;
            while (_moved[free]) free++;
            FixedKvCache target = _caches[free++];
            target.CopyFrom(_caches[parents[b]]);
            _next[b] = target;
        }
        (_caches, _next) = (_next, _caches);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (FixedKvCache c in _caches) c.Dispose();
    }
}
