using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Recent text-encoder outputs by prompt text. The encoder is deterministic for a fixed checkpoint and presentation, so a
/// repeated prompt reuses its states and skips the encoder (ComfyUI caches conditioning the same way). Entries are copies: each
/// <see cref="Get"/> returns a fresh tensor, because the pipeline disposes the states it is given.</summary>
internal sealed class PromptEncodingCache(int capacity)
{
    private readonly object _gate = new();
    private readonly List<Entry> _entries = new();

    private sealed record Entry(string Prompt, float[] Values, TensorShape Shape, IReadOnlyList<(int Start, int Stop, int Tag)> TagRuns);

    internal int Count
    {
        get
        {
            lock (_gate)
            {
                return _entries.Count;
            }
        }
    }

    /// <summary>A fresh copy of the states encoded for <paramref name="prompt"/>, or null when that prompt is not cached.</summary>
    internal MiniMaxH3TextEncoder.Result? Get(string prompt)
    {
        lock (_gate)
        {
            Entry? hit = _entries.Find(entry => string.Equals(entry.Prompt, prompt, StringComparison.Ordinal));
            if (hit is null)
            {
                return null;
            }
            Tensor states = new Tensor(hit.Shape, DType.F32);
            hit.Values.AsSpan().CopyTo(states.AsSpan<float>());
            return new MiniMaxH3TextEncoder.Result { HiddenStates = states, TagRuns = hit.TagRuns };
        }
    }

    /// <summary>Stores a copy of <paramref name="result"/> for <paramref name="prompt"/>, evicting the oldest entries past capacity.</summary>
    internal void Put(string prompt, MiniMaxH3TextEncoder.Result result)
    {
        float[] values = result.HiddenStates.AsSpan<float>().ToArray();
        lock (_gate)
        {
            _entries.RemoveAll(entry => string.Equals(entry.Prompt, prompt, StringComparison.Ordinal));
            _entries.Add(new Entry(prompt, values, result.HiddenStates.Shape, result.TagRuns));
            while (_entries.Count > capacity)
            {
                _entries.RemoveAt(0);
            }
        }
    }
}
