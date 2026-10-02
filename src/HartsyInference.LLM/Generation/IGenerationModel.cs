using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Generation;

/// <summary>The one seam between generation drivers (pipeline, scheduler) and a model: sequence state, prefill, batched decode and logits, independent of how the model stores its KV or places its layers.</summary>
/// <remarks>Tensors returned from <see cref="Prefill"/> and <see cref="DecodeBatch"/> are hidden states owned by the caller; <see cref="ProjectLogits"/> turns them into F32 logits on <see cref="OutputBackend"/>.</remarks>
public interface IGenerationModel : IDisposable
{
    GenerationModelInfo Info { get; }

    GenerationCapabilities Capabilities { get; }

    /// <summary>Backend that owns the final hidden state, logits and sampler inputs.</summary>
    IBackend OutputBackend { get; }

    /// <summary>Creates an empty sequence sized by <paramref name="options"/>; the caller disposes it.</summary>
    ISequenceState CreateSequenceState(SequenceStateOptions options);

    /// <summary>A new sequence with room for <paramref name="capacity"/> tokens holding a device copy of
    /// <paramref name="state"/>'s committed tokens, or null when this model's state cannot be copied that way.
    /// <paramref name="state"/> is left unchanged and still owned by the caller, who also disposes the copy.</summary>
    ISequenceState? ResizeSequenceState(ISequenceState state, int capacity) => null;

    /// <summary>Runs <paramref name="chunk"/> against <paramref name="state"/>, commits its tokens, and returns hidden <c>[1, rows, hidden]</c> (rows = 1 when <see cref="PrefillChunk.LastRowOnly"/>).</summary>
    Tensor Prefill(in PrefillChunk chunk, ISequenceState state);

    /// <summary>As <see cref="Prefill(in PrefillChunk, ISequenceState)"/>, stopping with <see cref="OperationCanceledException"/>
    /// once <paramref name="cancel"/> is signalled. A stopped prefill commits nothing: <paramref name="state"/>'s length is
    /// unchanged. The default implementation checks only before the call. An implementation that checks during it must
    /// leave an uncancelled call's result untouched (<see cref="GenericTransformerModel"/> checks between layers), and a
    /// decorator must forward this overload itself, or the model it wraps is only checked before the call.</summary>
    Tensor Prefill(in PrefillChunk chunk, ISequenceState state, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        return Prefill(chunk, state);
    }

    /// <summary>One decode step for each sequence in <paramref name="states"/> (token i belongs to state i), committing one token per state; returns hidden <c>[1, N, hidden]</c>.</summary>
    Tensor DecodeBatch(ReadOnlySpan<int> tokenIds, ISequenceState[] states);

    /// <summary>Projects hidden <c>[1, rows, hidden]</c> to F32 logits <c>[1, rows, vocab]</c>.</summary>
    Tensor ProjectLogits(Tensor hidden, int rows);

    /// <summary>Every weight tensor, for residency planning.</summary>
    IEnumerable<Tensor> EnumerateWeights(bool includeRedundantSplits);

    /// <summary>Bytes one sequence of <paramref name="contextTokens"/> tokens will hold in device memory.</summary>
    long EstimateSequenceBytes(int contextTokens);

    /// <summary>Current device headroom on the output device.</summary>
    CapacitySnapshot Capacity();
}
