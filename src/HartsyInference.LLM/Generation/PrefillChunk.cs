using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Generation;

/// <summary>One contiguous span of a sequence handed to <see cref="IGenerationModel.Prefill"/>; also the shape of a single decode step or a speculative verify pass.</summary>
/// <param name="TokenIds">Token ids of the span.</param>
/// <param name="PosStart">Absolute position of the first token; must equal the state's committed length.</param>
/// <param name="LastRowOnly">Return only the last position's hidden row, which is all sampling reads.</param>
/// <param name="Embeds">Optional <c>[1, T, hidden]</c> embedding override (image splice) used instead of looking up <paramref name="TokenIds"/>.</param>
/// <param name="TokenKinds">Optional per-token kind tags, same length as <paramref name="TokenIds"/>, for models that route by kind.</param>
public readonly record struct PrefillChunk(
    ReadOnlyMemory<int> TokenIds,
    int PosStart,
    bool LastRowOnly = false,
    Tensor? Embeds = null,
    ReadOnlyMemory<byte> TokenKinds = default);
