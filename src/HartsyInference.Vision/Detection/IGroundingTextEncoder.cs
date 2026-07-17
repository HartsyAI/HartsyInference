using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection;

/// <summary>Text tower abstraction for Grounding DINO: turns caption token ids into contextual token embeddings
/// <c>[1, numTokens, TextDim]</c>. Grounding DINO injects the caption through a frozen BERT encoder, but this
/// interface deliberately decouples the detector from any concrete BERT implementation.
///
/// <para><b>Why an interface and not a direct BERT dependency.</b> A working <c>BertModel</c> already exists in
/// <c>HartsyInference.Audio</c> (built for GPT-SoVITS / MeloTTS text conditioning), but
/// <c>HartsyInference.Vision</c> does not — and must not — reference <c>HartsyInference.Audio</c>: doing so would
/// drag the entire audio stack across a package boundary for the sake of one encoder. So Grounding DINO depends on
/// this small interface via dependency injection instead of on <c>BertModel</c> directly.</para>
///
/// <para><b>Recommendation for real deployment.</b> The real deployment supplies a BERT-backed implementation;
/// because Vision cannot reference Audio, the recommendation is to hoist
/// <c>HartsyInference.Audio/Models/Bert/{BertModel,BertConfig}.cs</c> into a shared package (e.g.
/// <c>HartsyInference.Core</c> or a new <c>HartsyInference.TextEncoders</c>) that both Audio and Vision can depend
/// on, then adapt it to <see cref="IGroundingTextEncoder"/>. Duplicating BERT is explicitly avoided.</para></summary>
public interface IGroundingTextEncoder
{
    /// <summary>Width of the token embeddings this encoder produces (BERT-base = 768).</summary>
    int TextDim { get; }

    /// <summary>Encodes caption token ids to contextual embeddings <c>[1, tokenIds.Length, TextDim]</c>.
    /// Caller owns the returned tensor.</summary>
    Tensor Encode(IBackend backend, ReadOnlySpan<int> tokenIds);
}
