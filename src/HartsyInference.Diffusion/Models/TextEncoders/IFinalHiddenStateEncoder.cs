using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.TextEncoders;

/// <summary>A causal language model used as a diffusion text encoder whose conditioning is its final, normalized hidden
/// state for every prompt token (transformers' <c>last_hidden_state</c>), e.g. MOSS-Image's Qwen3.5-9B. It is an
/// interface because such encoders live in the LLM package, which the diffusion pipelines do not reference; the engine
/// supplies the implementation.</summary>
public interface IFinalHiddenStateEncoder
{
    /// <summary>Width of the hidden state.</summary>
    int HiddenSize { get; }

    /// <summary>Encodes one prompt's token ids from a fresh state and returns <c>[1, ids.Count, HiddenSize]</c> F32 on the host.</summary>
    Tensor Encode(IBackend backend, IReadOnlyList<int> tokenIds);

    /// <summary>Every weight the encoder may place on the backend, so the pipeline can free them after encoding.</summary>
    IEnumerable<Tensor> EnumerateWeights();
}
