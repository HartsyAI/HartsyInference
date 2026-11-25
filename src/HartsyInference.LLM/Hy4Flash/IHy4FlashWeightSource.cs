using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Hy4Flash;

/// <summary>Where a Hy4 Flash model reads its tensors from, by checkpoint key.</summary>
public interface IHy4FlashWeightSource
{
    /// <summary>The tensor as the caller may dispose it; it may be a view of storage the source keeps alive. Floating-point tensors come in the dtype <c>IBackend.Linear</c> takes.</summary>
    /// <exception cref="KeyNotFoundException">The source has no tensor <paramref name="name"/>.</exception>
    Tensor Read(string name);

    /// <summary>Whether the source holds <paramref name="name"/>.</summary>
    bool Contains(string name);
}
