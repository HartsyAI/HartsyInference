using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection;

/// <summary>Weight source for the Grounding DINO modules. A single <c>Build</c> pass per module resolves every
/// parameter through this abstraction, so the exact key names and shapes are declared once and shared by both the
/// real checkpoint loader (<see cref="DictionaryWeightBag"/>) and the synthetic test generator
/// (<see cref="SyntheticWeightBag"/>) — there is no second copy of the key list to drift out of sync.</summary>
public interface IGroundingWeightBag
{
    /// <summary>Resolves the named parameter with the given row-major dimensions, returning an F32 tensor.</summary>
    Tensor Get(string name, params int[] dims);
}
