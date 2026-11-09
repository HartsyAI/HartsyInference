using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Decode;

/// <summary>One of q, k or v inside a projection output: the F32 tensor that holds it and the element its section starts at.
/// A section is read, never written; <see cref="Offset"/> 0 on a tensor of the section's own size is a tensor of its own.</summary>
/// <param name="Tensor">The projection output the section lives in.</param>
/// <param name="Offset">The first element of the section.</param>
public readonly record struct QkvSection(Tensor Tensor, long Offset)
{
    /// <summary>A tensor that holds the section alone.</summary>
    public static implicit operator QkvSection(Tensor tensor) => new(tensor, 0);
}
