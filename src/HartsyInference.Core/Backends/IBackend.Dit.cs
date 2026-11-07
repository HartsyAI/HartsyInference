using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

// Fused DiT block primitives with a host reference default; GPU backends override them.
public partial interface IBackend
{
    /// <summary>SiTU-GLU over the two halves of the last dimension: with <c>u</c> the first half and <c>g</c> the second,
    /// <c>out = (4·tanh(g/4)·sigmoid(g)) · (25·tanh(u/25))</c> (MOSS-Image's soft-capped gated FFN activation).
    /// <paramref name="input"/> is <c>[.., 2·F]</c> and <paramref name="output"/> <c>[.., F]</c>, both F32 or both F16.</summary>
    void SituGlu(Tensor output, Tensor input) => SituGluReference.Apply(output, input);
}
