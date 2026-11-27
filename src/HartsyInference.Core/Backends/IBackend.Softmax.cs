using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

// Row softmax, shared by any model that normalizes a small set of logits (block residual mixes, routers, gates).
public partial interface IBackend
{
    /// <summary>Softmax over the last axis of an F32 <c>[rows, cols]</c> tensor: per row <c>exp(x − max) / Σ exp(x − max)</c>.
    /// The host reference is the default; a backend with a device kernel overrides it.</summary>
    void SoftmaxRows(Tensor output, Tensor input, int rows, int cols) => SoftmaxReference.Rows(output, input, rows, cols);
}
