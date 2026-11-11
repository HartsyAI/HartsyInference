using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>Turns the prepared image into the conditioning tokens both flows cross-attend to.</summary>
public interface ITrellis2Conditioner
{
    /// <summary>Tokens <c>[1, L, C]</c> for an input of <paramref name="resolution"/> pixels. The caller owns the result.</summary>
    Tensor Encode(IBackend backend, Trellis2PreparedImage image, int resolution);
}
