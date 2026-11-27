using HartsyInference.Core.Backends;
using HartsyInference.ThreeD.Models.Pixal3D;

namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>Turns the prepared image into the (global tokens, per-cell features) pair a Pixal3D flow cross-attends to.</summary>
public interface IPixal3DConditioner
{
    /// <summary>The conditioning for <paramref name="spec"/> seen from <paramref name="view"/>. The caller owns the result.</summary>
    Pixal3DCondition Encode(IBackend backend, Trellis2PreparedImage image, Pixal3DConditionSpec spec, Pixal3DView view);
}
