using HartsyInference.Core.Backends;
using HartsyInference.Core.Sparse;
using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>Decodes a sparse latent into a dual-grid field on the finer grid.</summary>
public interface ISparseShapeDecoder
{
    /// <summary>Decodes a single-sample <paramref name="latent"/>. The caller keeps ownership of the latent.</summary>
    DualGridField Decode(IBackend backend, SparseTensor latent);
}
