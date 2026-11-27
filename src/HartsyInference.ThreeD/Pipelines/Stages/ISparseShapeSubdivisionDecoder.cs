using HartsyInference.Core.Backends;
using HartsyInference.Core.Sparse;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Models.TrellisV2;

namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>A shape decoder that can also keep the subdivisions it chose, which the texture decoder follows.</summary>
public interface ISparseShapeSubdivisionDecoder : ISparseShapeDecoder
{
    /// <summary>Decodes <paramref name="latent"/> and records each level's child logits in <paramref name="record"/>, which the caller owns.</summary>
    DualGridField Decode(IBackend backend, SparseTensor latent, ShapeVaeSubdivisions record);
}
