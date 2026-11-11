using HartsyInference.Core.Backends;
using HartsyInference.Core.Sparse;
using HartsyInference.ThreeD.Models.TrellisV2;

namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>Decodes a texture latent into per-voxel PBR attributes on the shape decoder's voxels.</summary>
public interface ITrellis2TextureDecoderStage
{
    /// <summary>The attributes <c>[1, M, 6]</c> (base color 0..2, metallic 3, roughness 4, alpha 5) over the finest voxels, nominally in
    /// [0, 1] but not clamped: the bake clamps when it quantizes, as upstream clips after sampling. They follow <paramref name="guides"/>
    /// from the shape decode of the same latent coordinates. The caller owns the result.</summary>
    SparseTensor Decode(IBackend backend, SparseTensor textureLatent, ShapeVaeSubdivisions guides);
}
