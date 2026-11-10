using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>Extracts a triangle mesh from a dual-grid field.</summary>
public interface IDualGridMesher
{
    /// <summary>Meshes <paramref name="field"/> in the world frame of <paramref name="grid"/>.</summary>
    Mesh Extract(DualGridField field, SceneGrid grid);
}
