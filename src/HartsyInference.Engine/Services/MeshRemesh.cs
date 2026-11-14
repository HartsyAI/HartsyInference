using HartsyInference.Engine.Requests;
using HartsyInference.ThreeD.Pipelines;

namespace HartsyInference.Engine.Services;

/// <summary>Where the remesh default of a mesh request is decided, so the mappers, the plan and the docs agree.</summary>
internal static class MeshRemesh
{
    /// <summary>Whether the textured post-process remeshes: the caller's choice when it made one, otherwise on for textured output at a grid of 512 or 1024
    /// (the remesh at 1536 would need several times the host memory of the 1024 one and was not tried) and off for shape-only output. Pass the request
    /// with its quality preset applied, so a preset's grid counts.</summary>
    public static bool Effective(MeshRequest request, string modelId)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (request.Remesh is bool chosen)
            return chosen;
        int defaultGrid = modelId == MeshModelIds.Pixal3D ? Pixal3DImageTo3DPipeline.DefaultGrid : Trellis2ImageTo3DPipeline.DefaultGrid;
        int grid = request.GridResolution == 0 ? defaultGrid : request.GridResolution;
        return request.Texture && grid <= 1024;
    }
}
