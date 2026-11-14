using HartsyInference.Engine.Requests;
using HartsyInference.ThreeD.Pipelines;

namespace HartsyInference.Engine.Services;

/// <summary>Where the remesh default of a mesh request is decided, so the mappers, the plan and the docs agree.</summary>
internal static class MeshRemesh
{
    /// <summary>Whether the textured post-process remeshes: the caller's choice when it made one, otherwise on for textured output at a grid below 1536
    /// (the remesh at 1536 would take more host memory than a 12 GB machine has) and off for shape-only output.</summary>
    public static bool Effective(MeshRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (request.Remesh is bool chosen)
            return chosen;
        int grid = request.GridResolution == 0 ? Trellis2ImageTo3DPipeline.DefaultGrid : request.GridResolution;
        return request.Texture && grid != 1536;
    }
}
