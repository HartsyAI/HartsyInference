using HartsyInference.Engine.Requests;
using HartsyInference.ThreeD.Operations;

namespace HartsyInference.Engine.Services;

/// <summary>Edits a model file: decimate, fill holes, remove floating pieces, smooth, recompute normals, recentre, rescale. It reads the file, not the model that made it, so it works on the output of every 3D model, and on a model added later; no model profile is involved. Operations keep textures, UVs and colours wherever the edit leaves the vertices alone, and say so when they cannot.</summary>
public interface IMeshOperationService
{
    /// <summary>The operations, with the numbers each takes (name, default, range).</summary>
    IReadOnlyList<IMeshOperation> Operations { get; }

    /// <summary>Applies <paramref name="steps"/> in order to <paramref name="data"/> and writes the result as <paramref name="target"/>.</summary>
    /// <param name="data">The file bytes (glb or obj; see <see cref="IMeshConversionService.Sources"/>).</param>
    /// <param name="source">The file's format.</param>
    /// <param name="steps">Between 1 and 16 steps; each names an operation and its numbers, which are validated before any step runs.</param>
    /// <param name="target">The output format, one of <see cref="IMeshConversionService.Targets"/>; <c>glb</c> keeps textures and the settings block.</param>
    /// <param name="name">The base name of the output files; <c>model</c> when null.</param>
    /// <param name="cancel">Observed between steps and inside the long ones.</param>
    /// <exception cref="NotSupportedException">The source, target or an operation is not supported.</exception>
    /// <exception cref="ArgumentException">A step's numbers are invalid.</exception>
    /// <exception cref="Core.Exceptions.HartsyInferenceException">The file is malformed or holds no triangles.</exception>
    MeshOperationResult Apply(byte[] data, string source, IReadOnlyList<MeshOperationStep> steps, string target = "glb", string? name = null, CancellationToken cancel = default);
}
