using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Converts a model file between formats. It reads the file, not the model that made it, so it works for the output of every 3D model, for a model a host adds later, and for a file the host made itself; no model profile is involved.</summary>
public interface IMeshConversionService
{
    /// <summary>The formats <see cref="Convert(byte[], string, string, string?)"/> reads.</summary>
    IReadOnlyList<string> Sources { get; }

    /// <summary>The formats it writes.</summary>
    IReadOnlyList<string> Targets { get; }

    /// <summary>Converts <paramref name="data"/> to <paramref name="target"/>. Geometry, normals, UVs and vertex colours go through where the target carries them; a skin and animations never do (the output is the bind pose); the result's notes list every loss.</summary>
    /// <param name="data">The file bytes.</param>
    /// <param name="source">The file's format, one of <see cref="Sources"/>.</param>
    /// <param name="target">The format to write, one of <see cref="Targets"/>.</param>
    /// <param name="name">The base name for the output files; <c>model</c> when null.</param>
    /// <exception cref="NotSupportedException">The source or target is not supported, or the file is a Gaussian-splat PLY, which holds splats rather than a mesh.</exception>
    /// <exception cref="Core.Exceptions.HartsyInferenceException">The file is malformed or holds no triangles.</exception>
    MeshConversion Convert(byte[] data, string source, string target, string? name = null);
}
