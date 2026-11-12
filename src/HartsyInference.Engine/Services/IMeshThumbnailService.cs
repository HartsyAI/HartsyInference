using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Draws a picture of a model file on the server, with no GPU: a software rasteriser for meshes (glb, obj) and a point renderer for splat and point clouds (ply). It reads the file, not the model that made it, so every 3D model's output, and a model added later, is drawn the same way.</summary>
public interface IMeshThumbnailService
{
    /// <summary>The formats <see cref="Render"/> reads.</summary>
    IReadOnlyList<string> Formats { get; }

    /// <summary>Draws <paramref name="data"/>.</summary>
    /// <param name="data">The file bytes.</param>
    /// <param name="format">The file's format, one of <see cref="Formats"/>.</param>
    /// <param name="request">Size, view and background; null takes the defaults.</param>
    /// <exception cref="NotSupportedException">The format is not supported.</exception>
    /// <exception cref="ArgumentException">An option is out of range or names an unknown view.</exception>
    /// <exception cref="Core.Exceptions.HartsyInferenceException">The file is malformed or has nothing to draw.</exception>
    MeshThumbnail Render(byte[] data, string format, MeshThumbnailRequest? request = null);
}
