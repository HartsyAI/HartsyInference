namespace HartsyInference.Engine.Worlds;

/// <summary>A world model's camera action language: expands a script into camera-to-world poses before any weight loads.</summary>
public interface IWorldCameraScript
{
    /// <summary>Expands <paramref name="script"/> into one pose per frame, the seed frame first; each pose is 16 row-major
    /// values of a 4x4 camera-to-world matrix in OpenCV convention.</summary>
    /// <exception cref="FormatException">The script is not valid in the language.</exception>
    IReadOnlyList<float[]> Expand(string script);
}
