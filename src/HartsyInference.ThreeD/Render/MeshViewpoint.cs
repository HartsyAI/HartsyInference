namespace HartsyInference.ThreeD.Render;

/// <summary>Where the camera is: an orbit around the model's centre. Azimuth 0 looks at the front (the camera on +Z), 90 from the right (+X), 180 from the back, -90 from the left; elevation 0 is level and 90 looks straight down.</summary>
/// <param name="AzimuthDegrees">Turn about the up axis.</param>
/// <param name="ElevationDegrees">Angle above the horizon, in [-90, 90].</param>
public sealed record MeshViewpoint(float AzimuthDegrees, float ElevationDegrees)
{
    /// <summary>The names <see cref="Named"/> accepts.</summary>
    public static IReadOnlyList<string> Names { get; } = ["front", "back", "left", "right", "top", "bottom", "iso"];

    /// <summary>A viewpoint by preset name (case-insensitive): <c>front</c>, <c>back</c>, <c>left</c>, <c>right</c>, <c>top</c>, <c>bottom</c> or <c>iso</c> (a three-quarter view from front-right and above).</summary>
    /// <exception cref="ArgumentException">The name is not one of <see cref="Names"/>.</exception>
    public static MeshViewpoint Named(string name) => name.ToLowerInvariant() switch
    {
        "front" => new MeshViewpoint(0, 0),
        "back" => new MeshViewpoint(180, 0),
        "left" => new MeshViewpoint(-90, 0),
        "right" => new MeshViewpoint(90, 0),
        "top" => new MeshViewpoint(0, 90),
        "bottom" => new MeshViewpoint(0, -90),
        "iso" => new MeshViewpoint(35, 25),
        _ => throw new ArgumentException($"View '{name}' is not one of {string.Join(", ", Names)}.", nameof(name)),
    };
}
