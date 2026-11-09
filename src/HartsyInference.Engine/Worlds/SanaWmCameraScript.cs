using HartsyInference.World.Camera;

namespace HartsyInference.Engine.Worlds;

/// <summary>SANA-WM's <c>--action</c> language (<c>&lt;keys&gt;-&lt;frames&gt;</c> segments over <c>wasdijkl</c>), rolled out by
/// <see cref="SanaWmAction"/> with upstream's smoothed velocity model. A script of N frames yields N + 1 poses.</summary>
public sealed class SanaWmCameraScript : IWorldCameraScript
{
    /// <summary>The shared instance; the language has no state.</summary>
    public static SanaWmCameraScript Instance { get; } = new();

    /// <inheritdoc/>
    public IReadOnlyList<float[]> Expand(string script) => SanaWmAction.ToCameraToWorld(script);
}
