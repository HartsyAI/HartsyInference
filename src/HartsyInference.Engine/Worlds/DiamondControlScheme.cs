namespace HartsyInference.Engine.Worlds;

/// <summary>DIAMOND's Atari controls. One discrete action goes to each step: a direction key beats fire, both directions
/// cancel, and holding nothing is a no-op.</summary>
public sealed class DiamondControlScheme : IWorldControlScheme
{
    private static readonly WorldControl[] ControlList =
    [
        new WorldControl("Space", "fire"),
        new WorldControl("A", "left"),
        new WorldControl("Left", "left"),
        new WorldControl("D", "right"),
        new WorldControl("Right", "right"),
    ];

    /// <summary>The shared instance; the scheme has no state.</summary>
    public static DiamondControlScheme Instance { get; } = new();

    /// <inheritdoc/>
    public string Name => "DIAMOND";

    /// <inheritdoc/>
    public WorldControlCadence Cadence => WorldControlCadence.PerStep;

    /// <inheritdoc/>
    public IReadOnlyList<WorldControl> Controls => ControlList;

    /// <inheritdoc/>
    public string Note => "DIAMOND renders one frame per action, so the picture moves at the model's pace, not the frame rate.";

    /// <inheritdoc/>
    public string ActionFor(IReadOnlyCollection<string> heldKeys)
    {
        ArgumentNullException.ThrowIfNull(heldKeys);
        bool left = WorldControlKeys.IsHeld(heldKeys, "A") || WorldControlKeys.IsHeld(heldKeys, "Left");
        bool right = WorldControlKeys.IsHeld(heldKeys, "D") || WorldControlKeys.IsHeld(heldKeys, "Right");
        if (left != right)
        {
            return left ? "left" : "right";
        }
        if (left)
        {
            return "noop";
        }
        return WorldControlKeys.IsHeld(heldKeys, "Space") ? "fire" : "noop";
    }
}
