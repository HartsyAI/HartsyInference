using System.Globalization;

namespace HartsyInference.Engine.Worlds;

/// <summary>Oasis's keyboard controls: WASD moves, space jumps, the arrow keys turn the camera. A held set becomes one
/// <c>+</c>-joined action such as <c>forward+camera:1,0</c>, the form Oasis's action parser reads.</summary>
public sealed class OasisControlScheme : IWorldControlScheme
{
    private static readonly WorldControl[] ControlList =
    [
        new WorldControl("W", "forward"),
        new WorldControl("S", "back"),
        new WorldControl("A", "left"),
        new WorldControl("D", "right"),
        new WorldControl("Space", "jump"),
        new WorldControl("LeftShift", "sneak"),
        new WorldControl("LeftCtrl", "sprint"),
        new WorldControl("F", "attack"),
        new WorldControl("E", "use"),
        new WorldControl("G", "drop"),
        new WorldControl("Tab", "inventory"),
        new WorldControl("Left", "look left"),
        new WorldControl("Right", "look right"),
        new WorldControl("Up", "look up"),
        new WorldControl("Down", "look down"),
    ];

    private static readonly string[] KeyActions =
    [
        "forward", "back", "left", "right", "jump", "sneak", "sprint", "attack", "use", "drop", "inventory",
    ];

    /// <summary>The shared instance; the scheme has no state.</summary>
    public static OasisControlScheme Instance { get; } = new();

    /// <inheritdoc/>
    public string Name => "Oasis";

    /// <inheritdoc/>
    public WorldControlCadence Cadence => WorldControlCadence.PerStep;

    /// <inheritdoc/>
    public IReadOnlyList<WorldControl> Controls => ControlList;

    /// <inheritdoc/>
    public string Note =>
        "Oasis takes one action per step and each step renders from the last frame alone, so control lags by a frame.";

    /// <inheritdoc/>
    public string ActionFor(IReadOnlyCollection<string> heldKeys)
    {
        ArgumentNullException.ThrowIfNull(heldKeys);
        List<string> terms = new(KeyActions.Length + 1);
        for (int i = 0; i < KeyActions.Length; i++)
        {
            if (WorldControlKeys.IsHeld(heldKeys, ControlList[i].Key))
            {
                terms.Add(KeyActions[i]);
            }
        }
        int x = (WorldControlKeys.IsHeld(heldKeys, "Right") ? 1 : 0) - (WorldControlKeys.IsHeld(heldKeys, "Left") ? 1 : 0);
        int y = (WorldControlKeys.IsHeld(heldKeys, "Down") ? 1 : 0) - (WorldControlKeys.IsHeld(heldKeys, "Up") ? 1 : 0);
        if (x != 0 || y != 0)
        {
            terms.Add("camera:" + x.ToString(CultureInfo.InvariantCulture) + "," + y.ToString(CultureInfo.InvariantCulture));
        }
        return terms.Count == 0 ? "none" : string.Join('+', terms);
    }
}
