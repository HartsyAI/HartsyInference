using System.Globalization;

namespace HartsyInference.Engine.Worlds;

/// <summary>Oasis's keyboard controls: WASD moves, space jumps, the arrow keys turn the camera. A held set becomes one
/// <c>+</c>-joined action such as <c>forward+camera:1,0</c>, the form Oasis's action parser reads.</summary>
public sealed class OasisControlScheme : IWorldControlScheme
{
    /// <summary>One key, its legend label, and the action term it adds while held. The arrow keys have no term of their own: they
    /// feed the camera term, which is why their term is null.</summary>
    private sealed record Binding(string Key, string Label, string? Term);

    // The legend and the action both read this table, so a reordered or added key cannot drift from its action.
    private static readonly Binding[] Bindings =
    [
        new("W", "forward", "forward"),
        new("S", "back", "back"),
        new("A", "left", "left"),
        new("D", "right", "right"),
        new("Space", "jump", "jump"),
        new("LeftShift", "sneak", "sneak"),
        new("LeftCtrl", "sprint", "sprint"),
        new("F", "attack", "attack"),
        new("E", "use", "use"),
        new("G", "drop", "drop"),
        new("Tab", "inventory", "inventory"),
        new("Left", "look left", null),
        new("Right", "look right", null),
        new("Up", "look up", null),
        new("Down", "look down", null),
    ];

    private static readonly WorldControl[] ControlList =
        Bindings.Select(binding => new WorldControl(binding.Key, binding.Label)).ToArray();

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
        List<string> terms = new(Bindings.Length + 1);
        foreach (Binding binding in Bindings)
        {
            if (binding.Term is not null && WorldControlKeys.IsHeld(heldKeys, binding.Key))
            {
                terms.Add(binding.Term);
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
