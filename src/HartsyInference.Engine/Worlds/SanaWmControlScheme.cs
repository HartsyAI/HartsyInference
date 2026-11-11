namespace HartsyInference.Engine.Worlds;

/// <summary>SANA-WM's live camera controls. Each held key sets one letter of the <c>wasdijkl</c> camera language, and the
/// stream applies the set to every frame until the next change.</summary>
public sealed class SanaWmControlScheme : IWorldControlScheme
{
    private const string Letters = "wsadikjl";

    private static readonly WorldControl[] ControlList =
    [
        new WorldControl("W", "forward"),
        new WorldControl("S", "back"),
        new WorldControl("A", "yaw left"),
        new WorldControl("D", "yaw right"),
        new WorldControl("I", "pitch up"),
        new WorldControl("K", "pitch down"),
        new WorldControl("J", "strafe left"),
        new WorldControl("L", "strafe right"),
    ];

    /// <summary>The shared instance; the scheme has no state.</summary>
    public static SanaWmControlScheme Instance { get; } = new();

    /// <inheritdoc/>
    public string Name => "SANA-WM camera";

    /// <inheritdoc/>
    public WorldControlCadence Cadence => WorldControlCadence.OnChange;

    /// <inheritdoc/>
    public IReadOnlyList<WorldControl> Controls => ControlList;

    /// <inheritdoc/>
    public string Note =>
        "The model moves the camera, not a character in the scene. W dollies toward the scene and can smear a close subject; "
        + "S pulls back and keeps it framed.";

    /// <inheritdoc/>
    public string ActionFor(IReadOnlyCollection<string> heldKeys)
    {
        ArgumentNullException.ThrowIfNull(heldKeys);
        char[] letters = new char[Letters.Length];
        int count = 0;
        for (int i = 0; i < ControlList.Length; i++)
        {
            if (WorldControlKeys.IsHeld(heldKeys, ControlList[i].Key))
            {
                letters[count++] = Letters[i];
            }
        }
        return count == 0 ? "none" : new string(letters, 0, count);
    }
}
