namespace HartsyInference.Engine.Worlds;

/// <summary>Key-name lookups shared by the built-in control schemes.</summary>
internal static class WorldControlKeys
{
    /// <summary>Whether <paramref name="key"/> is among the held keys, comparing names case-insensitively.</summary>
    internal static bool IsHeld(IReadOnlyCollection<string> heldKeys, string key)
    {
        foreach (string held in heldKeys)
        {
            if (string.Equals(held, key, StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }
        return false;
    }
}
