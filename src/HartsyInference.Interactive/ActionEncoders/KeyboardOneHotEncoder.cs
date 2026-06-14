namespace HartsyInference.Interactive.ActionEncoders;

/// <summary>Generic discrete-key one-hot helper shared by the world-model action encoders: one byte of key state
/// (0 = up, non-zero = down) becomes one float (0/1) per key slot.</summary>
public static class KeyboardOneHotEncoder
{
    /// <summary>Writes <paramref name="keyStates"/> as a 0/1 one-hot vector into <paramref name="destination"/>.</summary>
    public static void Encode(ReadOnlySpan<byte> keyStates, Span<float> destination)
    {
        if (destination.Length < keyStates.Length)
            throw new ArgumentException($"destination ({destination.Length}) shorter than key count ({keyStates.Length}).", nameof(destination));
        for (int i = 0; i < keyStates.Length; i++)
            destination[i] = keyStates[i] != 0 ? 1f : 0f;
    }
}
