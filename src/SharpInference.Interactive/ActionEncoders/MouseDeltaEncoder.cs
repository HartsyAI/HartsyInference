namespace SharpInference.Interactive.ActionEncoders;

/// <summary>Generic mouse-delta helper shared by the world-model action encoders: clamps a raw (Δx, Δy) pair into the
/// normalized camera-rotation range the models were trained on (Matrix-Game's canned bench uses ±0.1 per keystroke;
/// free-form play supplies arbitrary floats which are clamped to <paramref name="maxMagnitude"/>).</summary>
public static class MouseDeltaEncoder
{
    /// <summary>Writes the clamped (Δx, Δy) into <paramref name="destination"/> (length ≥ 2).</summary>
    public static void Encode(float deltaX, float deltaY, float maxMagnitude, Span<float> destination)
    {
        if (destination.Length < 2)
            throw new ArgumentException("destination must hold at least 2 floats.", nameof(destination));
        destination[0] = Math.Clamp(deltaX, -maxMagnitude, maxMagnitude);
        destination[1] = Math.Clamp(deltaY, -maxMagnitude, maxMagnitude);
    }
}
