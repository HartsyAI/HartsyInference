namespace HartsyInference.Core.Numerics;

/// <summary>Validation for normalized scalar inputs that must remain finite.</summary>
public static class UnitInterval
{
    /// <summary>Whether <paramref name="value"/> is finite and lies in the inclusive interval <c>[0,1]</c>.</summary>
    public static bool Contains(float value) => float.IsFinite(value) && value >= 0f && value <= 1f;
}
