namespace HartsyInference.Core.Rope;

/// <summary>Operation order for <c>position × frequency</c>. Results can differ in the last bit between the two forms.</summary>
public enum RopeAngleOrder
{
    /// <summary>Angle is <c>position × (1 / Pow(theta, 2k/d))</c>, the standard form.</summary>
    MultiplyByInverse,

    /// <summary>Angle is <c>position / Pow(theta, 2k/d)</c>, used by QwenImage21.</summary>
    DivideByDenominator,
}
