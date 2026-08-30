namespace HartsyInference.Engine.Requests;

/// <summary>Shared scalar contracts for planning and executing a video control stream.</summary>
internal static class VideoControlValidation
{
    /// <summary>Whether a residual strength can be represented by the F32 model input without changing sign.</summary>
    internal static bool IsValidStrength(double strength) =>
        double.IsFinite(strength) && strength >= 0.0 && strength <= float.MaxValue;

    /// <summary>Whether an inclusive normalized denoise window is ordered and finite.</summary>
    internal static bool IsValidWindow(double start, double end) =>
        double.IsFinite(start) && double.IsFinite(end) && start >= 0.0 && end <= 1.0 && start <= end;
}
