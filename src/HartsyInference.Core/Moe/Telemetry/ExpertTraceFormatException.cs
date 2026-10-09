namespace HartsyInference.Core.Moe.Telemetry;

/// <summary>A trace stream is not a valid version of the expert trace format.</summary>
public sealed class ExpertTraceFormatException : Exception
{
    /// <summary>Creates the exception with a description of what was rejected.</summary>
    public ExpertTraceFormatException(string message) : base(message)
    {
    }
}
