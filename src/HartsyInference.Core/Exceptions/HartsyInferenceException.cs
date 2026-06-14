namespace HartsyInference.Core.Exceptions;

/// <summary>Base exception for all HartsyInference errors.</summary>
public class HartsyInferenceException : Exception
{
    public HartsyInferenceException() { }

    public HartsyInferenceException(string message) : base(message) { }

    public HartsyInferenceException(string message, Exception innerException) : base(message, innerException) { }
}
