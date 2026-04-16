namespace SharpInference.Core.Exceptions;

/// <summary>Base exception for all SharpInference errors.</summary>
public class SharpInferenceException : Exception
{
    public SharpInferenceException() { }

    public SharpInferenceException(string message) : base(message) { }

    public SharpInferenceException(string message, Exception innerException) : base(message, innerException) { }
}
