namespace SharpInference.Core.Exceptions;

/// <summary>Thrown when a model file's architecture or format is not supported by SharpInference.</summary>
public sealed class UnsupportedModelException : SharpInferenceException
{
    /// <summary>The architecture identifier that was not recognized.</summary>
    public string? Architecture { get; }

    /// <summary>The model format that was not supported.</summary>
    public string? Format { get; }

    public UnsupportedModelException(string message) : base(message) { }

    public UnsupportedModelException(string message, string? architecture, string? format)
        : base(message)
    {
        Architecture = architecture;
        Format = format;
    }

    public UnsupportedModelException(string message, Exception innerException) : base(message, innerException) { }
}
