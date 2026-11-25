namespace HartsyInference.Core.Exceptions;

/// <summary>A file was needed from the network and nobody allowed the download.</summary>
public sealed class DownloadRefusedException(string message) : HartsyInferenceException(message);
