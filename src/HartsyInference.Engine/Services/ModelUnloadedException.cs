namespace HartsyInference.Engine.Services;

/// <summary>A request was queued on a loaded model that was unloaded before it could run. Retryable: the model is reloaded on the next request.</summary>
public sealed class ModelUnloadedException(string message) : Exception(message);
