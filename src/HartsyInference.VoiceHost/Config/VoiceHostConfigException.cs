using HartsyInference.Core.Exceptions;

namespace HartsyInference.VoiceHost.Config;

/// <summary>The configuration file, or a secret file it names, cannot be used; the message says what to fix.</summary>
public sealed class VoiceHostConfigException : HartsyInferenceException
{
    public VoiceHostConfigException() { }

    public VoiceHostConfigException(string message) : base(message) { }

    public VoiceHostConfigException(string message, Exception innerException) : base(message, innerException) { }
}
