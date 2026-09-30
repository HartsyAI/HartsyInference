using HartsyInference.Core.Exceptions;

namespace HartsyInference.PhoneGateway.Config;

/// <summary>The configuration file or the environment it names cannot be used; the message says what to fix.</summary>
public sealed class GatewayConfigException : HartsyInferenceException
{
    public GatewayConfigException() { }

    public GatewayConfigException(string message) : base(message) { }

    public GatewayConfigException(string message, Exception innerException) : base(message, innerException) { }
}
