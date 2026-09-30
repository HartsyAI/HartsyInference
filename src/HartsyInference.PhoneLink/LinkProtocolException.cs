using HartsyInference.Core.Exceptions;

namespace HartsyInference.PhoneLink;

/// <summary>The peer sent bytes that do not form a valid frame or message: oversize payload, truncated stream, wrong type or
/// malformed body. The connection cannot be resynchronized and must be closed.</summary>
public sealed class LinkProtocolException : HartsyInferenceException
{
    public LinkProtocolException() { }

    public LinkProtocolException(string message) : base(message) { }

    public LinkProtocolException(string message, Exception innerException) : base(message, innerException) { }
}
