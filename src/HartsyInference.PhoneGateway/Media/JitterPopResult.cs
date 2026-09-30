namespace HartsyInference.PhoneGateway.Media;

/// <summary>Outcome of one <see cref="RtpJitterBuffer.Pop"/>.</summary>
public enum JitterPopResult
{
    /// <summary>No stream is established or the buffer is still pre-filling; the destination is untouched.</summary>
    Silence,
    /// <summary>A frame from the wire was copied to the destination.</summary>
    Frame,
    /// <summary>The frame was missing; the destination holds a concealment (the previous frame once, then silence).</summary>
    Concealed,
}
