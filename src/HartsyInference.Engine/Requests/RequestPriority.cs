namespace HartsyInference.Engine.Requests;

/// <summary>Queue priority of a text request when the server is busy. A higher priority is admitted first; equal priorities keep arrival (FIFO) order.</summary>
public enum RequestPriority
{
    /// <summary>Admitted after normal and high requests.</summary>
    Low,

    /// <summary>The default.</summary>
    Normal,

    /// <summary>Admitted before normal and low requests.</summary>
    High,
}
