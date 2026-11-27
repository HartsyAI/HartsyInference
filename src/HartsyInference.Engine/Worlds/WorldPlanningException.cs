namespace HartsyInference.Engine.Worlds;

/// <summary>A world request refused at planning time, before any weight is loaded. The message names the rejected
/// option and the values the model accepts.</summary>
public sealed class WorldPlanningException : ArgumentException
{
    /// <summary>Creates the refusal.</summary>
    public WorldPlanningException(string message)
        : base(message)
    {
    }
}
