namespace HartsyInference.Engine.Worlds;

/// <summary>How a world model takes camera input.</summary>
public enum WorldCameraMode
{
    /// <summary>One camera-to-world pose per frame, carried by <see cref="Requests.WorldCameraPath"/>.</summary>
    PerFramePose,
}
