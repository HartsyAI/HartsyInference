namespace HartsyInference.Engine.Worlds;

/// <summary>How a world model takes camera input.</summary>
public enum WorldCameraMode
{
    /// <summary>One camera-to-world pose per frame, carried by <see cref="Requests.WorldCameraPath.Poses"/>.</summary>
    PerFramePose,

    /// <summary>A camera path written in the model's action language (<see cref="Requests.WorldCameraPath.Script"/>), expanded
    /// to one pose per frame at planning time by <see cref="WorldCapabilities.CameraScript"/>.</summary>
    ActionScript,
}
