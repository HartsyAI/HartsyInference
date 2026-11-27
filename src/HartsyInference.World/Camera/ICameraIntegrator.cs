namespace HartsyInference.World.Camera;

/// <summary>Turns the controls held on each frame into camera poses, one frame at a time, so a camera path can be extended
/// while a world is being generated. <typeparamref name="TControls"/> is the model's control set for one frame.</summary>
/// <typeparam name="TControls">The controls held during one frame.</typeparam>
public interface ICameraIntegrator<TControls>
{
    /// <summary>The current camera-to-world pose: 16 row-major floats, OpenCV convention. A new array each call.</summary>
    float[] Pose { get; }

    /// <summary>Advances one frame under <paramref name="controls"/> and returns the pose after it.</summary>
    float[] Step(TControls controls);
}
