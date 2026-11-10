namespace HartsyInference.World.Camera;

/// <summary>The SANA-WM camera controls held during one frame, as the action DSL keys map to them
/// (<c>camera_control.DSL_KEY_TO_CONTROL</c>).</summary>
[Flags]
public enum SanaWmControls
{
    /// <summary>No key held.</summary>
    None = 0,

    /// <summary><c>w</c>: translate forward along the heading.</summary>
    Forward = 1 << 0,

    /// <summary><c>s</c>: translate back along the heading.</summary>
    Back = 1 << 1,

    /// <summary><c>a</c>: yaw left.</summary>
    YawLeft = 1 << 2,

    /// <summary><c>d</c>: yaw right.</summary>
    YawRight = 1 << 3,

    /// <summary><c>i</c>: pitch up.</summary>
    PitchUp = 1 << 4,

    /// <summary><c>k</c>: pitch down.</summary>
    PitchDown = 1 << 5,

    /// <summary><c>j</c>: strafe left.</summary>
    StrafeLeft = 1 << 6,

    /// <summary><c>l</c>: strafe right.</summary>
    StrafeRight = 1 << 7,
}
