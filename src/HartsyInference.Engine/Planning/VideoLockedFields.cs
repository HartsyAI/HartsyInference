namespace HartsyInference.Engine.Planning;

/// <summary>Sampling fields a checkpoint profile fixes to its declared recipe.</summary>
[Flags]
public enum VideoLockedFields
{
    /// <summary>No sampling field is profile-locked.</summary>
    None = 0,

    /// <summary>Denoising evaluation count.</summary>
    Steps = 1,

    /// <summary>Classifier-free guidance scale.</summary>
    CfgScale = 2,

    /// <summary>Video flow shift.</summary>
    FlowShift = 4,

    /// <summary>Audio flow shift.</summary>
    AudioFlowShift = 8,

    /// <summary>Numerical sampler.</summary>
    Sampler = 16,

    /// <summary>Sigma scheduler.</summary>
    Scheduler = 32,

    /// <summary>Output width and height.</summary>
    Geometry = 64,
}
