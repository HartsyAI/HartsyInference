namespace HartsyInference.Engine.Worlds;

/// <summary>Whether a world model takes pinhole intrinsics alongside its camera path.</summary>
public enum WorldIntrinsicsPolicy
{
    /// <summary>The model takes no intrinsics; a request that carries them is refused.</summary>
    NotAccepted,

    /// <summary>Intrinsics are accepted and validated when given.</summary>
    Optional,

    /// <summary>Typed camera input is refused without intrinsics.</summary>
    Required,
}
