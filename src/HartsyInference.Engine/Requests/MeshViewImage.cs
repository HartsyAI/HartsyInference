namespace HartsyInference.Engine.Requests;

/// <summary>The side of the subject an extra picture shows (<see cref="MeshViewImage"/>); the front picture is <see cref="MeshRequest.Image"/>.</summary>
public enum MeshView
{
    /// <summary>The subject's own left side: the camera turned a quarter round from the front toward the subject's left hand.</summary>
    Left = 1,

    /// <summary>The subject's back.</summary>
    Back = 2,

    /// <summary>The subject's own right side.</summary>
    Right = 3,
}

/// <summary>An extra picture of the subject, taken from one side (<see cref="MeshRequest.Views"/>). Like <see cref="MeshRequest.Image"/>, the subject is
/// framed alone, ideally with an alpha plane; the same pose and scale as the front picture lines it up with the mesh.</summary>
public sealed record MeshViewImage
{
    /// <summary>The side the picture shows.</summary>
    public required MeshView View { get; init; }

    /// <summary>The picture.</summary>
    public required ImageData Image { get; init; }
}
