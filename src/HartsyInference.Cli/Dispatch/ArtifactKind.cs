namespace HartsyInference.Cli.Dispatch;

/// <summary>The output form a modality produces, used to decide how an artifact is written and previewed.</summary>
public enum ArtifactKind
{
    /// <summary>Plain generated text (LLM output).</summary>
    Text,

    /// <summary>An encoded raster image (PNG/BMP bytes).</summary>
    Image,

    /// <summary>Encoded audio (WAV bytes).</summary>
    Audio,

    /// <summary>Encoded video (MP4/GIF bytes).</summary>
    Video,

    /// <summary>A 3D mesh (glTF/GLB/OBJ bytes).</summary>
    Mesh,

    /// <summary>Structured data (JSON), e.g. embeddings or detections.</summary>
    Data,
}
