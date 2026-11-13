namespace HartsyInference.ThreeD.Pipelines;

/// <summary>The stage names the staged shape pipelines (TRELLIS.2, Pixal3D) report in <c>GenerationProgress.Stage</c>, shared with the consumer that weighs them so a misspelling is a compile error and not a silent fallback.</summary>
public static class MeshStageNames
{
    /// <summary>TRELLIS.2: the DINOv3 image conditioning.</summary>
    public const string ImageConditioning = "Image conditioning";

    /// <summary>Pixal3D: the conditioning of the structure flow.</summary>
    public const string StructureConditioning = "Structure conditioning";

    /// <summary>The sparse-structure flow.</summary>
    public const string StructureFlow = "Structure flow";

    /// <summary>Pixal3D: the conditioning of the coarse shape flow.</summary>
    public const string CoarseConditioning = "Coarse conditioning";

    /// <summary>Pixal3D: the coarse (512) shape flow.</summary>
    public const string CoarseShapeFlow = "Coarse shape flow";

    /// <summary>TRELLIS.2: the 512 shape flow.</summary>
    public const string ShapeFlow512 = "Shape flow 512";

    /// <summary>The upsampling of the coarse voxels to the fine grid.</summary>
    public const string Upsample = "Upsample";

    /// <summary>Pixal3D: the conditioning of the fine shape flow.</summary>
    public const string FineConditioning = "Fine conditioning";

    /// <summary>Pixal3D: the fine (1024 or 1536) shape flow.</summary>
    public const string FineShapeFlow = "Fine shape flow";

    /// <summary>TRELLIS.2: the 1024 shape flow of the cascade.</summary>
    public const string ShapeFlow1024 = "Shape flow 1024";

    /// <summary>The shape VAE decode.</summary>
    public const string ShapeDecode = "Shape decode";

    /// <summary>The dual-grid meshing (and, shape-only, the mesh cleanup).</summary>
    public const string MeshExtraction = "Mesh extraction";

    /// <summary>Pixal3D: the conditioning of the texture flow.</summary>
    public const string TextureConditioning = "Texture conditioning";

    /// <summary>The texture flow.</summary>
    public const string TextureFlow = "Texture flow";

    /// <summary>The texture decode.</summary>
    public const string TextureDecode = "Texture decode";

    /// <summary>The CPU bake of the textured mesh.</summary>
    public const string TextureBake = "Texture bake";
}
