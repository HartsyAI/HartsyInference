namespace HartsyInference.Audio.Models.ControlFoley;

/// <summary>Frame sampling rates and sizes of the three video streams (official <c>inference_utils</c> constants).</summary>
public sealed record ControlFoleyVideoOptions
{
    /// <summary>The released configuration: CLIP 384 px at 8 fps, CAV-MAE 224 px at 4 fps, Synchformer 224 px at 25 fps.</summary>
    public static ControlFoleyVideoOptions Default { get; } = new();

    /// <summary>CLIP frame edge (the whole frame is resized, not cropped).</summary>
    public int ClipSize { get; init; } = 384;

    /// <summary>CAV-MAE frame edge (short side resized, centre crop).</summary>
    public int VisualSize { get; init; } = 224;

    /// <summary>Synchformer frame edge (short side resized, centre crop).</summary>
    public int SyncSize { get; init; } = 224;

    /// <summary>CLIP frames per second.</summary>
    public double ClipFps { get; init; } = 8.0;

    /// <summary>CAV-MAE frames per second.</summary>
    public double VisualFps { get; init; } = 4.0;

    /// <summary>Synchformer frames per second.</summary>
    public double SyncFps { get; init; } = 25.0;
}
