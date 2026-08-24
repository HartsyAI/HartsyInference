namespace HartsyInference.Core.MemoryManagement;

/// <summary>Tri-state for one VRAM lever. <see cref="Auto"/> defers to the tier's preset; the other two pin it regardless of tier, which is what makes every combination independently selectable.</summary>
public enum LeverState
{
    /// <summary>Follow whatever the resolved <see cref="VramTier"/> chose for this lever.</summary>
    Auto = 0,

    /// <summary>Force the lever on even where the tier would leave it off.</summary>
    On = 1,

    /// <summary>Force the lever off even where the tier would turn it on.</summary>
    Off = 2,
}
