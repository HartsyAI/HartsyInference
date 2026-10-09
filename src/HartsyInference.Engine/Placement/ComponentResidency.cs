namespace HartsyInference.Engine.Placement;

/// <summary>One component's placement: its mode, the device or storage it runs from, and its stored bytes.</summary>
public sealed record ComponentResidency(ResidencyComponent Component, ResidencyMode Mode, string Device, long Bytes);
