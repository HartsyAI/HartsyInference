namespace HartsyInference.Engine.Audio.Wake.Wyoming;

/// <summary>A wake word firing, reported to Home Assistant as a <c>detection</c> event.</summary>
public readonly record struct WyomingWakeHit(string Name, float Score);
