namespace HartsyInference.Audio.Dsp.Windowing;

/// <summary>One planned window of a clip: its sample range, whether it is silent, and whether it ends on a hard cut.</summary>
public readonly record struct SpeechWindow(int Start, int Length, bool Silent, bool HardCut);
