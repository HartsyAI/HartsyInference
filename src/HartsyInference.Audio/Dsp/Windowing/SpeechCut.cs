namespace HartsyInference.Audio.Dsp.Windowing;

/// <summary>A chosen cut inside a band: the sample offset within the band, and whether no quiet run was found.</summary>
public readonly record struct SpeechCut(int Offset, bool HardCut);
