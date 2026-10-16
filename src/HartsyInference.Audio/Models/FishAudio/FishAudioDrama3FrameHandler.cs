namespace HartsyInference.Audio.Models.FishAudio;

/// <summary>Receives one generated Drama 3 frame's codes (all N codebooks) as it is drawn; the span is only valid during the call.</summary>
public delegate void FishAudioDrama3FrameHandler(ReadOnlySpan<int> codes);
