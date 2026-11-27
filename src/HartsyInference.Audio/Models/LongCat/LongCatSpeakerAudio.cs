namespace HartsyInference.Audio.Models.LongCat;

/// <summary>The prepared multi-speaker audio: one encoder input per track, all of one length (the two speakers, then a silent one when asked
/// for), and the soundtrack both speakers are heard in, at 16 kHz.</summary>
public sealed record LongCatSpeakerAudio(float[][] Tracks, float[] Soundtrack);
