namespace HartsyInference.Audio.Models.Haru;

/// <summary>One 133 ms frame of the model: its audio (3200 samples at 24 kHz, clamped to plus or minus one), the text token it
/// emitted and the text that token adds.</summary>
internal sealed record HaruFrame(float[] Audio, int Token, string Text, int Index, double Seconds, bool Done);
