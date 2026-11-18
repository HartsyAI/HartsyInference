namespace HartsyInference.Audio.Models.Whisper;

/// <summary>One decoder cross-attention head whose weights track which audio frame the decoder is reading.</summary>
public readonly record struct WhisperAlignmentHead(int Layer, int Head);
