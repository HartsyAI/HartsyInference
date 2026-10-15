namespace HartsyInference.Audio.Pipelines;

/// <summary>Receives one generation stage as flat row-major float32 values, for parity tests against upstream
/// VibeVoice dumps. Production passes no trace, so the generation loop does no extra work.</summary>
internal delegate void VibeVoiceStageTrace(string stage, ReadOnlySpan<float> values);
