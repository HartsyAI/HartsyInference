using HartsyInference.Voice;

namespace HartsyInference.VoiceHost.Config;

/// <summary>Models and devices (<c>models</c> section); the defaults are <see cref="VoiceAgentOptions"/>'s.</summary>
public sealed record HostModelsConfig
{
    private static readonly VoiceAgentOptions _defaults = new();

    /// <summary>Language model: a catalog id, local path or repository id.</summary>
    public string LlmModel { get; set; } = _defaults.LlmModel;

    /// <summary>Device every language-model request names; the engine itself is built on <see cref="AudioDevice"/>.</summary>
    public string LlmDevice { get; set; } = _defaults.LlmDevice;

    /// <summary>Device the engine is built on, where speech recognition and synthesis run.</summary>
    public string AudioDevice { get; set; } = _defaults.AudioDevice;

    public string SttModel { get; set; } = _defaults.SttModel;

    public string TtsModel { get; set; } = _defaults.TtsModel;

    /// <summary>Run RNNoise ahead of the VAD; start-up fails when its weights are missing.</summary>
    public bool Denoise { get; set; } = _defaults.Denoise;

    /// <summary>Folder holding the <c>vad</c> and <c>denoise</c> weights; null means the models root's <c>audio/wake</c>.</summary>
    public string? WakeModelRoot { get; set; }
}
