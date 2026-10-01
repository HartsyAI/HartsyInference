using HartsyInference.Audio.Models.Denoise;
using HartsyInference.Engine;
using HartsyInference.Tests.Common;

namespace HartsyInference.Voice.Tests.Fakes;

/// <summary>Opens the audio engine for a GPU test on the RTX 3060 and loads the voice models on it. Run with
/// <c>CUDA_VISIBLE_DEVICES=1</c>: CUDA then shows only the 3060, as engine ordinal 0. Any other card fails the test,
/// because the 4090 is shared with SwarmUI and must not be used without an explicit grant.</summary>
internal static class GpuVoiceRig
{
    public const string RequiredDevice = "3060";

    /// <summary>The engine on the only visible CUDA device, after checking it is the 3060; null (skip) when CUDA is absent.</summary>
    public static InferenceEngine? OpenAudioEngine(Action<string> log, int ordinal = 0, EngineOptions? options = null)
    {
        string? unavailable = BackendGate.UnavailableReason("cuda");
        if (unavailable is not null)
        {
            if (Environment.GetEnvironmentVariable(BackendGate.RequireEnvVar) == "1")
            {
                throw new InvalidOperationException($"{BackendGate.RequireEnvVar}=1 but CUDA is unavailable: {unavailable}");
            }
            log($"SKIPPED: CUDA unavailable — {unavailable}");
            return null;
        }
        InferenceEngine engine = new("cuda", ordinal, options);
        string name = engine.ComputeBackend.Capabilities.DeviceName;
        log($"audio engine on {engine.ComputeBackend.Device}: {name}");
        if (!name.Contains(RequiredDevice, StringComparison.Ordinal))
        {
            engine.Dispose();
            throw new InvalidOperationException($"The voice GPU tests run on the RTX {RequiredDevice}, got '{name}'. Set CUDA_VISIBLE_DEVICES=1.");
        }
        return engine;
    }

    /// <summary>Options for a session on <paramref name="engine"/> with Whisper small.en and Kokoro, 16 kHz out.</summary>
    public static VoiceAgentOptions Options(InferenceEngine engine, string llmDevice = "cpu") => new()
    {
        AudioDevice = engine.ComputeBackend.Device.ToString(),
        LlmDevice = llmDevice,
        OutboundSampleRate = 16_000,
    };

    /// <summary>Every asset a session on Whisper small.en and Kokoro needs, including RNNoise's F32 weights and int8
    /// tables: <see cref="VoiceAgentOptions.Denoise"/> defaults on, so a session built from <see cref="Options"/>
    /// loads the denoiser at <see cref="RnnoisePrecision.Int8"/>.</summary>
    public static string[] Assets() =>
        [VoiceAssets.SileroWeights, VoiceAssets.RnnoiseWeights, VoiceAssets.RnnoiseInt8Tables, VoiceAssets.Jfk,
            .. VoiceAssets.WhisperFiles("openai/whisper-small.en"), .. VoiceAssets.KokoroFiles()];
}
