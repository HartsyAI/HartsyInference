using HartsyInference.Audio.Cache;
using HartsyInference.Engine;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;

namespace HartsyInference.VoiceHost.Tests.Loopback;

/// <summary>What the loopback calls need on disk (Silero, the JFK clip, Whisper small.en, Kokoro af_heart) and the check that
/// the one CUDA device the test can see is the RTX 3060.</summary>
internal static class LoopbackAssets
{
    public const string RequiredDevice = "3060";

    public static string WakeRoot => Path.Combine(TestPaths.ModelsDir, "audio", "wake");

    public static string Jfk => Path.Combine(RepoRoot.Path, "tests", "python-reference", "silerovad_reference", "jfk.wav");

    public static string[] All()
    {
        string whisper = AudioModelCache.GetRepoDirectory("openai/whisper-small.en", "stt");
        string kokoro = AudioModelCache.GetRepoDirectory("hexgrad/Kokoro-82M", "tts");
        return
        [
            Path.Combine(WakeRoot, "vad", "silero_vad_16k.safetensors"), Jfk,
            Path.Combine(whisper, "model.safetensors"), Path.Combine(whisper, "added_tokens.json"),
            Path.Combine(kokoro, "config.json"), Path.Combine(kokoro, "voices", "af_heart.bin"),
            Path.Combine(RepoPaths.ModelsRoot(), "audio", "cmudict.dict"),
        ];
    }

    /// <summary>Rate of <see cref="JfkPhonePcm"/>: the G.711 line rate, so the softphone plays it without resampling.</summary>
    public const int PhoneRate = 8_000;

    /// <summary>The JFK clip as 8 kHz mono PCM16 bytes, what the softphone plays.</summary>
    public static byte[] JfkPhonePcm()
    {
        float[] samples = AudioClipCodec.DecodeMono(new AudioClip { Data = File.ReadAllBytes(Jfk), Format = "wav" }, PhoneRate);
        byte[] pcm = new byte[samples.Length * 2];
        for (int i = 0; i < samples.Length; i++)
        {
            short value = (short)Math.Clamp(samples[i] * 32767f, short.MinValue, short.MaxValue);
            BitConverter.TryWriteBytes(pcm.AsSpan(i * 2), value);
        }
        return pcm;
    }

    /// <summary>The engine device key of the RTX 3060 when it is the only CUDA device visible (run with
    /// <c>CUDA_VISIBLE_DEVICES=1</c>); null, after logging why, when CUDA is absent. Any other card fails the test before a
    /// model loads, because the 4090 is shared with SwarmUI.</summary>
    public static string? AudioDevice(Action<string> log)
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
        using InferenceEngine probe = new("cuda", 0);
        string name = probe.ComputeBackend.Capabilities.DeviceName;
        log($"audio device cuda:0 is {name}");
        if (!name.Contains(RequiredDevice, StringComparison.Ordinal))
        {
            throw new InvalidOperationException($"The loopback calls run on the RTX {RequiredDevice}, got '{name}'. Set CUDA_VISIBLE_DEVICES=1.");
        }
        return probe.ComputeBackend.Device.ToString();
    }
}
