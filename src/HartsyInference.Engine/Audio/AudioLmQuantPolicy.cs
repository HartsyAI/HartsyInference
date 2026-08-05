using HartsyInference.Core.Logging;

namespace HartsyInference.Engine.Audio;

/// <summary>Resolves the audio-LM quant policy: <c>HARTSY_AUDIO_LM_QUANT</c> (<c>q4k</c>|<c>q8</c>|<c>off</c>)
/// wins when set; otherwise Q4_K single-device (today's fit behavior, byte-identical) and Off when a
/// layer-split placement pools VRAM — the point of sharding is not having to quantize.</summary>
internal static class AudioLmQuantPolicy
{
    internal const string EnvVar = "HARTSY_AUDIO_LM_QUANT";

    internal static AudioLmQuant Resolve(bool sharded)
    {
        string? env = Environment.GetEnvironmentVariable(EnvVar)?.Trim().ToLowerInvariant();
        if (string.IsNullOrEmpty(env))
        {
            return sharded ? AudioLmQuant.Off : AudioLmQuant.Q4K;
        }
        switch (env)
        {
            case "q4" or "q4k" or "q4_k":
                return AudioLmQuant.Q4K;
            case "q8" or "q8_0":
                return AudioLmQuant.Q8;
            case "off" or "none" or "f16" or "bf16":
                return AudioLmQuant.Off;
            default:
                Logs.Warning($"[Audio] Unrecognized {EnvVar}='{env}' — expected q4k|q8|off; using the default "
                    + $"({(sharded ? "off (layer-split pools VRAM)" : "q4k (single-GPU fit)")}).");
                return sharded ? AudioLmQuant.Off : AudioLmQuant.Q4K;
        }
    }
}
