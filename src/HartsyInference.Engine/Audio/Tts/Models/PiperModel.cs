using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Logging;

namespace HartsyInference.Engine.Audio;

/// <summary>Piper (VITS) — CPU TTS at 22.05 kHz. Self-contained: the engine pipeline bundles the pure-C# espeak-ng phonemizer and reads each voice's phoneme id map and espeak language from the Piper <c>.onnx.json</c>. Not zero-shot — no voice reference required.</summary>
internal static class PiperModel
{
    /// <summary>Default English voice; its <c>.onnx</c> and <c>.onnx.json</c> auto-download on first use.</summary>
    private const string DefaultVoice = "en_US-lessac-medium";

    internal static TtsModelDescriptor Descriptor { get; } = new TtsModelDescriptor
    {
        ResolveRepo = _ => "rhasspy/piper-voices",
        VoiceSelectsWeights = true,
        LoadAsync = async (_, variant, cancel) =>
        {
            // The variant IS the voice here (VoiceSelectsWeights) — each voice is its own .onnx download.
            string voice = string.IsNullOrWhiteSpace(variant) || variant.Equals("default", StringComparison.OrdinalIgnoreCase)
                ? DefaultVoice : variant;
            PiperPipeline pipeline = await PiperPipeline.LoadAsync(voice, ct: cancel).ConfigureAwait(false);
            Logs.Info($"[Audio][Piper] Loaded rhasspy/piper-voices {voice} (VITS 22.05 kHz).");
            return new TtsRunner(pipeline.SampleRate,
                (backend, job) => pipeline.SynthesizeText(backend, job.Text, seed: job.Seed), pipeline);
        },
    };
}
