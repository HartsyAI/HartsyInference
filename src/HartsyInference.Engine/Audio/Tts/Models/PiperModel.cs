using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Logging;

namespace HartsyInference.Engine.Audio;

/// <summary>Piper (VITS) — CPU TTS at 22.05 kHz. Self-contained: the engine pipeline bundles the pure-C# espeak-ng
/// phonemizer and reads each voice's phoneme id map and espeak language from the Piper <c>.onnx.json</c>. Not
/// zero-shot — no voice reference required.</summary>
internal static class PiperModel
{
    /// <summary>Default English voice; its <c>.onnx</c> and <c>.onnx.json</c> auto-download on first use.</summary>
    private const string DefaultVoice = "en_US-lessac-medium";

    /// <summary>The Piper descriptor.</summary>
    internal static TtsModelDescriptor Descriptor { get; } = new TtsModelDescriptor
    {
        ResolveRepo = _ => "rhasspy/piper-voices",
        LoadAsync = async (_, _, cancel) =>
        {
            PiperPipeline pipeline = await PiperPipeline.LoadAsync(DefaultVoice, ct: cancel).ConfigureAwait(false);
            Logs.Info($"[Audio][Piper] Loaded rhasspy/piper-voices {DefaultVoice} (VITS 22.05 kHz).");
            return new TtsRunner(pipeline.SampleRate,
                (backend, job) => pipeline.SynthesizeText(backend, job.Text, seed: job.Seed),
                pipeline);
        },
    };
}
