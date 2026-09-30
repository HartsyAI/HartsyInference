using HartsyInference.Audio.Frontends;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Logging;

namespace HartsyInference.Engine.Audio;

/// <summary>Piper (VITS) — CPU TTS at 22.05 kHz. Self-contained: the engine pipeline bundles the pure-C# espeak-ng phonemizer and reads each voice's phoneme id map and espeak language from the Piper <c>.onnx.json</c>. Not zero-shot — no voice reference required.
///
/// <para>Streaming is a sentence at a time through <see cref="SentenceChunkedSynthesis"/>: VITS turns a whole
/// phoneme sequence into a whole utterance, so the sentence is the unit, and the listener waits for one sentence's
/// synthesis instead of the passage's. Each sentence gets its own prosody and its own leading and trailing silence,
/// so the streamed result is not sample-identical to the whole-text <c>Synthesize</c>, which is unchanged.</para></summary>
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
            return new StreamingTtsRunner(pipeline.SampleRate,
                (backend, job) => pipeline.SynthesizeText(backend, job.Text, seed: job.Seed),
                // Synthesis is a long synchronous burn across every core; Task.Run keeps it off the consumer's thread.
                (backend, job, ct) => SentenceChunkedSynthesis.StreamBySentence(job.Text, pipeline.SampleRate,
                    (sentence, _) => pipeline.SynthesizeText(backend, sentence, seed: job.Seed),
                    static (work, token) => Task.Run(work, token),
                    SentenceSplitter.MinSentenceLength, SentenceChunkedSynthesis.NoClauseLimit, ct),
                pipeline);
        },
    };
}
