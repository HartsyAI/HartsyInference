using System.Runtime.CompilerServices;
using HartsyInference.Audio.Frontends;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;
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
            return new StreamingTtsRunner(pipeline.SampleRate,
                (backend, job) => pipeline.SynthesizeText(backend, job.Text, seed: job.Seed),
                (backend, job, ct) => StreamBySentence(pipeline, backend, job, ct),
                pipeline);
        },
    };

    /// <summary>Synthesizes one sentence at a time, so the first can be heard while the rest is still being made.
    ///
    /// <para>Piper has no incremental decode loop — VITS turns a whole phoneme sequence into a whole utterance —
    /// so this is not the per-frame streaming a diffusion or LM-based voice can do. The unit is the sentence,
    /// and the shape of the win is the same: the listener waits for one sentence's synthesis instead of the
    /// whole passage's. On a reply that takes five seconds to synthesize entire, the first sentence is out in
    /// well under one.</para>
    ///
    /// <para>The result is not sample-identical to the whole-text call. Each sentence gets its own prosody
    /// contour and its own leading and trailing silence, which is the price of the split and the reason
    /// <see cref="SentenceSplitter"/> refuses to cut anywhere it is not sure. The non-streaming
    /// <c>Synthesize</c> path still passes the text through in one piece, unchanged.</para></summary>
    private static async IAsyncEnumerable<AudioChunk> StreamBySentence(PiperPipeline pipeline, IBackend backend,
        TtsJob job, [EnumeratorCancellation] CancellationToken cancel)
    {
        IReadOnlyList<string> sentences = SentenceSplitter.Split(job.Text);
        if (sentences.Count == 0)
        {
            yield break;
        }
        long offset = 0;
        for (int i = 0; i < sentences.Count; i++)
        {
            cancel.ThrowIfCancellationRequested();
            long started = Environment.TickCount64;
            // Synthesis is a long synchronous burn across every core. Running it inline would block this
            // async iterator's thread for its whole duration, including the consumer's continuation.
            string sentence = sentences[i];
            float[] samples = await Task.Run(() => pipeline.SynthesizeText(backend, sentence, seed: job.Seed), cancel)
                .ConfigureAwait(false);
            if (samples is null || samples.Length == 0)
            {
                continue;
            }
            if (i == 0)
            {
                Logs.Verbose($"[Audio][Piper] First of {sentences.Count} sentence(s) in {Environment.TickCount64 - started}ms.");
            }
            yield return new AudioChunk(samples, pipeline.SampleRate, Channels: 1, StartSampleOffset: offset);
            offset += samples.Length;
        }
    }
}
