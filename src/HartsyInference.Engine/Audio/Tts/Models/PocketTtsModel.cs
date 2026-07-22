using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Logging;

namespace HartsyInference.Engine.Audio;

/// <summary>Kyutai Pocket-TTS — a continuous-latent flow-LM (Helium-style backbone at 12.5 Hz → Mimi → 24 kHz). A
/// voice is REQUIRED: the model conditions on a pre-primed speaker KV state and ends into near-silence without one;
/// predefined English voices (default <c>alba</c>) ship as per-layer KV-cache safetensors. Arbitrary voice cloning
/// lives behind separate weights and is not wired.</summary>
internal static class PocketTtsModel
{
    private const string Repo = "kyutai/pocket-tts-without-voice-cloning";
    private const string Revision = "d29db7978e464fb90cb3359ee0c69a273b9142cc";
    private const string Language = "english";
    private const string WeightsFile = "languages/english/model.safetensors";
    private const string SpmFile = "tokenizer.model";
    private const string DefaultVoice = "alba";

    /// <summary>The Pocket-TTS descriptor.</summary>
    internal static TtsModelDescriptor Descriptor { get; } = new TtsModelDescriptor
    {
        ResolveRepo = _ => Repo,
        LoadAsync = async (_, cancel) =>
        {
            string weights = await AudioModelCache.GetAsync(Repo, WeightsFile, Revision, ct: cancel).ConfigureAwait(false);
            string spm = await AudioModelCache.GetAsync(Repo, SpmFile, Revision, ct: cancel).ConfigureAwait(false);
            PocketTtsPipeline pipeline = PocketTtsPipeline.LoadFromCheckpoint(weights, spm);
            await EnsureVoiceAsync(pipeline, DefaultVoice, cancel).ConfigureAwait(false);
            Logs.Info("[Audio][Pocket-TTS] Loaded kyutai/pocket-tts (continuous-latent flow-LM, 24 kHz, English).");

            object voiceLock = new();
            return new TtsRunner(pipeline.SampleRate, (backend, job) =>
            {
                string voice = string.IsNullOrEmpty(job.Voice) ? DefaultVoice : job.Voice.ToLowerInvariant();
                if (!pipeline.HasVoice(voice))
                {
                    lock (voiceLock)
                    {
                        if (!pipeline.HasVoice(voice))
                        {
                            EnsureVoiceAsync(pipeline, voice, CancellationToken.None).GetAwaiter().GetResult();
                        }
                    }
                }
                return pipeline.Synthesize(backend, job.Text, voice, job.Seed);
            }, pipeline);
        },
    };

    /// <summary>Downloads a predefined English voice's KV-state safetensors and registers it with the pipeline.</summary>
    private static async Task EnsureVoiceAsync(PocketTtsPipeline pipeline, string voiceName, CancellationToken cancel)
    {
        if (pipeline.HasVoice(voiceName))
        {
            return;
        }
        string path = await AudioModelCache.GetAsync(Repo, $"languages/{Language}/embeddings/{voiceName}.safetensors", Revision, ct: cancel).ConfigureAwait(false);
        pipeline.RegisterVoiceFromFile(voiceName, path);
    }
}
