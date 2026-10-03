using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;

namespace HartsyInference.Engine.Audio;

/// <summary>IndexTTS-1.5 (<c>IndexTeam/IndexTTS-1.5</c>): zero-shot voice cloning via a GPT-2 text-to-speech
/// decoder with Conformer-Perceiver speech conditioning, vocoded by a custom 24 kHz BigVGAN-v2 that reads the
/// GPT's own latent hidden states directly (no intermediate codec decode — the shipped <c>dvae.pth</c> is
/// training-only, confirmed unused by the reference <c>infer()</c>). No emotion or duration control — that is
/// IndexTTS-2, a later addition.</summary>
internal static class IndexTtsModel
{
    internal const string Repo = "IndexTeam/IndexTTS-1.5";
    internal const string ConfigFile = "config.yaml";
    internal const string TokenizerFile = "bpe.model";
    internal const string GptFile = "gpt.pth";
    internal const string BigVganFile = "bigvgan_generator.pth";

    internal static TtsModelDescriptor Descriptor { get; } = new TtsModelDescriptor
    {
        ResolveRepo = _ => Repo,
        ResolveFiles = (_, _) => Task.FromResult<IReadOnlyList<AudioModelFile>>(
        [
            new AudioModelFile(ConfigFile),
            new AudioModelFile(TokenizerFile),
            new AudioModelFile(GptFile),
            new AudioModelFile(BigVganFile),
        ]),
        LoadAsync = async (_, _, cancel) =>
        {
            // dvae.pth (243 MB) is confirmed unused at inference (see IndexTtsPipeline's remarks) and
            // deliberately not fetched -- no point costing every first-time install that download.
            IReadOnlyDictionary<string, string> fetched = await AudioModelCache.FetchAllAsync(Repo,
                [
                    new AudioModelFile(ConfigFile),
                    new AudioModelFile(TokenizerFile),
                    new AudioModelFile(BigVganFile),
                    new AudioModelFile(GptFile),
                ], "tts", ct: cancel).ConfigureAwait(false);

            IndexTtsPipeline pipeline = await IndexTtsPipeline.LoadAsync(
                fetched[TokenizerFile], fetched[GptFile], fetched[BigVganFile], cfg: null, cancel).ConfigureAwait(false);
            Logs.Info($"[Audio][IndexTTS] Loaded {Repo} (IndexTTS-1.5, zero-shot cloning, 24 kHz).");

            return new TtsRunner(24_000, (backend, job) => Synthesize(pipeline, backend, job), pipeline);
        },
    };

    private static float[] Synthesize(IndexTtsPipeline pipeline, IBackend backend, TtsJob job)
    {
        if (job.ReferenceMono24k is not { Length: > 0 })
        {
            throw new InvalidOperationException("IndexTTS-1.5 needs a voice reference clip (zero-shot cloning only; no instruction-based voice design).");
        }
        IndexTtsOptions options = new()
        {
            Seed = unchecked((ulong)job.Seed),
            Temperature = job.Temperature.HasValue ? (float)job.Temperature.Value : 0.8f,
            TopK = job.TopK ?? 30,
            MaxMelTokens = job.MaxTokens,
        };
        return pipeline.Synthesize(backend, job.Text, job.ReferenceMono24k, 24_000, options);
    }
}
