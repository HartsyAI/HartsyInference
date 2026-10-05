using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Logging;

namespace HartsyInference.Engine.Audio;

/// <summary>IndexTTS-2.5 (<c>IndexTeam/IndexTTS-2.5</c>): emotion-controllable zero-shot voice cloning —
/// a GPT text-to-speech decoder (CAM++ speaker conditioning) generating semantic-codec codes, an S2Mel
/// flow-matching DiT converting them to an 80-band mel, vocoded by the stock
/// <c>nvidia/bigvgan_v2_22khz_80band_256x</c> to 22050 Hz. Spans 4 HuggingFace repos: the main checkpoint,
/// <c>facebook/w2v-bert-2.0</c> (semantic conditioning), <c>funasr/campplus</c> (speaker style), and the
/// BigVGAN vocoder — <see cref="AudioModelFile.Repo"/> overrides resolve each, same pattern as
/// <see cref="AukModel"/>'s multi-repo wiring.
/// <para>This wiring covers zero-shot cloning with the DEFAULT emotion (the speaker's own natural emotion,
/// matching the real reference's own fallback when no emotion control is requested) — the explicit
/// emotion-vector, emotion-reference-clip and QwenEmotion-text modes <see cref="IndexTts2Options"/> already
/// supports are not yet surfaced as job/CLI/HTTP knobs (needs new <see cref="TtsJob"/> fields and
/// <c>SpeechCommand</c>/<c>AudioEndpoints</c> plumbing — deliberately out of scope here to keep this change
/// focused on getting the model loadable and callable at all). IndexTTS-2.0 is not wired — see
/// <see cref="IndexTts2Config"/>'s remarks.</para></summary>
internal static class IndexTts2Model
{
    internal const string Repo = "IndexTeam/IndexTTS-2.5";
    internal const string W2vBertRepo = "facebook/w2v-bert-2.0";
    internal const string CamplusRepo = "funasr/campplus";
    internal const string BigVganRepo = "nvidia/bigvgan_v2_22khz_80band_256x";

    internal const string TiktokenFile = "multilingual_zh_ja_yue_char_del.tiktoken";
    internal const string GptFile = "gpt.pth";
    internal const string S2MelFile = "s2mel.pth";
    internal const string CodecFile = "codec.pth";
    internal const string W2vStatsFile = "wav2vec2bert_stats.pt";
    internal const string W2vBertWeightsFile = "model.safetensors";
    internal const string CamplusFile = "campplus_cn_common.bin";
    internal const string BigVganFile = "bigvgan_generator.pt";

    private static IReadOnlyList<AudioModelFile> Files() =>
    [
        new AudioModelFile(TiktokenFile),
        new AudioModelFile(GptFile),
        new AudioModelFile(S2MelFile),
        new AudioModelFile(CodecFile),
        new AudioModelFile(W2vStatsFile),
        new AudioModelFile(W2vBertWeightsFile, Repo: W2vBertRepo),
        new AudioModelFile(CamplusFile, Repo: CamplusRepo),
        new AudioModelFile(BigVganFile, Repo: BigVganRepo),
    ];

    internal static TtsModelDescriptor Descriptor { get; } = new TtsModelDescriptor
    {
        ResolveRepo = _ => Repo,
        ResolveFiles = (_, _) => Task.FromResult(Files()),
        LoadAsync = async (context, _, cancel) =>
        {
            IReadOnlyDictionary<string, string> fetched = await AudioModelCache.FetchAllAsync(Repo, Files(), "tts", ct: cancel).ConfigureAwait(false);

            IndexTts2Pipeline pipeline = await IndexTts2Pipeline.LoadAsync(
                fetched[TiktokenFile], fetched[GptFile], fetched[S2MelFile], fetched[CodecFile],
                fetched[W2vBertWeightsFile], fetched[W2vStatsFile], fetched[CamplusFile], fetched[BigVganFile],
                feat1Path: null, feat2Path: null, qwenEmoDir: null, qwenBackend: null,
                cfg: null, cancel).ConfigureAwait(false);
            Logs.Info($"[Audio][IndexTTS2] Loaded {Repo} (IndexTTS-2.5, emotion-controllable zero-shot cloning, 22050 Hz).");

            return new TtsRunner(22_050, (backend, job) => Synthesize(pipeline, backend, job), pipeline);
        },
    };

    private static float[] Synthesize(IndexTts2Pipeline pipeline, HartsyInference.Core.Backends.IBackend backend, TtsJob job)
    {
        if (job.ReferenceMono24k is not { Length: > 0 })
        {
            throw new InvalidOperationException("IndexTTS-2.5 needs a voice reference clip (zero-shot cloning only).");
        }
        IndexTts2Options options = new()
        {
            Seed = unchecked((ulong)job.Seed),
            Temperature = job.Temperature.HasValue ? (float)job.Temperature.Value : 0.8f,
            TopK = job.TopK ?? 30,
            TopP = job.TopP.HasValue ? (float)job.TopP.Value : 0.8f,
            MaxMelTokens = job.MaxTokens,
        };
        // job.ReferenceMono24k is 24 kHz (the engine's shared reference-decode rate); Synthesize resamples
        // internally to whatever each of its own front ends needs (16k/22050), same as every other model
        // that takes a reference clip at a different native rate than the engine's shared decode.
        return pipeline.Synthesize(backend, job.Text, job.ReferenceMono24k, 24_000, options);
    }
}
