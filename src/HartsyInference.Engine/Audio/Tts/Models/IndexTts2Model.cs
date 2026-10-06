using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.IndexTts2;
using System.Runtime.CompilerServices;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Logging;

namespace HartsyInference.Engine.Audio;

/// <summary>IndexTTS-2.5 (<c>indextts2</c>, default) and IndexTTS-2.0 (<c>indextts2:2.0</c>): emotion-controllable
/// zero-shot voice cloning — a GPT text-to-speech decoder generating semantic-codec codes, an S2Mel flow-matching
/// DiT converting them to an 80-band mel, vocoded by the stock <c>nvidia/bigvgan_v2_22khz_80band_256x</c> to
/// 22050 Hz. The two checkpoints share that stack; they differ in the GPT's speaker conditioning (2.5: CAM++
/// projection, 2.0: Conformer+Perceiver), the semantic codec (2.5: the repo's own <c>codec.pth</c>, 2.0:
/// <c>amphion/MaskGCT</c>'s RepCodec) and the tokenizer (2.5: tiktoken, 2.0: SentencePiece). Each spans several
/// HuggingFace repos — <see cref="AudioModelFile.Repo"/> overrides resolve them, same pattern as
/// <see cref="AukModel"/>'s variant-aware multi-repo wiring.
/// <para>Emotion control follows the real <c>infer_generator</c>: by default the speaker's own clip doubles as the
/// emotion reference; <see cref="TtsJob.EmotionReference"/> supplies a separate clip (blended by
/// <see cref="TtsJob.EmotionAlpha"/>), <see cref="TtsJob.Emotion"/> an explicit 8-way vector in IndexTTS-2's own
/// order (happy, angry, sad, afraid, disgusted, melancholic, surprised, calm — NOT Zonos's order), and
/// <see cref="TtsJob.EmotionText"/>/<see cref="TtsJob.EmotionFromText"/> a free-text description classified by the
/// bundled QwenEmotion model (loaded on first use).</para></summary>
internal static class IndexTts2Model
{
    internal const string Repo25 = "IndexTeam/IndexTTS-2.5";
    internal const string Repo20 = "IndexTeam/IndexTTS-2";
    internal const string W2vBertRepo = "facebook/w2v-bert-2.0";
    internal const string CamplusRepo = "funasr/campplus";
    internal const string BigVganRepo = "nvidia/bigvgan_v2_22khz_80band_256x";
    internal const string MaskGctRepo = "amphion/MaskGCT";

    internal const string TiktokenFile = "multilingual_zh_ja_yue_char_del.tiktoken";
    internal const string BpeFile = "bpe.model";
    internal const string GptFile = "gpt.pth";
    internal const string S2MelFile = "s2mel.pth";
    internal const string CodecFile = "codec.pth";
    internal const string MaskGctCodecFile = "semantic_codec/model.safetensors";
    internal const string W2vStatsFile = "wav2vec2bert_stats.pt";
    internal const string Feat1File = "feat1.pt";
    internal const string Feat2File = "feat2.pt";
    internal const string W2vBertWeightsFile = "model.safetensors";
    internal const string CamplusFile = "campplus_cn_common.bin";
    internal const string BigVganFile = "bigvgan_generator.pt";
    internal const string QwenDir = "qwen0.6bemo4-merge";
    internal static readonly string[] QwenFiles = ["config.json", "model.safetensors", "tokenizer.json", "chat_template.jinja"];

    /// <summary>True for the 2.0 checkpoint (exactly <c>2.0</c>, <c>v2_0</c> or <c>IndexTeam/IndexTTS-2</c>); everything else, including no variant and any other custom repo id, is 2.5.</summary>
    internal static bool IsV2_0(string? variant)
    {
        string v = (variant ?? string.Empty).Trim();
        return v.Equals("2.0", StringComparison.OrdinalIgnoreCase)
            || v.Equals("v2_0", StringComparison.OrdinalIgnoreCase)
            || v.Equals(Repo20, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>Maps the variant hint to the repo holding the GPT/S2Mel checkpoint; a custom <c>owner/name</c> passes through.</summary>
    internal static string ResolveRepo(string? variant)
    {
        string id = (variant ?? string.Empty).Trim();
        if (id.Contains('/', StringComparison.Ordinal))
        {
            return id;
        }
        return IsV2_0(id) ? Repo20 : Repo25;
    }

    private static string QwenPath(string file) => $"{QwenDir}/{file}";

    internal static IReadOnlyList<AudioModelFile> Files(string? variant)
    {
        bool v20 = IsV2_0(variant);
        List<AudioModelFile> files =
        [
            new AudioModelFile(v20 ? BpeFile : TiktokenFile),
            new AudioModelFile(GptFile),
            new AudioModelFile(S2MelFile),
            new AudioModelFile(W2vStatsFile),
            new AudioModelFile(Feat1File),
            new AudioModelFile(Feat2File),
        ];
        files.Add(v20 ? new AudioModelFile(MaskGctCodecFile, Repo: MaskGctRepo) : new AudioModelFile(CodecFile));
        foreach (string qwen in QwenFiles)
        {
            // Optional: a missing classifier file disables free-text emotion rather than failing the whole load.
            files.Add(new AudioModelFile(QwenPath(qwen), Required: false));
        }
        files.Add(new AudioModelFile(W2vBertWeightsFile, Repo: W2vBertRepo));
        files.Add(new AudioModelFile(CamplusFile, Repo: CamplusRepo));
        files.Add(new AudioModelFile(BigVganFile, Repo: BigVganRepo));
        return files;
    }

    internal static TtsModelDescriptor Descriptor { get; } = new TtsModelDescriptor
    {
        ResolveRepo = ResolveRepo,
        ResolveFiles = (variant, _) => Task.FromResult(Files(variant)),
        LoadAsync = async (context, variant, cancel) =>
        {
            bool v20 = IsV2_0(variant);
            string repo = ResolveRepo(variant);
            IReadOnlyDictionary<string, string> fetched = await AudioModelCache.FetchAllAsync(repo, Files(variant), "tts", ct: cancel).ConfigureAwait(false);

            string[] qwenPaths = [.. QwenFiles.Select(f => QwenPath(f))];
            bool haveQwen = qwenPaths.All(fetched.ContainsKey);
            if (!haveQwen)
            {
                Logs.Info("[Audio][IndexTTS2] QwenEmotion files not all available; free-text emotion (--emotion-text) is disabled.");
            }

            IndexTts2Pipeline pipeline = await IndexTts2Pipeline.LoadAsync(
                fetched[v20 ? BpeFile : TiktokenFile], fetched[GptFile], fetched[S2MelFile],
                fetched[v20 ? MaskGctCodecFile : CodecFile],
                fetched[W2vBertWeightsFile], fetched[W2vStatsFile], fetched[CamplusFile], fetched[BigVganFile],
                feat1Path: fetched[Feat1File], feat2Path: fetched[Feat2File],
                qwenEmoDir: haveQwen ? Path.GetDirectoryName(fetched[qwenPaths[0]]) : null, qwenBackend: haveQwen ? context.Backend : null,
                cfg: v20 ? IndexTts2Config.V2_0 : IndexTts2Config.V2_5, cancel).ConfigureAwait(false);
            Logs.Info($"[Audio][IndexTTS2] Loaded {repo} (IndexTTS-{(v20 ? "2.0" : "2.5")}, emotion-controllable zero-shot cloning, 22050 Hz).");

            ReferenceCache references = new(pipeline);
            return new StreamingTtsRunner(22_050,
                (backend, job) => Synthesize(pipeline, references, backend, job),
                (backend, job, token) => Stream(pipeline, references, backend, job, token),
                references, pipeline);
        },
    };

    /// <summary>Maps the engine job to <see cref="IndexTts2Options"/>. The reference clips arrive at the engine's shared 24 kHz decode rate; the pipeline resamples internally.</summary>
    internal static IndexTts2Options BuildOptions(TtsJob job)
    {
        float[]? vector = null;
        if (job.Emotion is { Count: > 0 } emotion)
        {
            if (emotion.Count != 8)
            {
                throw new ArgumentException("IndexTTS-2 emotion needs exactly 8 values (happy, angry, sad, afraid, disgusted, melancholic, surprised, calm).");
            }
            foreach (double v in emotion)
            {
                if (!double.IsFinite(v) || v < 0d || v > 1.2d)
                {
                    throw new ArgumentException($"IndexTTS-2 emotion weights must be between 0 and 1.2; got {v}.");
                }
            }
            vector = [.. emotion.Select(v => (float)v)];
        }

        (float[] Audio, int SampleRate)? emoReference = null;
        if (job.EmotionReference is { Data.Length: > 0 } clip)
        {
            emoReference = (AudioClipCodec.DecodeMono(clip, ReferenceRate), ReferenceRate);
        }

        return new IndexTts2Options
        {
            Seed = unchecked((ulong)job.Seed),
            Temperature = job.Temperature.HasValue ? (float)job.Temperature.Value : 0.8f,
            TopK = job.TopK ?? 30,
            TopP = job.TopP.HasValue ? (float)job.TopP.Value : 0.8f,
            MaxMelTokens = job.MaxTokens,
            EmoVector = vector,
            EmoAudioReference = emoReference,
            EmoAlpha = job.EmotionAlpha.HasValue ? (float)job.EmotionAlpha.Value : 1.0f,
            UseEmoText = !string.IsNullOrWhiteSpace(job.EmotionText) || job.EmotionFromText,
            EmoText = string.IsNullOrWhiteSpace(job.EmotionText) ? null : job.EmotionText,
        };
    }

    private const int ReferenceRate = 24_000;

    /// <summary>While streaming, short neighbouring sentences are not merged until this many tokens have been
    /// consumed, so the first audible segment stays small (the reference's <c>quick_streaming_tokens</c>).</summary>
    private const int StreamQuickTokens = 40;

    private static float[] Synthesize(IndexTts2Pipeline pipeline, ReferenceCache references, HartsyInference.Core.Backends.IBackend backend, TtsJob job)
    {
        IndexTts2Reference reference = references.Get(backend, RequireReference(pipeline, job));
        return pipeline.Synthesize(backend, job.Text, reference, BuildOptions(job));
    }

    private static float[] RequireReference(IndexTts2Pipeline pipeline, TtsJob job)
    {
        if (job.ReferenceMono24k is not { Length: > 0 } clip)
        {
            throw new InvalidOperationException($"{pipeline.ModelName} needs a voice reference clip (zero-shot cloning only).");
        }
        return clip;
    }

    /// <summary>Streams one text segment at a time: the first chunk is audible as soon as the first segment is generated.</summary>
    private static async IAsyncEnumerable<AudioChunk> Stream(IndexTts2Pipeline pipeline, ReferenceCache references,
        HartsyInference.Core.Backends.IBackend backend, TtsJob job, [EnumeratorCancellation] CancellationToken cancel)
    {
        float[] clip = RequireReference(pipeline, job);
        IndexTts2Options options = BuildOptions(job) with { QuickStreamingTokens = StreamQuickTokens };
        IndexTts2Reference reference = await Task.Run(() => references.Get(backend, clip), cancel).ConfigureAwait(false);
        using IEnumerator<float[]> chunks = pipeline.SynthesizeStream(backend, job.Text, reference, options, cancel).GetEnumerator();
        long offset = 0;
        while (true)
        {
            float[]? next = await Task.Run(() => chunks.MoveNext() ? chunks.Current : null, cancel).ConfigureAwait(false);
            if (next is null)
            {
                yield break;
            }
            yield return new AudioChunk(next, 22_050, 1, offset);
            offset += next.Length;
        }
    }

    /// <summary>Keeps the prepared reference conditioning of the most recent voice clip — the 24-layer w2v-bert pass over
    /// the clip is the costliest front-end step and depends on nothing else, so repeated generations with one voice (a
    /// batch, a conversation) pay it once, as the reference implementation's <c>cache_spk_cond</c> does.</summary>
    private sealed class ReferenceCache(IndexTts2Pipeline pipeline) : IDisposable
    {
        private readonly object _lock = new();
        private byte[]? _key;
        private IndexTts2Reference? _reference;

        public IndexTts2Reference Get(HartsyInference.Core.Backends.IBackend backend, float[] clip)
        {
            byte[] key = System.Security.Cryptography.SHA256.HashData(System.Runtime.InteropServices.MemoryMarshal.AsBytes<float>(clip));
            lock (_lock)
            {
                if (_reference is not null && _key is not null && key.AsSpan().SequenceEqual(_key))
                {
                    return _reference;
                }
                _reference?.Dispose();
                _reference = null;
                _reference = pipeline.PrepareReference(backend, clip, ReferenceRate);
                _key = key;
                return _reference;
            }
        }

        public void Dispose()
        {
            lock (_lock)
            {
                _reference?.Dispose();
                _reference = null;
            }
        }
    }
}
