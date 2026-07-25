using HartsyInference.Audio.Frontends;
using HartsyInference.Audio.Models.Csm;
using HartsyInference.Audio.Models.HeartMula;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Audio;

/// <summary>HeartMuLa-oss-3B (Apache 2.0) — a CSM-shaped music LM (Llama-3B global + Llama-300M depth over an
/// 8-codebook grid) whose flow-matching HeartCodec decodes to 48 kHz. The LM and codec both auto-download; the
/// <c>-q8</c>/<c>-q4</c> variant suffixes select a disk-cached quantized weight set.
///
/// <para>Runtime-pending: the engine's HeartCodec and MuQ conditioning are wired but not yet validated against the
/// real checkpoints (the LM reuses the verified CSM stack), and MuQ style conditioning is omitted.</para></summary>
internal static class HeartMulaMusicModel
{
    private const string BaseRepo = "HeartMuLa/HeartMuLa-oss-3B";
    private const string HappyNewYearRepo = "HeartMuLa/HeartMuLa-oss-3B-happy-new-year";
    // The dated RL repo is public; the undated one is gated (401) — resolve to the dated one.
    private const string RlRepo = "HeartMuLa/HeartMuLa-RL-oss-3B-20260123";
    private const string CodecRepo = "HeartMuLa/HeartCodec-oss-20260123";

    /// <summary>The HeartMuLa descriptor.</summary>
    internal static MusicModelDescriptor Descriptor { get; } = new MusicModelDescriptor
    {
        ManagesOwnWeights = true,
        // The quant is part of the cache key so bf16/Q8/Q4 variants of the same repo cache as SEPARATE runners.
        CacheKey = selector => $"{ResolveRepo(selector.Variant)}|{ResolveQuant(selector.Variant) ?? "bf16"}",
        LoadAsync = (_, selector, cancel) => LoadAsync(ResolveRepo(selector.Variant), ResolveQuant(selector.Variant), cancel),
    };

    private static string ResolveRepo(string variant)
    {
        string id = (variant ?? string.Empty).Trim();
        if (id.Contains('/', StringComparison.Ordinal))
        {
            return id;
        }
        // The precision suffix does not affect repo selection — these contains-checks ignore it.
        if (id.Contains("rl", StringComparison.OrdinalIgnoreCase))
        {
            return RlRepo;
        }
        if (id.Contains("hny", StringComparison.OrdinalIgnoreCase) || id.Contains("happy", StringComparison.OrdinalIgnoreCase))
        {
            return HappyNewYearRepo;
        }
        return BaseRepo;
    }

    /// <summary>Maps a model id's precision suffix to a quant mode: <c>-q8</c> → q8_0, <c>-q4</c> → q4_k, else null
    /// (full bf16). The conversion runs once on first generation into a disk cache.</summary>
    private static string? ResolveQuant(string variant)
    {
        string lower = (variant ?? string.Empty).Trim().ToLowerInvariant();
        if (lower.EndsWith("-q8", StringComparison.Ordinal) || lower.EndsWith("-q8_0", StringComparison.Ordinal))
        {
            return "q8_0";
        }
        return lower.EndsWith("-q4", StringComparison.Ordinal) || lower.EndsWith("-q4_k", StringComparison.Ordinal) ? "q4_k" : null;
    }

    private static async Task<IMusicRunner> LoadAsync(string repo, string? quant, CancellationToken cancel)
    {
        (IReadOnlyDictionary<string, Tensor> lmWeights, IDisposable[] lmLoaders) = await AudioCheckpoints.LoadAsync(repo, cancel).ConfigureAwait(false);
        (IReadOnlyDictionary<string, Tensor> codecWeights, IDisposable[] codecLoaders) = await AudioCheckpoints.LoadAsync(CodecRepo, cancel).ConfigureAwait(false);

        // The AR decode is memory-bandwidth-bound (per-token weight streaming), so the Q8/Q4 variants are faster AND
        // smaller. The pipeline quantizes the REMAPPED weights once to a disk GGUF cache and loads from there after.
        bool useQuant = CsmWeightCache.ResolveQuant(quant) is not null;

        HeartMulaConfig config = HeartMulaConfig.Oss3B;
        HeartMulaPipeline pipeline = new HeartMulaPipeline(config);
        if (useQuant)
        {
            string quantCache = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                ".cache", "hartsyinference", "heartmula", $"{repo.Replace('/', '_')}-{quant!.ToLowerInvariant()}.gguf");
            pipeline.LoadWeights(lmWeights, quant, quantCache);   // raw → remap → quantize (post-remap), disk-cached
            // The GGUF cache owns the weights now — drop the source safetensors mmaps.
            foreach (IDisposable loader in lmLoaders)
            {
                loader.Dispose();
            }
            lmLoaders = [];
        }
        else
        {
            // Upstream inference dtypes: LM bf16 (halves the per-frame weight streaming too), codec f32.
            // Every entry is materialized into OWNED memory (CastTo deep-copies even at same dtype), so the
            // F32 source mmaps can be dropped below — passing non-F32 tensors through by reference forced
            // the loaders into `keep`, pinning the full ~15 GB checkpoint resident NEXT TO the ~6 GB BF16
            // working set for the model's whole lifetime. That duplication was the bulk of the ~29 GB
            // load-time host RSS behind the 2026-07-24 sweep's OOM-kill of the Swarm process; the quant
            // path already disposes its sources once its GGUF cache owns the weights.
            Dictionary<string, Tensor> bf16 = new Dictionary<string, Tensor>(lmWeights.Count, StringComparer.Ordinal);
            foreach ((string key, Tensor tensor) in lmWeights)
            {
                bf16[key] = tensor.CastTo(tensor.DType == DType.F32 ? DType.BF16 : tensor.DType);
            }
            pipeline.LoadWeights(bf16);
            foreach (IDisposable loader in lmLoaders)
            {
                loader.Dispose();
            }
            lmLoaders = [];
        }
        pipeline.LoadCodecWeights(codecWeights);

        Logs.Info($"[Audio][HeartMuLa] Loaded {repo} + {CodecRepo} (CSM-shaped LM{(useQuant ? $", {quant} quantized" : " bf16")} + HeartCodec, 48 kHz).");

        MusicAudio Synth(IBackend backend, MusicRequest request, CancellationToken ct)
        {
            ct.ThrowIfCancellationRequested();
            // Upstream prompt layout: [<tag>genre</tag>, MuQ row (pipeline-injected), lyrics] — Llama-3 BPE, each text
            // section lowercased and BOS/EOS-wrapped.
            int[] tags = AudioTextFrontend.HeartMulaTags(request.Genre ?? string.Empty);
            int[] lyrics = AudioTextFrontend.HeartMulaLyrics(request.Prompt ?? string.Empty);
            int maxFrames = (int)(Math.Clamp(request.Duration, 1d, 300d) * 12.5);   // HeartCodec frame rate = 12.5 Hz
            // Per-frame progress: the AR decode is the long pole, and silence in the log reads as "stuck".
            void OnFrame(int done, int total)
            {
                if (done <= 3 || done % 5 == 0 || done == total)
                {
                    Logs.Info($"[Audio][HeartMuLa] Frame {done}/{total} ({done / 12.5:0.0}s of audio).");
                }
            }
            float[] samples = pipeline.Generate(backend, lyrics, maxFrames, seed: request.Seed,
                temperature: request.Temperature.HasValue ? (float)request.Temperature.Value : null,
                topK: request.TopK,
                cfgScale: request.CfgScale.HasValue ? (float)request.CfgScale.Value : null,
                cancel: ct, onFrame: OnFrame, tagsTokens: tags);
            return MusicAudio.Mono(samples);
        }

        IDisposable?[] keep = [pipeline, .. lmLoaders, .. codecLoaders];
        return new MusicRunner(pipeline.SampleRate, Synth, keep);
    }
}
