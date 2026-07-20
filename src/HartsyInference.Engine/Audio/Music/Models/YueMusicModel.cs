using HartsyInference.Audio.Models.Codecs.Vocos;
using HartsyInference.Audio.Models.Codecs.XCodec;
using HartsyInference.Audio.Models.Music;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Requests;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Audio;

/// <summary>YuE Stage-1 (m-a-p/YuE-s1-7B-anneal-*) — a folder checkpoint plus a sibling tokenizer.model and
/// xcodec.safetensors. The optional sibling s2 folder adds the residual upsampler (full 8-codebook decode) and the
/// per-stem Vocos vocoders raise the output from the 16 kHz x-codec draft to 44.1 kHz.</summary>
internal static class YueMusicModel
{
    /// <summary>The YuE descriptor.</summary>
    internal static MusicModelDescriptor Descriptor { get; } = new MusicModelDescriptor
    {
        ManagesOwnWeights = false,
        CacheKey = selector => MusicCatalog.ResolveLocalCheckpoint(AudioWeightsCatalog.YueId, selector),
        LoadAsync = (_, selector, cancel) => LoadAsync(MusicCatalog.ResolveLocalCheckpoint(AudioWeightsCatalog.YueId, selector), cancel),
    };

    private static Task<IMusicRunner> LoadAsync(string folder, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        if (!Directory.Exists(folder))
        {
            throw new DirectoryNotFoundException(
                $"YuE checkpoint folder not found: '{folder}'. Place the m-a-p/YuE-s1-7B-anneal-* folder there; its "
                + "'tokenizer.model' ships inside it, and 'xcodec.safetensors' (converted from m-a-p/xcodec_mini_infer) "
                + "belongs in or beside it.");
        }
        string tokenizerPath = FindSibling(folder, "tokenizer.model")
            ?? throw new InvalidOperationException($"YuE needs 'tokenizer.model' (mm_tokenizer_v0.2_hf) in or beside '{folder}'.");
        string xcodecPath = EnsureXCodec(folder)
            ?? throw new InvalidOperationException(
                $"YuE needs 'xcodec.safetensors' (or the X-Codec 'ckpt_00360000.pth', which this loader auto-converts) in or beside '{folder}'.");

        YueConfig config = YueConfig.V1;
        YueTokenizer tokenizer = new YueTokenizer(tokenizerPath);
        (Dictionary<string, Tensor> stage1Weights, IDisposable stage1Loader) = YueCheckpointConverter.LoadStage1(folder, castToF32: false);
        YueCheckpointConverter.QuantizeLmWeights(stage1Weights, DType.Q4_K);   // 14 GB bf16 → ~3.5 GB: resident + fast dp4a GEMV
        YueStage1Lm stage1 = new YueStage1Lm(config);
        stage1.LoadWeights(stage1Weights, prefix: "model");

        (Dictionary<string, Tensor> codecWeights, SafeTensorsLoader codecLoader) = YueCheckpointConverter.LoadXCodec(xcodecPath, castToF32: true);
        XCodec xcodec = new XCodec(XCodecConfig.XCodec16kHz);
        xcodec.LoadWeights(codecWeights);

        // Stage-2 residual upsampler (predicts codebooks 1-7 from Stage-1's cb0) — an optional sibling folder.
        string? stage2Folder = FindSiblingFolder(folder, "s2") ?? FindSiblingFolder(folder, "stage2");
        YueStage2Lm? stage2 = null;
        IDisposable? stage2Loader = null;
        if (stage2Folder is not null)
        {
            (Dictionary<string, Tensor> stage2Weights, IDisposable loader) = YueCheckpointConverter.LoadStage2(stage2Folder, castToF32: false);
            YueCheckpointConverter.QuantizeLmWeights(stage2Weights, DType.Q4_K);
            stage2Loader = loader;
            stage2 = new YueStage2Lm(config, tokenizer.Soa, tokenizer.Stage1, tokenizer.Stage2);
            stage2.LoadWeights(stage2Weights, prefix: "model");
        }
        // Per-stem Vocos vocoders — YuE's real 44.1 kHz output. Optional: falls back to the 16 kHz x-codec draft.
        string? vocalVocoderPath = FindSibling(folder, "vocal_vocoder.safetensors");
        string? instrumentalVocoderPath = FindSibling(folder, "inst_vocoder.safetensors");
        VocosDecoder? vocalVocoder = null;
        VocosDecoder? instrumentalVocoder = null;
        IDisposable? vocalLoader = null;
        IDisposable? instrumentalLoader = null;
        if (vocalVocoderPath is not null && instrumentalVocoderPath is not null)
        {
            (Dictionary<string, Tensor> vocalWeights, SafeTensorsLoader loadedVocal) = YueCheckpointConverter.LoadVocoder(vocalVocoderPath);
            vocalLoader = loadedVocal;
            vocalVocoder = new VocosDecoder();
            vocalVocoder.LoadWeights(vocalWeights);
            (Dictionary<string, Tensor> instrumentalWeights, SafeTensorsLoader loadedInstrumental) = YueCheckpointConverter.LoadVocoder(instrumentalVocoderPath);
            instrumentalLoader = loadedInstrumental;
            instrumentalVocoder = new VocosDecoder();
            instrumentalVocoder.LoadWeights(instrumentalWeights);
        }

        YuePipeline pipeline = new YuePipeline(config, stage1, xcodec, stage2, vocalVocoder, instrumentalVocoder);
        Logs.Info($"[Audio][YuE] Loaded Stage-1{(stage2 is not null ? " + Stage-2 (full 8-codebook)" : " (vocal-cb0 only — no s2 folder)")}"
            + $"{(vocalVocoder is not null ? " + Vocos vocoders (44.1 kHz)" : " (16 kHz x-codec draft — no vocoders)")} from '{folder}'.");

        MusicAudio Synth(IBackend backend, MusicRequest request, CancellationToken ct)
        {
            ct.ThrowIfCancellationRequested();
            string genre = string.IsNullOrWhiteSpace(request.Genre) ? "pop" : request.Genre;
            // Segment-by-segment Stage-1 (infer.py): a head prompt plus one prompt per [label] section, each injecting
            // its own lyrics. Falls back to a single unstructured segment when the lyrics carry no [label] markers.
            int[] headIds = tokenizer.EncodeStage1Head(genre, request.Prompt);
            IReadOnlyList<string> segments = tokenizer.Stage1Segments(request.Prompt);
            List<int[]> segmentPrompts = new List<int[]>(Math.Max(segments.Count, 1));
            for (int i = 0; i < segments.Count; i++)
            {
                segmentPrompts.Add(tokenizer.EncodeSegmentPrompt(segments[i], isFirst: i == 0));
            }
            if (segmentPrompts.Count == 0)
            {
                segmentPrompts.Add(tokenizer.EncodeSegmentPrompt(request.Prompt ?? string.Empty, isFirst: true));
            }
            int maxFrames = (int)(Math.Clamp(request.Duration, 5d, 300d) * config.FrameRateHz);
            int perSegment = Math.Max(config.FrameRateHz, maxFrames / segmentPrompts.Count);
            return MusicAudio.Mono(pipeline.Synthesize(backend, headIds, segmentPrompts, maxFramesPerSegment: perSegment, seed: request.Seed));
        }

        List<IDisposable?> disposables = [pipeline, stage1Loader, codecLoader, tokenizer];
        if (stage2Loader is not null)
        {
            disposables.Add(stage2Loader);
        }
        if (vocalLoader is not null)
        {
            disposables.Add(vocalLoader);
        }
        if (instrumentalLoader is not null)
        {
            disposables.Add(instrumentalLoader);
        }
        return Task.FromResult<IMusicRunner>(new MusicRunner(pipeline.OutputSampleRate, Synth, [.. disposables]));
    }

    /// <summary>Ensures a loadable <c>xcodec.safetensors</c> exists, converting the downloaded X-Codec torch
    /// checkpoint on first use. Only the tensors the engine's X-Codec loader maps are kept, with their original keys,
    /// so the normal load path re-maps them identically. Null when neither form is present.</summary>
    private static string? EnsureXCodec(string folder)
    {
        string? existing = FindSibling(folder, "xcodec.safetensors");
        if (existing is not null)
        {
            return existing;
        }
        string parent = Directory.GetParent(folder)?.FullName ?? folder;
        string[] candidates =
        [
            Path.Combine(parent, "xcodec", "ckpt_00360000.pth"),
            Path.Combine(folder, "xcodec", "ckpt_00360000.pth"),
            Path.Combine(parent, "ckpt_00360000.pth"),
            Path.Combine(folder, "ckpt_00360000.pth"),
        ];
        string? checkpoint = candidates.FirstOrDefault(File.Exists);
        if (checkpoint is null)
        {
            return null;
        }
        string outputPath = Path.Combine(parent, "xcodec.safetensors");
        Logs.Info($"[Audio][YuE] Converting X-Codec '{Path.GetFileName(checkpoint)}' → xcodec.safetensors (one-time)...");
        using PytorchPickleLoader loader = new PytorchPickleLoader();
        loader.Load(checkpoint, recursiveFlatten: true);
        Dictionary<string, Tensor> raw = loader.GetAllTensors();
        Dictionary<string, Tensor> keep = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        foreach ((string key, Tensor tensor) in raw)
        {
            if (YueCheckpointConverter.MapXCodecKey(key) is not null)
            {
                keep[key] = tensor;
            }
        }
        if (keep.Count == 0)
        {
            throw new InvalidDataException($"X-Codec checkpoint '{checkpoint}' yielded no usable acoustic tensors — unexpected key layout.");
        }
        SafeTensorsWriter.Save(outputPath, keep);
        Logs.Info($"[Audio][YuE] Wrote {keep.Count} X-Codec acoustic tensors to '{outputPath}'.");
        return outputPath;
    }

    /// <summary>Finds a file inside the checkpoint folder, then one directory up (so variants can share one copy).</summary>
    private static string? FindSibling(string folder, string fileName)
    {
        string inside = Path.Combine(folder, fileName);
        if (File.Exists(inside))
        {
            return inside;
        }
        string parent = Path.Combine(Directory.GetParent(folder)?.FullName ?? folder, fileName);
        return File.Exists(parent) ? parent : null;
    }

    /// <summary>Directory analog of <see cref="FindSibling"/>: a subfolder of the checkpoint, then one level up.</summary>
    private static string? FindSiblingFolder(string folder, string name)
    {
        string inside = Path.Combine(folder, name);
        if (Directory.Exists(inside))
        {
            return inside;
        }
        string parent = Path.Combine(Directory.GetParent(folder)?.FullName ?? folder, name);
        return Directory.Exists(parent) ? parent : null;
    }
}
