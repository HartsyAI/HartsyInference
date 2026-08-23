using HartsyInference.Audio.Models.Codecs.Vocos;
using HartsyInference.Audio.Models.Codecs.XCodec;
using HartsyInference.Audio.Models.Music;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Placement;
using HartsyInference.Engine.Requests;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Audio;

/// <summary>YuE Stage-1 (m-a-p/YuE-s1-7B-anneal-*) — a folder checkpoint plus a sibling tokenizer.model and xcodec.safetensors. The optional sibling s2 folder adds the residual upsampler (full 8-codebook decode) and the per-stem Vocos vocoders raise the output from the 16 kHz x-codec draft to 44.1 kHz.</summary>
internal static class YueMusicModel
{
    /// <summary>Holds a loader created after the runner's disposable set was snapshotted (the lazily loaded x-codec encode branch), so it still gets disposed with the model.</summary>
    private sealed class LoaderBox : IDisposable
    {
        public IDisposable? Inner;
        public void Dispose() => Inner?.Dispose();
    }

    internal static MusicModelDescriptor Descriptor { get; } = new MusicModelDescriptor
    {
        ManagesOwnWeights = false,
        CacheKey = selector => MusicCatalog.ResolveLocalCheckpoint(AudioWeightsCatalog.YueId, selector),
        LoadAsync = (context, selector, cancel) => LoadAsync(context, MusicCatalog.ResolveLocalCheckpoint(AudioWeightsCatalog.YueId, selector), cancel),
    };

    private static Task<IMusicRunner> LoadAsync(MusicLoadContext context, string folder, CancellationToken cancel)
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
        ApplyLmQuant(stage1Weights, context.LmQuant);
        YueStage1Lm stage1 = new YueStage1Lm(config);
        stage1.LoadWeights(stage1Weights, prefix: "model");
        if (context.IsSharded)
        {
            stage1.Placement = BuildStage1Placement(context, stage1, config);
        }

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
            ApplyLmQuant(stage2Weights, context.LmQuant);
            stage2Loader = loader;
            stage2 = new YueStage2Lm(config, tokenizer.Soa, tokenizer.Stage1, tokenizer.Stage2);
            stage2.LoadWeights(stage2Weights, prefix: "model");
        }
        // Per-stem Vocos vocoders — YuE's real 44.1 kHz output. Optional: falls back to the 16 kHz x-codec draft.
        // Auto-converts the upstream torch checkpoints (decoder_131000/151000.pth) on first use, like EnsureXCodec.
        EnsureVocoders(folder);
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
        Logs.Info($"[Audio][YuE] Loaded Stage-1 (quant={context.LmQuant}"
            + $"{(stage1.Placement is not null ? ", layer-split across " + stage1.Placement.Stages.Count + " GPUs" : "")})"
            + $"{(stage2 is not null ? " + Stage-2 (full 8-codebook)" : " (vocal-cb0 only — no s2 folder)")}"
            + $"{(vocalVocoder is not null ? " + Vocos vocoders (44.1 kHz)" : " (16 kHz x-codec draft — no vocoders)")} from '{folder}'.");

        // Reference-audio ICL loads the codec's ~95M-param encode branch on first use only, so the ordinary
        // text-to-music path never pays for it. The extra loader lives in a box the runner disposes.
        LoaderBox encodeLoader = new();
        object encodeGate = new();

        bool EnsureEncodeBranch()
        {
            if (xcodec.CanEncode) return true;
            lock (encodeGate)
            {
                if (xcodec.CanEncode) return true;
                (Dictionary<string, Tensor> encodeWeights, SafeTensorsLoader loader) =
                    YueCheckpointConverter.LoadXCodec(xcodecPath, castToF32: true, forEncode: true);
                if (!xcodec.TryLoadEncodeWeights(encodeWeights))
                {
                    loader.Dispose();
                    return false;
                }
                encodeLoader.Inner = loader;
                Logs.Info("[Audio][YuE] Loaded the x-codec encode branch for reference-audio prompting.");
                return true;
            }
        }

        int[] EncodeReferenceCb0(IBackend backend, AudioClip clip, string what)
        {
            float[] mono = AudioClipCodec.DecodeMono(clip, XCodecConfig.XCodec16kHz.SampleRate);
            if (mono.Length < 320)
            {
                throw new InvalidOperationException(
                    $"YuE reference {what}audio decoded to {mono.Length} samples — too short to encode (needs at least "
                    + "one 320-sample codec frame). Supply a longer WAV clip.");
            }
            using Tensor pcm = new(new TensorShape(1, 1, mono.Length), DType.F32);
            mono.CopyTo(pcm.AsSpan<float>());
            using Tensor codes = xcodec.Encode(backend, pcm, mono.Length, nQ: 1);
            return codes.AsReadOnlySpan<int>().ToArray();
        }

        // infer.py: dual-track wins over the single-track prompt when both are supplied.
        int[] BuildReferenceBlock(IBackend backend, MusicRequest request)
        {
            bool dual = request.ReferenceVocal is not null && request.ReferenceInstrumental is not null;
            if (!dual && (request.ReferenceVocal is not null || request.ReferenceInstrumental is not null))
            {
                throw new InvalidOperationException(
                    "YuE dual-track reference audio needs BOTH the vocal and instrumental stems. Supply both, or use "
                    + "the single-track reference audio input instead.");
            }
            if (!dual && request.ReferenceAudio is null) return [];
            if (!EnsureEncodeBranch())
            {
                throw new InvalidOperationException(
                    $"YuE reference audio was supplied but '{xcodecPath}' carries no encode branch, so it cannot be "
                    + "converted to codec tokens. Restore the x-codec 'ckpt_00360000.pth' beside the checkpoint and "
                    + "reload so the export can be rebuilt.");
            }

            int[] vocal = EncodeReferenceCb0(backend, dual ? request.ReferenceVocal! : request.ReferenceAudio!,
                dual ? "vocal " : "");
            int[] instrumental = dual ? EncodeReferenceCb0(backend, request.ReferenceInstrumental!, "instrumental ") : [];
            int[] promptCodec = YueTokenizer.BuildAudioPromptCodec(vocal, instrumental,
                request.ReferenceStartSeconds, request.ReferenceEndSeconds);
            if (promptCodec.Length == 0)
            {
                Logs.Warning($"[Audio][YuE] Reference audio window [{request.ReferenceStartSeconds}s, "
                    + $"{request.ReferenceEndSeconds}s] selected no frames of a {vocal.Length / (double)config.FrameRateHz:0.##}s "
                    + "clip — generating without a reference.");
                return [];
            }
            Logs.Info($"[Audio][YuE] Reference audio: {promptCodec.Length} codec tokens "
                + $"({(dual ? "dual-track" : "single-track")}, {request.ReferenceStartSeconds}-{request.ReferenceEndSeconds}s).");
            return tokenizer.EncodeReferenceBlock(promptCodec);
        }

        MusicAudio Synth(IBackend backend, MusicRequest request, CancellationToken ct)
        {
            ct.ThrowIfCancellationRequested();
            string genre = string.IsNullOrWhiteSpace(request.Genre) ? "pop" : request.Genre;
            // Segment-by-segment Stage-1 (infer.py): a head prompt plus one prompt per [label] section, each injecting
            // its own lyrics. Falls back to a single unstructured segment when the lyrics carry no [label] markers.
            int[] headIds = tokenizer.EncodeStage1Head(genre, request.Prompt);
            // infer.py appends the reference block AFTER the head text, so only segment 0 carries it; later segments
            // inherit it through the running context. The pipeline collects only generated tokens, so the reference
            // codes are never decoded as output (upstream's `range_begin = 1` skip has no analogue here).
            int[] referenceBlock = BuildReferenceBlock(backend, request);
            if (referenceBlock.Length > 0)
            {
                headIds = [.. headIds, .. referenceBlock];
            }
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
            return MusicAudio.Mono(pipeline.Synthesize(backend, headIds, segmentPrompts, maxFramesPerSegment: perSegment,
                seed: request.Seed,
                temperature: (float?)request.Temperature,
                topK: request.TopK,
                topP: (float?)request.TopP,
                // < 1 rewards repetition and collapses Stage-1 into a loop; the config default is the reference 1.1.
                repetitionPenalty: request.RepetitionPenalty is { } penalty ? (float)Math.Max(1d, penalty) : null,
                guidanceScale: (float?)request.CfgScale));
        }

        List<IDisposable?> disposables = [pipeline, stage1Loader, codecLoader, tokenizer, encodeLoader];
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

    /// <summary>Applies the resolved LM precision policy: Q4_K/Q8_0 quantize the big GEMM weights in place; <see cref="AudioLmQuant.Off"/> keeps checkpoint precision (bf16) — the pooled-VRAM quality path.</summary>
    private static void ApplyLmQuant(Dictionary<string, Tensor> weights, AudioLmQuant quant)
    {
        switch (quant)
        {
            case AudioLmQuant.Q4K:
                YueCheckpointConverter.QuantizeLmWeights(weights, DType.Q4_K);
                break;
            case AudioLmQuant.Q8:
                YueCheckpointConverter.QuantizeLmWeights(weights, DType.Q8_0);
                break;
            case AudioLmQuant.Off:
                break;
        }
    }

    /// <summary>Plans the Stage-1 layer split across the context's shard devices (explicit ratios win, else proportional to live free VRAM) and binds each stage to its resolved backend.</summary>
    private static LlmPlacement BuildStage1Placement(MusicLoadContext context, YueStage1Lm stage1, YueConfig config)
    {
        int layers = config.Stage1.NumHiddenLayers;
        long totalBytes = 0;
        foreach (Tensor tensor in stage1.EnumerateWeights())
        {
            totalBytes += Tensor.ComputeByteSize(tensor.Shape, tensor.DType);
        }
        string[] devices = [.. context.ShardStages!.Select(s => s.Selector)];
        IReadOnlyList<LlmStagePlan> plan = PlacementPlanner.LlmSplitPlan(
            devices, context.ShardRatios, layers, totalBytes / Math.Max(1, layers));
        List<LlmStage> stages = new(plan.Count);
        foreach (LlmStagePlan stagePlan in plan)
        {
            IBackend backend = context.ShardStages!.First(s => s.Selector == stagePlan.Device).Backend;
            stages.Add(new LlmStage(backend, stagePlan.StartLayer, stagePlan.EndLayer));
        }
        Logs.Info($"[Audio][YuE] Stage-1 layer split: "
            + string.Join(" + ", plan.Select(p => $"{p.Device}[{p.StartLayer},{p.EndLayer})")) + ".");
        return new LlmPlacement([.. stages]);
    }

    /// <summary>Ensures a loadable <c>xcodec.safetensors</c> exists, converting the downloaded X-Codec torch checkpoint on first use. Only the tensors the engine's X-Codec loader maps are kept, with their original keys, so the normal load path re-maps them identically. Null when neither form is present.</summary>
    private static string? EnsureXCodec(string folder)
    {
        string? existing = FindSibling(folder, "xcodec.safetensors");
        string parent = Directory.GetParent(folder)?.FullName ?? folder;
        string[] candidates =
        [
            Path.Combine(parent, "xcodec", "ckpt_00360000.pth"),
            Path.Combine(folder, "xcodec", "ckpt_00360000.pth"),
            Path.Combine(parent, "ckpt_00360000.pth"),
            Path.Combine(folder, "ckpt_00360000.pth"),
        ];
        string? checkpoint = candidates.FirstOrDefault(File.Exists);

        if (existing is not null)
        {
            // Exports written before the encode roots were kept are decode-only, and returning early on their mere
            // existence would strand every installed copy encode-less forever. Re-repack in place (the stale file may
            // sit in `folder` while a fresh one would land in `parent`, where it would stay shadowed).
            if (XCodecExportCanEncode(existing))
            {
                return existing;
            }
            if (checkpoint is null)
            {
                Logs.Warning($"[Audio][YuE] '{existing}' predates the x-codec encode branch and the source "
                    + "'ckpt_00360000.pth' is gone, so it cannot be rebuilt — reference-audio (ICL) prompting will be "
                    + "refused. Restore the checkpoint from m-a-p/xcodec_mini_infer to enable it.");
                return existing;
            }
            Logs.Info($"[Audio][YuE] '{Path.GetFileName(existing)}' lacks the x-codec encode branch — rebuilding it "
                + $"from '{Path.GetFileName(checkpoint)}' (one-time)...");
            RepackXCodec(checkpoint, existing);
            return existing;
        }

        if (checkpoint is null)
        {
            return null;
        }
        string outputPath = Path.Combine(parent, "xcodec.safetensors");
        Logs.Info($"[Audio][YuE] Converting X-Codec '{Path.GetFileName(checkpoint)}' → xcodec.safetensors (one-time)...");
        RepackXCodec(checkpoint, outputPath);
        return outputPath;
    }

    /// <summary>Repacks the torch x-codec checkpoint keeping every tensor the loader maps in EITHER direction, under its original key, so the decode load path re-maps it identically and the ICL encode path finds its roots. Writes via a temp file so a failure cannot leave a truncated export shadowing a good one.</summary>
    private static void RepackXCodec(string checkpoint, string outputPath)
    {
        string temp = outputPath + ".tmp";
        try
        {
            PickleCheckpointRepacker.Repack(checkpoint, temp,
                key => YueCheckpointConverter.MapXCodecKey(key, forEncode: true) is not null ? key : null,
                recursiveFlatten: true);
            File.Move(temp, outputPath, overwrite: true);
        }
        finally
        {
            if (File.Exists(temp)) File.Delete(temp);
        }
    }

    /// <summary>Whether an existing export carries the encode roots. Reads only the safetensors header.</summary>
    private static bool XCodecExportCanEncode(string path)
    {
        try
        {
            using SafeTensorsLoader loader = new();
            loader.Load(path);
            return YueCheckpointConverter.XCodecExportHasEncodeRoots(loader.Descriptors.Keys);
        }
        catch (Exception ex)
        {
            // Treat an unreadable header as "good enough" — the real load right after will surface the true error,
            // and answering false here would rewrite the file on every single load.
            Logs.Error($"[Audio][YuE] Could not inspect '{path}' for the x-codec encode branch", ex);
            return true;
        }
    }

    /// <summary>Converts the upstream xcodec_mini_infer Vocos vocoder checkpoints (<c>decoder_131000.pth</c> = vocal, <c>decoder_151000.pth</c> = instrumental) to the safetensors the loader consumes, one time. The torch state dicts already use VocosDecoder's exact key layout (backbone.*/head.out), so this is a pure format conversion — same self-healing pattern as <see cref="EnsureXCodec"/>. No-op when the converted files exist or the sources are absent (16 kHz draft fallback stays available).</summary>
    private static void EnsureVocoders(string folder)
    {
        if (FindSibling(folder, "vocal_vocoder.safetensors") is not null
            && FindSibling(folder, "inst_vocoder.safetensors") is not null)
        {
            return;
        }
        string parent = Directory.GetParent(folder)?.FullName ?? folder;
        ConvertVocoder(folder, parent, "decoder_131000.pth", "vocal_vocoder.safetensors");
        ConvertVocoder(folder, parent, "decoder_151000.pth", "inst_vocoder.safetensors");
    }

    private static void ConvertVocoder(string folder, string parent, string sourceName, string targetName)
    {
        if (File.Exists(Path.Combine(parent, targetName)) || File.Exists(Path.Combine(folder, targetName)))
        {
            return;
        }
        string[] candidates =
        [
            Path.Combine(parent, "xcodec_vocoder_src", sourceName),
            Path.Combine(folder, "xcodec_vocoder_src", sourceName),
            Path.Combine(parent, "decoders", sourceName),
            Path.Combine(folder, "decoders", sourceName),
            Path.Combine(parent, sourceName),
            Path.Combine(folder, sourceName),
        ];
        string? checkpoint = candidates.FirstOrDefault(File.Exists);
        if (checkpoint is null)
        {
            return;
        }
        try
        {
            string outputPath = Path.Combine(parent, targetName);
            Logs.Info($"[Audio][YuE] Converting Vocos vocoder '{Path.GetFileName(checkpoint)}' → {targetName} (one-time)...");
            // Torch keys already match VocosDecoder's layout, so this is a pure format conversion.
            PickleCheckpointRepacker.Repack(checkpoint, outputPath, recursiveFlatten: true);
        }
        catch (Exception ex)
        {
            Logs.Error($"[Audio][YuE] Vocoder conversion failed for '{checkpoint}' — continuing with the 16 kHz x-codec draft", ex);
        }
    }

    /// <summary>Finds a file inside the checkpoint folder, then one directory up (so variants can share one copy).</summary>
    private static string? FindSibling(string folder, string fileName)
        => Probe(folder, fileName, File.Exists);

    /// <summary>Directory analog of <see cref="FindSibling"/>: a subfolder of the checkpoint, then one level up.</summary>
    private static string? FindSiblingFolder(string folder, string name)
        => Probe(folder, name, Directory.Exists);

    /// <summary>Looks for <paramref name="name"/> under the checkpoint folder, its parent, and any case-variant of that parent — consumers disagree on the family root's casing (the engine resolves <c>music/yue</c> from the family id, AudioLab builds <c>music/YuE</c> from the provider's display prefix), and on a case-sensitive filesystem those are two trees. Without this, sidecars dropped in one are invisible from the other and the pipeline silently degrades to the cb0-only 16 kHz draft. Each level is matched case-insensitively too.</summary>
    private static string? Probe(string folder, string name, Func<string, bool> exists)
    {
        string? parent = Directory.GetParent(folder)?.FullName;
        foreach (string root in Roots(folder, parent))
        {
            string exact = Path.Combine(root, name);
            if (exists(exact))
            {
                return exact;
            }
            if (!Directory.Exists(root))
            {
                continue;
            }
            string? match = Directory.EnumerateFileSystemEntries(root)
                .FirstOrDefault(e => string.Equals(Path.GetFileName(e), name, StringComparison.OrdinalIgnoreCase) && exists(e));
            if (match is not null)
            {
                return match;
            }
        }
        return null;
    }

    /// <summary>Search roots in priority order: the checkpoint folder, its parent, then the parent's case-variant siblings (the split-family-root case).</summary>
    private static IEnumerable<string> Roots(string folder, string? parent)
    {
        yield return folder;
        if (parent is null)
        {
            yield break;
        }
        yield return parent;
        string? grandparent = Directory.GetParent(parent)?.FullName;
        if (grandparent is null)
        {
            yield break;
        }
        string parentName = Path.GetFileName(parent);
        foreach (string sibling in Directory.EnumerateDirectories(grandparent))
        {
            if (!string.Equals(sibling, parent, StringComparison.Ordinal)
                && string.Equals(Path.GetFileName(sibling), parentName, StringComparison.OrdinalIgnoreCase))
            {
                yield return sibling;
            }
        }
    }
}
