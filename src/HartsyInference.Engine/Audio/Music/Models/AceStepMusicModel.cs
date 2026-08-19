using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Frontends;
using HartsyInference.Audio.Models.Codecs.Oobleck;
using HartsyInference.Audio.Models.Music;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Music;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.Engine.Requests;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Audio;

/// <summary>ACE-Step 1.5 — a flow-matching music DiT over 25 Hz Oobleck latents → 48 kHz stereo. The DiT variant is a
/// registered checkpoint that self-heals by download when missing; the Oobleck VAE and the Qwen3-Embedding condition
/// encoder come from the Comfy-Org distribution. Optionally driven by the 5 Hz LM planner.</summary>
internal static class AceStepMusicModel
{
    private const int QwenEosId = 151643;   // Qwen3-Embedding <|endoftext|>
    private const string AceStep15Repo = "Comfy-Org/ace_step_1.5_ComfyUI_files";
    private const string AceStep15VaeFile = "split_files/vae/ace_1.5_vae.safetensors";
    private const string AceStep15TurboFile = "split_files/diffusion_models/acestep_v1.5_turbo.safetensors";
    private const string QwenEmbeddingRepo = "Qwen/Qwen3-Embedding-0.6B";
    private const string QwenEmbeddingFile = "model.safetensors";

    /// <summary>The ACE-Step descriptor.</summary>
    internal static MusicModelDescriptor Descriptor { get; } = new MusicModelDescriptor
    {
        ManagesOwnWeights = false,
        CacheKey = selector => MusicCatalog.ResolveLocalCheckpoint(AudioWeightsCatalog.AceStepId, selector),
        LoadAsync = LoadAsync,
    };

    private static async Task<IMusicRunner> LoadAsync(MusicLoadContext context, AudioModelSelector selector, CancellationToken cancel)
    {
        string variant = (selector.Variant ?? string.Empty).Trim();
        string localPath = MusicCatalog.ResolveLocalCheckpoint(AudioWeightsCatalog.AceStepId, selector);
        // Migration: the legacy Comfy-Org filename is byte-identical to the official bundle turbo (same upstream LFS
        // sha256) — rename instead of re-downloading 4.8 GB.
        if (!File.Exists(localPath) && variant.Equals("turbo", StringComparison.OrdinalIgnoreCase))
        {
            string legacy = Path.Combine(Path.GetDirectoryName(localPath)!, "acestep_v1.5_turbo.safetensors");
            if (File.Exists(legacy))
            {
                Logs.Info($"[Audio][ACE-Step] Migrating legacy turbo checkpoint filename → '{Path.GetFileName(localPath)}'.");
                File.Move(legacy, localPath);
            }
        }
        string mainPath;
        if (File.Exists(localPath))
        {
            mainPath = localPath;
        }
        else if (AudioWeightsCatalog.AssetsFor(AudioWeightsCatalog.AceStepId, variant).Count > 0)
        {
            Logs.Info($"[Audio][ACE-Step] '{variant}' weights missing — downloading this variant's file set.");
            await AudioWeightsCatalog.EnsureAsync(AudioWeightsCatalog.AceStepId, variant, cancel).ConfigureAwait(false);
            mainPath = localPath;
        }
        else
        {
            mainPath = await AudioModelCache.GetAsync(AceStep15Repo, AceStep15TurboFile, category: "music", ct: cancel).ConfigureAwait(false);
        }
        // v1 all-in-one (Comfy repackage) routes to its own arm — BEFORE the 1.5 companion downloads.
        {
            bool isV1;
            using (SafeTensorsLoader sniff = new SafeTensorsLoader())
            {
                sniff.Load(mainPath);
                isV1 = AceStepCheckpointConverter.IsV1AllInOne(sniff.Descriptors);
            }
            if (isV1)
            {
                return await LoadV1Async(context, mainPath, cancel).ConfigureAwait(false);
            }
        }

        string vaePath = await AudioModelCache.GetAsync(AceStep15Repo, AceStep15VaeFile, category: "music", ct: cancel).ConfigureAwait(false);
        string qwenPath = await AudioModelCache.GetAsync(QwenEmbeddingRepo, QwenEmbeddingFile, category: "music", ct: cancel).ConfigureAwait(false);

        // The sidecar config.json (downloaded with the variant) drives dims + is_turbo; absent = 2B turbo defaults.
        string sidecarConfig = Path.ChangeExtension(mainPath, null) + ".config.json";
        AceStep15Config config = File.Exists(sidecarConfig)
            ? AceStep15Config.FromJson(await File.ReadAllTextAsync(sidecarConfig, cancel).ConfigureAwait(false))
            : new AceStep15Config();
        // Keep bf16 residency (upstream's inference dtype; ~half the host RAM + PCIe streaming — XL would not fit as
        // F32). Host-read tensors are cast to F32 selectively inside the models.
        (Dictionary<string, Tensor> weights, SafeTensorsLoader mainLoader) = AceStepCheckpointConverter.LoadModel15(mainPath, castToF32: false);
        AceStep15Dit dit = new AceStep15Dit(config);
        dit.LoadWeights(weights);
        // The condition side runs at encoder width (XL: 2048 under a 2560 decoder; identity for 2B).
        AceStep15ConditionEncoder conditionEncoder = new AceStep15ConditionEncoder(config.EncoderVariant());
        conditionEncoder.LoadWeights(weights);

        SafeTensorsLoader vaeLoader = new SafeTensorsLoader();
        vaeLoader.Load(vaePath);
        OobleckVae vae = new OobleckVae(OobleckConfig.AceStep15);
        vae.LoadWeights(vaeLoader.GetAllTensors());

        SafeTensorsLoader qwenLoader = new SafeTensorsLoader();
        qwenLoader.Load(qwenPath);
        LlamaStyleEncoder qwen = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Qwen3_Embedding_0_6B);
        qwen.LoadWeights(qwenLoader.GetAllTensors());
        Qwen3Tokenizer tokenizer = new Qwen3Tokenizer();

        AceStepPipeline15 pipeline = new AceStepPipeline15(context.Backend, dit, conditionEncoder, vae, config);
        LoadSilenceLatent(pipeline, Path.GetDirectoryName(mainPath));
        // 5 Hz code detokenizer (LM-planner hints → 25 Hz latents); its weights ride in the same checkpoint.
        AceStep15AudioDetokenizer detokenizer = new AceStep15AudioDetokenizer(config);
        detokenizer.LoadWeights(weights);
        PlannerHolder plannerHolder = new PlannerHolder();
        // The learned CFG uncond row ships in every checkpoint; base/sft need it (turbo keeps CFG off).
        if (weights.TryGetValue("null_condition_emb", out Tensor? nullEmb))
        {
            pipeline.SetNullConditionEmb(CopyToF32(nullEmb));
        }
        (int defaultSteps, float defaultShift) = VariantDefaults(variant);
        Logs.Info($"[Audio][ACE-Step] Loaded 1.5 '{variant}' ({(config.IsTurbo ? "turbo" : "CFG")}, 48 kHz stereo, "
            + $"default {defaultSteps} steps, shift {defaultShift}).");

        // <|endoftext|> is a special token: tokenize the surrounding text separately and splice the id in. The
        // Qwen3-Embedding post-processor then appends one more <|endoftext|>, which is replicated here.
        int[] TokenizeWithEos(string body, string tail = "")
        {
            int[] raw = AudioTextFrontend.Qwen3Ids(body ?? string.Empty);
            int[] tailIds = tail.Length == 0 ? [] : AudioTextFrontend.Qwen3Ids(tail);
            int[] tokens = new int[raw.Length + 1 + tailIds.Length + 1];
            raw.CopyTo(tokens, 0);
            tokens[raw.Length] = QwenEosId;
            tailIds.CopyTo(tokens, raw.Length + 1);
            tokens[^1] = QwenEosId;
            return tokens;
        }

        // Text/caption branch: a full Qwen3-Embedding forward → last_hidden_state (upstream infer_text_embeddings).
        Tensor EncodeQwenText(IBackend device, int[] tokens)
        {
            Tensor batch = qwen.Encode(device, [tokens]);
            Tensor sliced = CfgHelper.SliceBatchElement(batch, 0, tokens.Length, config.TextHiddenDim);
            batch.Dispose();
            return sliced;
        }

        // Lyric branch: embedding-table lookup ONLY — upstream infer_lyric_embeddings uses embed_tokens, not a forward.
        Tensor EmbedQwenLyrics(int[] tokens)
        {
            Tensor batch = qwen.LookupEmbeddings(tokens);
            Tensor sliced = CfgHelper.SliceBatchElement(batch, 0, tokens.Length, config.TextHiddenDim);
            batch.Dispose();
            return sliced;
        }

        MusicAudio Synth(IBackend device, MusicRequest request, CancellationToken ct)
        {
            ct.ThrowIfCancellationRequested();
            if (request.UseErgTag is not null || request.UseErgLyric is not null || request.UseErgDiffusion is not null)
            {
                throw new NotSupportedException(
                    "The ERG toggles are ACE-Step v1 only — 1.5's guidance stack has no ERG stage. Unset them or pick a v1 checkpoint.");
            }
            EditInputs? edit = ResolveEdit(request, config);
            // Continuation re-generates the source AND the appended tail in one pass, so the DiT duration is the sum;
            // repaint/cover run at the source's own length and ignore the requested duration.
            double duration = edit is null ? Math.Clamp(request.Duration, 1d, 600d) : edit.TotalSeconds;
            // Upstream SFT_GEN_PROMPT layout: the DiT is trained on THIS format — a bare style string is
            // out-of-distribution.
            string caption = string.IsNullOrWhiteSpace(request.Genre) ? "pop" : request.Genre;
            string metas =
                $"- bpm: {request.Bpm?.ToString(System.Globalization.CultureInfo.InvariantCulture) ?? "N/A"}\n"
                + $"- timesignature: {(string.IsNullOrWhiteSpace(request.TimeSignature) ? "N/A" : request.TimeSignature)}\n"
                + $"- keyscale: {(string.IsNullOrWhiteSpace(request.KeyScale) ? "N/A" : request.KeyScale)}\n"
                + $"- duration: {(int)duration} seconds\n";
            // Upstream swaps the instruction line per task (TASK_INSTRUCTIONS); the DiT is trained on the task-specific
            // wording, and continuation is a masked repaint of the appended tail.
            string instruction = edit?.Mode switch
            {
                "repaint" or "continuation" => "Repaint the mask area based on the given conditions:",
                "cover" => "Generate audio semantic tokens based on the given conditions:",
                _ => "Fill the audio semantic mask based on the given conditions:",
            };
            string textPrompt = $"# Instruction\n{instruction}\n\n# Caption\n{caption}\n\n# Metas\n{metas}";
            Tensor textHidden = EncodeQwenText(device, TokenizeWithEos(textPrompt, tail: "\n"));
            // Upstream _format_lyrics; "[Instrumental]" is the upstream convention for no vocals.
            string language = string.IsNullOrWhiteSpace(request.VocalLanguage) ? "en" : request.VocalLanguage;
            string lyrics = string.IsNullOrWhiteSpace(request.Prompt) ? "[Instrumental]" : request.Prompt;
            Tensor lyricHidden = EmbedQwenLyrics(TokenizeWithEos($"# Languages\n{language}\n\n# Lyric\n{lyrics}"));
            device.Sync();
            device.FreeWeights(qwen.EnumerateWeights());
            try
            {
                // Optional 5 Hz LM planner: caption+lyrics → FSQ codes → detokenized 25 Hz hints.
                Tensor? lmHints = null;
                bool lmThink = false;
                string lmKind = (request.LmModel ?? string.Empty).Trim().ToLowerInvariant();
                if (edit is not null && lmKind is not ("" or "none" or "disabled"))
                {
                    throw new NotSupportedException(
                        $"ACE-Step '{edit.Mode}' and the 5 Hz LM planner both occupy the src_latents slot — "
                        + "disable the planner (LmModel = \"none\") to use an audio-conditioned edit mode.");
                }
                if (lmKind is not ("" or "none" or "disabled"))
                {
                    AceStepLmPlanner planner = plannerHolder.GetOrLoad(lmKind);
                    lmThink = request.Thinking;
                    (int[] codes, string _) = planner.Plan(device, caption, lyrics, duration, new AceStepPlannerOptions
                    {
                        Thinking = request.Thinking,
                        Temperature = (float)request.LmTemperature,
                        CfgScale = (float)request.LmCfgScale,
                        TopK = request.LmTopK,
                        TopP = (float)request.LmTopP,
                        NegativePrompt = request.LmNegativePrompt ?? string.Empty,
                        Seed = request.Seed,
                    }, ct);
                    device.Sync();
                    device.FreeWeights(planner.EnumerateWeights());
                    Tensor raw = detokenizer.Decode(device, codes);   // [1, codes*5, 64]
                    lmHints = FitHints(raw, config.FrameCount(duration), config.LatentChannels);
                }
                AceStep15GenerateOptions options = new AceStep15GenerateOptions
                {
                    Shift = request.Shift.HasValue ? (float)request.Shift.Value : defaultShift,
                    Seed = request.Seed,
                    InferSteps = request.InferSteps ?? defaultSteps,
                    // The pipeline clamps guidance to 1.0 on turbo configs itself (upstream behavior).
                    GuidanceScale = request.CfgScale.HasValue ? (float)request.CfgScale.Value : 7f,
                    UseAdg = request.UseAdg,
                    GuidanceType = request.GuidanceType,
                    CfgIntervalStart = (float)request.CfgIntervalStart,
                    CfgIntervalEnd = (float)request.CfgIntervalEnd,
                    InferMethod = string.IsNullOrWhiteSpace(request.InferMethod) ? "ode" : request.InferMethod,
                    // Upstream DCW scalers flip with the LM-think state (dcw_defaults.py).
                    DcwScaler = lmHints is not null && lmThink ? 0.02f : 0.05f,
                    DcwHighScaler = lmHints is not null && lmThink ? 0.06f : 0.02f,
                };
                AceStep15EditPlan? editPlan = edit is null ? null : BuildEditPlan(device, vae, config, edit);
                try
                {
                    (float[] left, float[] right, int _, int _) =
                        pipeline.Generate(textHidden, lyricHidden, duration, options, editPlan, lmHints: lmHints, cancel: ct);
                    return MusicAudio.Stereo(left, right);
                }
                finally
                {
                    lmHints?.Dispose();
                    editPlan?.SrcLatent.Dispose();
                }
            }
            finally
            {
                textHidden.Dispose();
                lyricHidden.Dispose();
            }
        }

        return new MusicRunner(48_000, Synth, pipeline, detokenizer, plannerHolder, qwen, tokenizer, mainLoader, vaeLoader, qwenLoader);
    }

    /// <summary>Copies a (possibly bf16) tensor into a fresh owned F32 tensor the pipeline can retain.</summary>
    private static unsafe Tensor CopyToF32(Tensor source)
    {
        Tensor copy = new Tensor(source.Shape, DType.F32);
        Tensor f32 = source.DType == DType.F32 ? source : source.CastTo(DType.F32);
        long bytes = f32.Shape.ElementCount * 4;
        Buffer.MemoryCopy((void*)f32.DataPointer, (void*)copy.DataPointer, bytes, bytes);
        if (!ReferenceEquals(f32, source))
        {
            f32.Dispose();
        }
        return copy;
    }

    /// <summary>Crops/pads detokenized hints <c>[1, T, 64]</c> to the pipeline's exact frame count (upstream crops to
    /// src length; short hints repeat the final frame).</summary>
    private static unsafe Tensor FitHints(Tensor raw, int frames, int latentChannels)
    {
        int t = (int)raw.Shape[1];
        if (t == frames)
        {
            return raw;
        }
        Tensor fitted = new Tensor(new TensorShape(1, frames, latentChannels), DType.F32);
        float* source = (float*)raw.DataPointer;
        float* destination = (float*)fitted.DataPointer;
        for (int i = 0; i < frames; i++)
        {
            long offset = (long)Math.Min(i, t - 1) * latentChannels;
            Buffer.MemoryCopy(source + offset, destination + (long)i * latentChannels, latentChannels * 4, latentChannels * 4);
        }
        raw.Dispose();
        return fitted;
    }

    /// <summary>Decodes and validates the editing-mode source clip, returning null for plain text-to-music. Only one of
    /// continuation/repaint/cover may be set (<see cref="Services.MusicService"/> enforces exclusivity first).</summary>
    private static EditInputs? ResolveEdit(MusicRequest request, AceStep15Config config)
    {
        AudioClip? clip = request.Continuation ?? request.Repaint ?? request.Cover;
        if (clip is null)
        {
            return null;
        }
        string mode = request.Continuation is not null ? "continuation" : request.Repaint is not null ? "repaint" : "cover";
        (float[] left, float[] right) = AudioClipCodec.DecodeStereo(clip, config.SampleRate);
        if (left.Length == 0)
        {
            throw new ArgumentException($"The ACE-Step '{mode}' source clip decoded to no audio.", nameof(request));
        }
        double sourceSeconds = Math.Min(left.Length, right.Length) / (double)config.SampleRate;
        double total = mode == "continuation" ? sourceSeconds + Math.Max(request.Duration, 0d) : sourceSeconds;
        if (total < 1d || total > 600d)
        {
            throw new ArgumentException(
                $"ACE-Step '{mode}' resolves to a {total:0.0}s generation; the pipeline accepts 1..600 s "
                + $"(source is {sourceSeconds:0.0}s).", nameof(request));
        }
        if (mode == "continuation" && request.Duration < 1d / config.LatentRate)
        {
            throw new ArgumentException(
                $"ACE-Step 'continuation' needs a Duration of at least one latent frame ({1d / config.LatentRate:0.00}s) to append.", nameof(request));
        }
        if (mode == "repaint")
        {
            if (request.RepaintEnd <= request.RepaintStart)
            {
                throw new ArgumentException(
                    $"ACE-Step 'repaint' needs RepaintEnd > RepaintStart; got [{request.RepaintStart:0.00}, {request.RepaintEnd:0.00}] s.", nameof(request));
            }
            if (request.RepaintStart < 0d || request.RepaintStart >= sourceSeconds)
            {
                throw new ArgumentException(
                    $"ACE-Step 'repaint' span starts at {request.RepaintStart:0.00}s, outside the {sourceSeconds:0.0}s source.", nameof(request));
            }
        }
        // Cover strength is the schedule entry point; 0 would emit the source unchanged, so the floor keeps one step.
        float startSigma = mode == "cover" ? (float)Math.Clamp(request.CoverStrength, 0.05d, 1d) : 1f;
        return new EditInputs
        {
            Mode = mode,
            Left = left,
            Right = right,
            SourceSeconds = sourceSeconds,
            TotalSeconds = total,
            RepaintStart = request.RepaintStart,
            RepaintEnd = request.RepaintEnd,
            StartSigmaFraction = startSigma,
        };
    }

    /// <summary>Builds the pipeline edit plan for one mode: source latents from the VAE encoder plus the per-frame
    /// chunk mask (1 = generate, 0 = preserve) and schedule entry point. <b>Cover is an approximation</b> — upstream
    /// feeds FSQ-detokenized 5 Hz hints as <c>src_latents</c> with <c>is_covers=1</c>, and the 25 Hz-latent → 5 Hz-code
    /// tokenizer half is not ported here, so raw 25 Hz Oobleck latents are substituted; upstream's cover strength also
    /// blends a cover-instruction and a text2music-instruction velocity per step, where this maps it to the schedule
    /// entry point instead. Parity-pending on all three modes — none is validated against real weights.</summary>
    private static AceStep15EditPlan BuildEditPlan(IBackend device, OobleckVae vae, AceStep15Config config, EditInputs edit)
    {
        int frames = config.FrameCount(edit.TotalSeconds);
        bool continuation = edit.Mode == "continuation";
        Tensor src = EncodeSourceLatents(device, vae, config, edit, frames, padWithLastFrame: !continuation, out int sourceFrames);
        try
        {
            float[] mask = new float[frames];
            if (continuation)
            {
                if (sourceFrames >= frames)
                {
                    throw new ArgumentException(
                        $"ACE-Step 'continuation' left no frames to generate: the source already fills all {frames} latent frames.");
                }
                for (int i = sourceFrames; i < frames; i++)
                {
                    mask[i] = 1f;
                }
            }
            else if (edit.Mode == "repaint")
            {
                int start = Math.Clamp((int)Math.Floor(edit.RepaintStart * config.LatentRate), 0, frames);
                int end = Math.Clamp((int)Math.Ceiling(edit.RepaintEnd * config.LatentRate), 0, frames);
                if (end <= start)
                {
                    throw new ArgumentException(
                        $"ACE-Step 'repaint' span [{edit.RepaintStart:0.00}, {edit.RepaintEnd:0.00}] s covers no latent frame at {config.LatentRate} Hz.");
                }
                for (int i = start; i < end; i++)
                {
                    mask[i] = 1f;
                }
            }
            else
            {
                Array.Fill(mask, 1f);
            }
            Logs.Info($"[Audio][ACE-Step] Edit mode '{edit.Mode}': {frames} latent frames, {sourceFrames} from source, "
                + $"start sigma {edit.StartSigmaFraction:0.00} (parity-pending — not validated against real weights).");
            return new AceStep15EditPlan
            {
                SrcLatent = src,
                ChunkMask = mask,
                StartSigmaFraction = edit.StartSigmaFraction,
                // Upstream substitutes the tiled silence latent inside a repaint window; a continuation's tail has no
                // source at all, so the same rule gives it the right src row. Cover conditions on the whole source.
                SilenceMaskedFrames = edit.Mode != "cover",
            };
        }
        catch (Exception ex)
        {
            Logs.Error($"[Audio][ACE-Step] Failed to build the '{edit.Mode}' edit plan.", ex);
            src.Dispose();
            throw;
        }
    }

    /// <summary>VAE-encodes the source PCM to src latents <c>[1, frames, latentChannels]</c> using the same
    /// <c>EncodeMode</c> + transpose recipe the pipeline's silence latent uses; rows past the source are either the
    /// repeated final frame or zero (continuation, where the context falls back to the silence latent instead).</summary>
    private static unsafe Tensor EncodeSourceLatents(IBackend device, OobleckVae vae, AceStep15Config config,
        EditInputs edit, int frames, bool padWithLastFrame, out int sourceFrames)
    {
        if (!vae.CanEncode)
        {
            throw new NotSupportedException(
                $"ACE-Step '{edit.Mode}' needs the Oobleck VAE encoder to turn the source clip into src latents, but the "
                + "loaded VAE checkpoint is decode-only.");
        }
        int channels = Math.Max(1, vae.AudioChannels);
        int available = Math.Min(edit.Left.Length, edit.Right.Length) / config.SamplesPerLatent;
        if (available < 1)
        {
            throw new ArgumentException(
                $"The ACE-Step '{edit.Mode}' source is shorter than one latent frame "
                + $"({config.SamplesPerLatent} samples at {config.SampleRate} Hz).");
        }
        int encodeFrames = Math.Min(available, frames);
        int samples = encodeFrames * config.SamplesPerLatent;
        Tensor pcm = new Tensor(new TensorShape(1, channels, samples), DType.F32);
        float* pcmPtr = (float*)pcm.DataPointer;
        for (int c = 0; c < channels; c++)
        {
            float[] source = c == 0 ? edit.Left : edit.Right;
            new ReadOnlySpan<float>(source, 0, samples).CopyTo(new Span<float>(pcmPtr + (long)c * samples, samples));
        }
        Tensor latent;
        try
        {
            device.PreloadWeights(vae.EnumerateWeights());
            latent = vae.EncodeMode(device, pcm);
            device.Sync();
            device.FreeWeights(vae.EnumerateWeights());
        }
        catch (Exception ex)
        {
            Logs.Error($"[Audio][ACE-Step] Oobleck encode of the '{edit.Mode}' source failed.", ex);
            pcm.Dispose();
            throw;
        }
        pcm.Dispose();
        int latentFrames = (int)latent.Shape[2];
        Tensor rows = new Tensor(new TensorShape(1, latentFrames, config.LatentChannels), DType.F32);
        device.Transpose2D(rows, latent, config.LatentChannels, latentFrames);
        latent.Dispose();
        sourceFrames = Math.Min(latentFrames, frames);
        if (latentFrames == frames)
        {
            return rows;
        }
        Tensor fitted = new Tensor(new TensorShape(1, frames, config.LatentChannels), DType.F32);
        float* sourceRows = (float*)rows.DataPointer;
        float* destination = (float*)fitted.DataPointer;
        new Span<float>(destination, (int)fitted.Shape.ElementCount).Clear();
        for (int i = 0; i < frames; i++)
        {
            int take = i < sourceFrames ? i : padWithLastFrame ? sourceFrames - 1 : -1;
            if (take < 0)
            {
                continue;
            }
            Buffer.MemoryCopy(sourceRows + (long)take * config.LatentChannels,
                destination + (long)i * config.LatentChannels, config.LatentChannels * 4, config.LatentChannels * 4);
        }
        rows.Dispose();
        return fitted;
    }

    private const string V1LyricVocabUrl =
        "https://raw.githubusercontent.com/ace-step/ACE-Step/main/acestep/models/lyrics_utils/vocab.json";

    /// <summary>ACE-Step v1 (3.5B) from the Comfy all-in-one single file: DiT + Music-DCAE + ADaMoS vocoder +
    /// UMT5-base all come from the selected checkpoint; only the tiny lyric-tokenizer vocab self-heals from the
    /// upstream repo. The v1 pipeline has no audio-edit or LM-planner path — those requests refuse loudly.</summary>
    private static async Task<IMusicRunner> LoadV1Async(MusicLoadContext context, string mainPath, CancellationToken cancel)
    {
        string lyricVocabPath = AudioModelRoot.SharedFile("acestep_v1_lyric_tokenizer.json");
        await AudioFileFetcher.EnsureAsync(V1LyricVocabUrl, lyricVocabPath, cancel).ConfigureAwait(false);

        (Dictionary<string, Tensor> ditWeights, Dictionary<string, Tensor> dcaeWeights,
            Dictionary<string, Tensor> vocoderWeights, Dictionary<string, Tensor> textWeights,
            SafeTensorsLoader loader) = AceStepCheckpointConverter.LoadV1AllInOne(mainPath);

        AceStepConfig config = new AceStepConfig();
        AceStepDit dit = new AceStepDit(config);
        dit.LoadWeights(ditWeights);
        MusicDcaeDecoder dcae = new MusicDcaeDecoder();
        dcae.LoadWeights(dcaeWeights);
        AdaMosHiFiGanV1 vocoder = new AdaMosHiFiGanV1();
        vocoder.LoadWeights(vocoderWeights);
        T5TextEncoder textEncoder = new T5TextEncoder(T5TextEncoderConfig.Umt5Base);
        textEncoder.LoadWeights(textWeights);
        T5Tokenizer textTokenizer = T5Tokenizer.CreateUmt5(maxLength: 256);
        AceStepLyricTokenizer lyricTokenizer = AceStepLyricTokenizer.FromTokenizerJson(lyricVocabPath);

        AceStepPipeline pipeline = new AceStepPipeline(context.Backend, dit, dcae, vocoder, config);
        Logs.Info($"[Audio][ACE-Step] Loaded v1 3.5B all-in-one '{Path.GetFileName(mainPath)}' "
            + "(44.1 kHz stereo; DiT, DCAE, vocoder and UMT5 all from the selected checkpoint).");

        MusicAudio Synth(IBackend device, MusicRequest request, CancellationToken ct)
        {
            ct.ThrowIfCancellationRequested();
            if (request.Continuation is not null || request.Repaint is not null || request.Cover is not null)
            {
                throw new NotSupportedException(
                    "ACE-Step v1 has no audio-conditioned edit path — continuation/repaint/cover are ACE-Step 1.5 features. "
                    + "Pick a 1.5 checkpoint or remove the Source Audio.");
            }
            if (!string.IsNullOrEmpty(request.LmModel) && request.LmModel != "none")
            {
                throw new NotSupportedException("The 5 Hz LM planner is ACE-Step 1.5 only — set ACE-Step LM Planner to none for v1 checkpoints.");
            }
            string guidanceType = string.IsNullOrEmpty(request.GuidanceType)
                ? (request.UseAdg ? "adg" : "apg")
                : request.GuidanceType.ToLowerInvariant();
            AceStepPipeline.GuidanceMode guidanceMode = guidanceType switch
            {
                "apg" => AceStepPipeline.GuidanceMode.Apg,
                "cfg" => AceStepPipeline.GuidanceMode.Cfg,
                _ => throw new NotSupportedException(
                    $"ACE-Step v1 guidance types are apg and cfg; '{guidanceType}' is ACE-Step 1.5 only."),
            };
            AceStepPipeline.SamplerMode sampler = (request.InferMethod ?? string.Empty).Trim().ToLowerInvariant() switch
            {
                "" or "ode" or "euler" => AceStepPipeline.SamplerMode.Euler,
                "heun" => AceStepPipeline.SamplerMode.Heun,
                "sde" or "pingpong" => AceStepPipeline.SamplerMode.PingPong,
                string other => throw new NotSupportedException(
                    $"Unknown ACE-Step v1 solver '{other}' — expected ode (Euler), heun, or sde (ping-pong)."),
            };

            // Upstream v1 conditions on the raw tag string via UMT5 (no SFT prompt template — that is 1.5's).
            // ERG defaults ON, matching upstream pipeline_ace_step.py.
            bool ergTag = request.UseErgTag ?? true;
            bool ergLyric = request.UseErgLyric ?? true;
            bool ergDiffusion = request.UseErgDiffusion ?? true;
            string tags = string.IsNullOrWhiteSpace(request.Genre) ? "pop" : request.Genre;
            IReadOnlyList<int> rawIds = textTokenizer.EncodeRaw(tags);
            int[] tokens = new int[rawIds.Count + 1];
            for (int i = 0; i < rawIds.Count; i++)
            {
                tokens[i] = rawIds[i];
            }
            tokens[^1] = T5Tokenizer.EosTokenId;
            Tensor textEmbeds;
            Tensor? ergTextEmbeds = null;
            try
            {
                device.PreloadWeights(textEncoder.EnumerateWeights());
                Tensor batch = textEncoder.Encode(device, [tokens], [T5Tokenizer.CreateAttentionMask(tokens)]);
                textEmbeds = CfgHelper.SliceBatchElement(batch, 0, tokens.Length, 768);
                batch.Dispose();
                if (ergTag)
                {
                    Tensor weak = textEncoder.EncodeWithQScale(device, [tokens], (8, 10, 0.01f),
                        [T5Tokenizer.CreateAttentionMask(tokens)]);
                    ergTextEmbeds = CfgHelper.SliceBatchElement(weak, 0, tokens.Length, 768);
                    weak.Dispose();
                }
            }
            finally
            {
                device.Sync();
                device.FreeWeights(textEncoder.EnumerateWeights());
            }
            try
            {
                int[] lyricIds = string.IsNullOrWhiteSpace(request.Prompt)
                    ? []
                    : lyricTokenizer.TokenizeLyrics(request.Prompt,
                        string.IsNullOrWhiteSpace(request.VocalLanguage) ? null : request.VocalLanguage);
                (float[] left, float[] right, int _, int _) = pipeline.Generate(
                    textEmbeds, lyricIds, Math.Clamp(request.Duration, 1d, 240d),
                    steps: request.InferSteps is > 0 ? request.InferSteps : null,
                    guidance: request.CfgScale is > 0 ? (float)request.CfgScale.Value : null,
                    guidanceMode: guidanceMode,
                    sampler: sampler,
                    seed: request.Seed,
                    ergTextEmbeds: ergTextEmbeds,
                    useErgLyric: ergLyric,
                    useErgDiffusion: ergDiffusion);
                return MusicAudio.Stereo(left, right);
            }
            finally
            {
                textEmbeds.Dispose();
                ergTextEmbeds?.Dispose();
            }
        }

        // MusicDcaeDecoder / AdaMosHiFiGanV1 are not IDisposable — their tensors belong to the loader.
        return new MusicRunner(44100, Synth,
            pipeline as IDisposable, dit as IDisposable, textEncoder as IDisposable, loader);
    }

    /// <summary>Per-variant inference defaults, mirroring upstream <c>get_ui_control_config</c>: the turbo family is
    /// 8 steps (shift 3 except the shift-1 checkpoint); sft is 50 and base 32 steps at shift 1.</summary>
    private static (int Steps, float Shift) VariantDefaults(string variant)
    {
        string value = (variant ?? string.Empty).Trim().ToLowerInvariant();
        if (value.Contains("shift1", StringComparison.Ordinal))
        {
            return (8, 1f);
        }
        if (value == "sft" || value.EndsWith("-sft", StringComparison.Ordinal))
        {
            return (50, 1f);
        }
        if (value == "base" || value.EndsWith("-base", StringComparison.Ordinal))
        {
            return (32, 1f);
        }
        return (8, 3f);   // turbo / turbo-shift3 / turbo-continuous / xl-turbo
    }

    /// <summary>Loads the shipped silence latent (fp32 [1, 64, 15000]) into the pipeline's src-latent slot, transposed
    /// to per-frame rows. An absent file keeps the VAE-recompute fallback.</summary>
    private static unsafe void LoadSilenceLatent(AceStepPipeline15 pipeline, string? weightsDirectory)
    {
        try
        {
            string path = Path.Combine(weightsDirectory ?? string.Empty, "acestep-v15-silence_latent.pt");
            if (!File.Exists(path))
            {
                return;
            }
            using PytorchPickleLoader loader = new PytorchPickleLoader();
            loader.Load(path);
            Tensor raw = loader.GetAllTensors().Values.FirstOrDefault()
                ?? throw new InvalidDataException("silence_latent.pt contained no tensor.");
            Tensor f32 = raw.DType == DType.F32 ? raw : raw.CastTo(DType.F32);
            int channels = (int)f32.Shape[1];
            int frames = (int)f32.Shape[2];
            Tensor rows = new Tensor(new TensorShape(frames, channels), DType.F32);
            float* source = (float*)f32.DataPointer;
            float* destination = (float*)rows.DataPointer;
            for (int c = 0; c < channels; c++)
            {
                for (int i = 0; i < frames; i++)
                {
                    destination[(long)i * channels + c] = source[(long)c * frames + i];
                }
            }
            if (!ReferenceEquals(f32, raw))
            {
                f32.Dispose();
            }
            pipeline.SetSilenceLatent(rows);
            Logs.Info($"[Audio][ACE-Step] Loaded shipped silence latent ({frames} frames).");
        }
        catch (Exception ex)
        {
            Logs.Warning($"[Audio][ACE-Step] Could not load silence_latent.pt ({ex.Message}) — using VAE recompute.");
        }
    }

    /// <summary>Decoded editing-mode source plus the resolved timing for one request.</summary>
    private sealed record EditInputs
    {
        /// <summary>"continuation", "repaint", or "cover".</summary>
        public required string Mode { get; init; }

        /// <summary>Source left channel at the VAE's 48 kHz rate.</summary>
        public required float[] Left { get; init; }

        /// <summary>Source right channel (a copy of the left for mono sources).</summary>
        public required float[] Right { get; init; }

        /// <summary>Source clip length in seconds.</summary>
        public required double SourceSeconds { get; init; }

        /// <summary>Total length the DiT generates (source + tail for continuation, source alone otherwise).</summary>
        public required double TotalSeconds { get; init; }

        /// <summary>Repaint span start in seconds (unused for other modes).</summary>
        public required double RepaintStart { get; init; }

        /// <summary>Repaint span end in seconds (unused for other modes).</summary>
        public required double RepaintEnd { get; init; }

        /// <summary>Schedule entry point: 1 for continuation/repaint, the clamped cover strength for cover.</summary>
        public required float StartSigmaFraction { get; init; }
    }

    /// <summary>Lazily loads and caches the 5 Hz LM planner ("0.6b"/"4b") from the official repos; disposed with the runner.</summary>
    private sealed class PlannerHolder : IDisposable
    {
        private readonly List<IDisposable> _loaders = [];
        private AceStepLmPlanner? _planner;
        private string _kind = "";

        public AceStepLmPlanner GetOrLoad(string kind)
        {
            string want = kind.Contains("4b", StringComparison.Ordinal) ? "4b" : "0.6b";
            if (_planner is not null && _kind == want)
            {
                return _planner;
            }
            _planner?.Dispose();
            string repo = want == "4b" ? "ACE-Step/acestep-5Hz-lm-4B" : "ACE-Step/acestep-5Hz-lm-0.6B";
            Logs.Info($"[Audio][ACE-Step] Loading 5 Hz LM planner '{repo}'...");
            (IReadOnlyDictionary<string, Tensor> weights, IDisposable[] loaders) =
                AudioCheckpoints.LoadAsync(repo, "music", CancellationToken.None).ConfigureAwait(false).GetAwaiter().GetResult();
            _loaders.AddRange(loaders);
            _planner = new AceStepLmPlanner(want == "4b" ? AceStepLmPlanner.Config4B : AceStepLmPlanner.Config0_6B, weights);
            _kind = want;
            return _planner;
        }

        public void Dispose()
        {
            _planner?.Dispose();
            foreach (IDisposable loader in _loaders)
            {
                loader.Dispose();
            }
        }
    }
}
