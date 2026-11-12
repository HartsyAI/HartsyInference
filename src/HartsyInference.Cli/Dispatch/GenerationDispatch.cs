using System.Globalization;
using System.Text;
using HartsyInference.Cli.Infra;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Diagnostics;
using HartsyInference.Core.Logging;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Services;
using HartsyInference.Engine.Worlds;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.ThreeD.Io;
using HartsyInference.Vision.Codec;
using HartsyInference.WorldViewer;
using HartsyInference.WorldViewer.Avalonia;
using HartsyInference.Core.Memory.Vram;

namespace HartsyInference.Cli.Dispatch;

/// <summary>Maps the CLI's flag bag onto typed per-modality services and adapts each result into a <see cref="GeneratedArtifact"/>.</summary>
/// <remarks>This is the CLI's only generation entry point: one <c>switch</c> over <see cref="Modality"/>, one typed request built
/// per branch.</remarks>
public static class GenerationDispatch
{

    /// <summary>Runs one generation for <paramref name="spec"/>.</summary>
    /// <remarks><paramref name="prompt"/> is the text prompt for the generative modalities and an input file path for
    /// transcribe / vision / 3d / world. Frame sequences (video, world) are written under <paramref name="outputDir"/>
    /// as they have no single-file artifact.</remarks>
    public static async Task<GeneratedArtifact> RunAsync(
        IInferenceEngine engine,
        ModelSpec spec,
        string prompt,
        ParamState parameters,
        string? outputDir,
        bool quiet,
        CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(engine);
        ArgumentNullException.ThrowIfNull(spec);
        ArgumentNullException.ThrowIfNull(parameters);
        return spec.Modality switch
        {
            Modality.Image => await ImageAsync(engine, spec, prompt, parameters, quiet, cancel).ConfigureAwait(false),
            Modality.Text => await TextAsync(engine, spec, prompt, parameters, quiet, cancel).ConfigureAwait(false),
            Modality.Speech => await SpeechAsync(engine, spec, prompt, parameters, cancel).ConfigureAwait(false),
            Modality.Music => await MusicAsync(engine, spec, prompt, parameters, quiet, cancel).ConfigureAwait(false),
            Modality.Transcribe => await TranscribeAsync(engine, spec, prompt, parameters, cancel).ConfigureAwait(false),
            Modality.Vision => await VisionAsync(engine, spec, prompt, parameters, outputDir, cancel).ConfigureAwait(false),
            Modality.Video => await VideoAsync(engine, spec, prompt, parameters, outputDir, quiet, cancel).ConfigureAwait(false),
            Modality.Mesh => await MeshAsync(engine, spec, prompt, parameters, quiet, cancel).ConfigureAwait(false),
            Modality.World => await WorldAsync(engine, spec, prompt, parameters, outputDir, quiet, cancel).ConfigureAwait(false),
            Modality.VoiceConvert => await VoiceConvertAsync(engine, spec, prompt, parameters, cancel).ConfigureAwait(false),
            Modality.Fx => await FxAsync(engine, spec, prompt, parameters, outputDir, cancel).ConfigureAwait(false),
            Modality.Restore => await RestoreAsync(engine, spec, prompt, parameters, outputDir, quiet, cancel).ConfigureAwait(false),
            _ => throw new NotSupportedException($"The '{Modalities.ToCliName(spec.Modality)}' modality has no CLI dispatch."),
        };
    }

    /// <summary>Text-to-image through <see cref="IImagesService"/>; the RGB result is PNG-encoded for saving and kept raw for preview.</summary>
    private static async Task<GeneratedArtifact> ImageAsync(
        IInferenceEngine engine, ModelSpec spec, string prompt, ParamState parameters, bool quiet, CancellationToken cancel)
    {
        // Every tunable the user did not set stays null so the engine fills it from the family's official defaults.
        ImageRequest request = new ImageRequest
        {
            Prompt = prompt,
            NegativePrompt = parameters.Get("negative"),
            Width = parameters.GetIntOrNull("width"),
            Height = parameters.GetIntOrNull("height"),
            Steps = parameters.GetIntOrNull("steps"),
            CfgScale = parameters.GetFloatOrNull("cfg"),
            FluxGuidance = parameters.GetFloatOrNull("flux-guidance"),
            TrueCfgScale = parameters.GetFloatOrNull("true-cfg"),
            Sampler = parameters.GetStringOrNull("sampler"),
            Loras = BuildLoraStack(parameters),
            Scheduler = parameters.GetStringOrNull("scheduler"),
            SigmaShift = parameters.GetDoubleOrNull("sigma-shift"),
            Vram = ParseVramOverrides(parameters),
            Seed = parameters.GetInt("seed", -1),
            InstructPix2PixCfg = parameters.GetDoubleOrNull("ip2p-cfg"),
            RemoveBackground = parameters.GetBool("remove-background", false) ? true : null,
            // SwarmUI's own default is on (UserInput.Get(T2IParamTypes.ModelSpecificEnhancements, true)), so the
            // flag turns the extras OFF rather than on.
            ModelSpecificEnhancements = !parameters.GetBool("no-model-enhancements", false),
            Img2Img = BuildImg2Img(parameters, spec),
            Inpaint = BuildInpaint(parameters),
            IpAdapter = SplitPaths(parameters.GetStringOrNull("prompt-images")) is { Length: > 0 } promptPaths
                ? new IpAdapter { PromptImages = [.. promptPaths.Select(LoadImage)] } : null,
            ReferenceImages = SplitPaths(parameters.GetStringOrNull("reference-images")) is { Length: > 0 } refPaths
                ? [.. refPaths.Select(LoadImage)] : null,
            Components = BuildImageComponents(parameters),
            Extra = BuildImageExtra(parameters),
        };

        ConsoleStepProgress? progress = quiet ? null
            : new ConsoleStepProgress("denoise", parameters.Get("preview-output"));
        ImageResult result;
        try
        {
            result = await engine.Images.GenerateAsync(spec, request, progress, cancel).ConfigureAwait(false);
        }
        finally
        {
            progress?.Finish();
        }

        GeneratedArtifact artifact = new GeneratedArtifact
        {
            Kind = ArtifactKind.Image,
            FileBytes = PngEncoder.Encode(result.Rgb, result.HasAlpha ? result.Alpha : null, result.Width, result.Height),
            Extension = "png",
            Text = $"{result.Width}x{result.Height} image (seed {result.Seed})"
                + (result.Layers is { Count: > 0 } layers ? $", {layers.Count} layers" : ""),
            PreviewRgb = result.Rgb,
            PreviewWidth = result.Width,
            PreviewHeight = result.Height,
            Companions = LayerCompanions(result),
        };
        artifact.Meta["size"] = $"{result.Width}x{result.Height}";
        artifact.Meta["seed"] = result.Seed.ToString(CultureInfo.InvariantCulture);
        // The step count reported is the one the pipeline actually ran (its own meta), since an unset --steps resolves
        // to the family's official default inside the engine.
        CopyMeta(result.Meta, artifact);
        return artifact;
    }

    /// <summary>A layered result's layers as RGBA PNG companions <c>layer-00.png</c> (bottom) onward, written next to the
    /// composite; empty for a single-layer result.</summary>
    internal static IReadOnlyList<(string Suffix, byte[] Bytes)> LayerCompanions(ImageResult result) =>
        result.Layers is not { Count: > 0 } layers ? []
            : layers.Select(l => ($"layer-{l.Index:D2}.png", PngEncoder.Encode(l.Rgb, l.Alpha, l.Width, l.Height))).ToList();

    /// <summary>The token ids of a <c>--scores-out</c> TSV, in order (its <c>token_id</c> column).</summary>
    private static int[] ReadScoredTokenIds(string path)
    {
        string[] lines = File.ReadAllLines(path);
        if (lines.Length < 2 || lines[0].Split('\t') is not [_, "token_id", ..])
            throw new FormatException($"{path} is not a --scores-out TSV (header 'index\ttoken_id\t...').");
        return [.. lines.Skip(1).Where(l => l.Length > 0).Select(l => int.Parse(l.Split('\t')[1], CultureInfo.InvariantCulture))];
    }

    /// <summary>Writes one row per completion token; log-probabilities round-trip exactly ("R"), so two runs compare bit for bit.</summary>
    private static void WriteTokenScores(string path, IReadOnlyList<TokenLogProb> scores)
    {
        StringBuilder sb = new("index\ttoken_id\tlogprob\ttop_id\ttop_logprob\n");
        for (int i = 0; i < scores.Count; i++)
        {
            TokenLogProb s = scores[i];
            sb.Append(CultureInfo.InvariantCulture, $"{i}\t{s.TokenId}\t{s.LogProb.ToString("R", CultureInfo.InvariantCulture)}\t{s.TopTokenId}\t{s.TopLogProb.ToString("R", CultureInfo.InvariantCulture)}\n");
        }
        File.WriteAllText(path, sb.ToString());
    }

    /// <summary>Chat completion through <see cref="ITextService"/>; streams live unless <c>--quiet</c> takes the one-shot path.</summary>
    private static async Task<GeneratedArtifact> TextAsync(
        IInferenceEngine engine, ModelSpec spec, string prompt, ParamState parameters, bool quiet, CancellationToken cancel)
    {
        if (parameters.GetBool("realtime", false))
            return await RealtimeDispatch.RunAsync(engine, spec, prompt, parameters, quiet, cancel).ConfigureAwait(false);
        float temperature = parameters.GetFloat("temperature", 0.7f);
        List<ImageData> images = [];
        if (parameters.GetStringOrNull("image") is { Length: > 0 } imagePaths)
        {
            foreach (string path in imagePaths.Split('\n', StringSplitOptions.RemoveEmptyEntries))
                images.Add(LoadImage(path));
        }
        string? scoresOut = parameters.GetStringOrNull("scores-out");
        IReadOnlyList<int>? forced = parameters.GetStringOrNull("teacher-force") is { Length: > 0 } forcePath ? ReadScoredTokenIds(forcePath) : null;
        TextRequest request = new TextRequest
        {
            // "cuda:0" pins an ordinal; the composite "cuda:0+cuda:1" form requests the LLM layer split
            // (TextService.LoadSharded) — layers pooled across both cards by free VRAM.
            Device = parameters.GetStringOrNull("device"),
            Messages =
            [
                new TextMessage
                {
                    Role = TextRole.User, Content = prompt, Images = images.Count > 0 ? images : null,
                    Video = parameters.GetStringOrNull("video") is { } videoPath
                        ? new VideoClip { Data = File.ReadAllBytes(videoPath), Format = Path.GetExtension(videoPath).TrimStart('.') }
                        : null,
                    VideoFrames = parameters.GetStringOrNull("frames-dir") is { } framesDir
                        ? FolderFrames(framesDir, parameters.GetDoubleOrNull("frame-fps") ?? 1.0)
                        : null,
                },
            ],
            VideoFps = parameters.GetDoubleOrNull("video-fps"),
            VideoMaxFrames = parameters.GetIntOrNull("video-max-frames"),
            SystemPrompt = parameters.GetStringOrNull("system"),
            MaxTokens = forced?.Count ?? parameters.GetInt("max-tokens", 256),
            ForcedTokenIds = forced,
            ReturnTokenScores = scoresOut is not null,
            Temperature = temperature,
            TopP = parameters.GetFloat("top-p", 0.95f),
            TopK = parameters.GetIntOrNull("top-k"),
            MinP = parameters.GetDoubleOrNull("min-p"),
            RepetitionPenalty = parameters.GetDoubleOrNull("repetition-penalty"),
            Seed = parameters.GetInt("seed", -1),
            Greedy = temperature <= 0f,
            GraphDecode = parameters.GetBool("graph-decode", false) ? true : null,
            EnableThinking = parameters.GetStringOrNull("thinking") is { Length: > 0 } thinking ? bool.Parse(thinking) : null,
            ReasoningEffort = ParseReasoningEffort(parameters.GetStringOrNull("reasoning-effort")),
            Loops = parameters.GetIntOrNull("loops"),
            // TextService.LoadInto only checks LowVramQuant for non-empty (a bool-shaped toggle, not an actual
            // target quant selector), so any non-empty sentinel works here — see its doc comment.
            LowVramQuant = parameters.GetBool("low-vram-quant", false) ? "on" : null,
            WeightPrecision = parameters.GetStringOrNull("precision"),
            AlwaysFreeMemory = parameters.GetBool("always-free-memory", false) ? true : null,
            Placement = parameters.GetStringOrNull("placement"),
        };

        StringBuilder text = new StringBuilder();
        StopReason stop = StopReason.Stop;
        // Decode rate over the streamed pieces after the first (the first carries prefill), for the regression gate.
        System.Diagnostics.Stopwatch decodeClock = new();
        int pieces = 0;
        if (quiet || scoresOut is not null)
        {
            // Scores ride the one-shot result, so a scored run takes this path even without --quiet.
            TextResult result = await engine.Text.GenerateAsync(spec, request, cancel).ConfigureAwait(false);
            text.Append(result.Text);
            stop = result.Stop;
            if (scoresOut is not null)
                WriteTokenScores(scoresOut, result.TokenScores ?? []);
        }
        else
        {
            await foreach (TextChunk chunk in engine.Text.StreamAsync(spec, request, cancel).ConfigureAwait(false))
            {
                switch (chunk.Kind)
                {
                    case TextChunkKind.Chunk when chunk.Text is { Length: > 0 } piece:
                        text.Append(piece);
                        Console.Write(piece);
                        if (pieces++ == 0) decodeClock.Start();
                        break;
                    case TextChunkKind.Result when chunk.Text is not null:
                        text.Clear();
                        text.Append(chunk.Text);
                        break;
                    case TextChunkKind.StopReason when chunk.Stop is { } reason:
                        stop = reason;
                        break;
                    default:
                        break;
                }
            }
        }

        GeneratedArtifact artifact = new GeneratedArtifact
        {
            Kind = ArtifactKind.Text,
            Text = text.ToString(),
            Extension = "txt",
            Streamed = !quiet,
        };
        artifact.Meta["stopped_on"] = stop.ToString().ToLowerInvariant();
        if (pieces > 1 && decodeClock.Elapsed.TotalSeconds > 0)
            artifact.Meta["decode_tok_s"] = ((pieces - 1) / decodeClock.Elapsed.TotalSeconds).ToString("F1", CultureInfo.InvariantCulture);
        return artifact;
    }

    /// <summary>Text-to-speech through <see cref="ISpeechService"/>; cloning models read an optional <c>reference</c> path.</summary>
    /// <remarks>F5-style models also read <c>ref-text</c> alongside <c>reference</c>. Fish Audio Drama 3 reads <c>references</c> and
    /// <c>history</c> turn files (<see cref="SpeechTurnFile"/>) and writes the take's turns to <c>save-turns</c>.</remarks>
    private static async Task<GeneratedArtifact> SpeechAsync(
        IInferenceEngine engine, ModelSpec spec, string prompt, ParamState parameters, CancellationToken cancel)
    {
        double speed = parameters.GetFloat("speed", 1f);
        SpeechRequest request = new SpeechRequest
        {
            Vram = ParseVramOverrides(parameters),
            Text = prompt,
            Voice = parameters.Get("voice"),
            Speed = speed > 0d ? speed : null,
            Reference = LoadAudioClip(parameters.GetStringOrNull("reference")),
            RefText = parameters.Get("ref-text") ?? "",
            References = SpeechTurnFile.Load(parameters.GetStringOrNull("references")),
            History = SpeechTurnFile.Load(parameters.GetStringOrNull("history")),
            Exaggeration = parameters.GetDoubleOrNull("exaggeration"),
            NfeStep = parameters.GetIntOrNull("nfe-step"),
            CfgScale = parameters.GetDoubleOrNull("cfg-scale"),
            Instruction = parameters.GetStringOrNull("instruction"),
            DurationSeconds = parameters.GetDoubleOrNull("duration"),
            Emotion = ParseEmotionVector(parameters.GetStringOrNull("emotion")),
            EmotionReference = LoadAudioClip(parameters.GetStringOrNull("emotion-reference")),
            EmotionAlpha = parameters.GetDoubleOrNull("emotion-alpha"),
            EmotionText = parameters.GetStringOrNull("emotion-text"),
            EmotionFromText = parameters.GetBool("emotion-from-text", false),
            Seed = Math.Max(0, parameters.GetInt("seed", 0)),
            ChunkWords = parameters.GetIntOrNull("chunk-words"),
            Temperature = parameters.GetDoubleOrNull("temperature"),
            TopP = parameters.GetDoubleOrNull("top-p"),
            TopK = parameters.GetIntOrNull("top-k"),
            MaxTokens = parameters.GetIntOrNull("max-tokens"),
            NormalizeLoudness = parameters.GetBool("no-normalize", false) ? false : null,
        };
        AudioResult result = await engine.Speech.SynthesizeAsync(spec, request, cancel).ConfigureAwait(false);
        GeneratedArtifact artifact = AudioArtifact(result, "speech");
        if (parameters.GetStringOrNull("save-turns") is { } turnsPath)
        {
            if (result.Turns is not { Count: > 0 } turns)
            {
                throw new InvalidOperationException("--save-turns needs a model that returns its turns (Fish Audio Drama 3).");
            }
            SpeechTurnFile.Save(turnsPath, turns);
            artifact.Meta["turns"] = turnsPath;
        }
        return artifact;
    }

    /// <summary>The <c>--reasoning-effort</c> value, read by <see cref="EncodeOptions.ParseReasoningEffort"/> as the chat API reads it: an integer in [1, 100], or a
    /// name the engine defines (<c>low</c>, <c>high</c>, <c>max</c>). Null when unset; <see cref="ArgumentException"/> for anything else.</summary>
    internal static int? ParseReasoningEffort(string? value) => value is { Length: > 0 } text ? EncodeOptions.ParseReasoningEffort(text) : null;

    /// <summary>Parses <c>--emotion</c>'s comma-separated weights; null when unset, <see cref="ArgumentException"/> unless exactly 8 numbers.</summary>
    internal static double[]? ParseEmotionVector(string? csv)
    {
        if (string.IsNullOrWhiteSpace(csv)) return null;
        string[] parts = csv.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
        double[] values = new double[parts.Length];
        for (int i = 0; i < parts.Length; i++)
        {
            if (!double.TryParse(parts[i], NumberStyles.Float, CultureInfo.InvariantCulture, out values[i]))
                throw new ArgumentException($"--emotion value '{parts[i]}' is not a number.");
        }
        if (values.Length != 8)
            throw new ArgumentException($"--emotion needs exactly 8 comma-separated weights, got {values.Length}.");
        return values;
    }

    /// <summary>Voice conversion through <see cref="IVoiceConversionService"/>; the CLI's "prompt" is the source audio path.</summary>
    /// <remarks>An optional <c>target-path</c> tunable supplies the tone-color reference (OpenVoice); RVC ignores it.</remarks>
    private static async Task<GeneratedArtifact> VoiceConvertAsync(
        IInferenceEngine engine, ModelSpec spec, string prompt, ParamState parameters, CancellationToken cancel)
    {
        VoiceConversionRequest request = new VoiceConversionRequest
        {
            Source = LoadAudioClip(prompt) ?? throw new FileNotFoundException($"Source audio not found: {prompt}"),
            Target = LoadAudioClip(parameters.GetStringOrNull("target-path")),
            PitchShift = parameters.GetFloat("pitch-shift", 0f),
        };
        AudioResult result = await engine.VoiceConversion.ConvertAsync(spec, request, cancel).ConfigureAwait(false);
        return AudioArtifact(result, "converted");
    }

    /// <summary>Audio effects (Demucs / Resemble-Enhance) through <see cref="IFxService"/>; the CLI's "prompt" is the input audio path.</summary>
    /// <remarks>The <c>mode</c> tunable picks the operation. Separation yields multiple stems, written directly to a fresh
    /// subfolder since there is no single-file result.</remarks>
    private static async Task<GeneratedArtifact> FxAsync(
        IInferenceEngine engine, ModelSpec spec, string prompt, ParamState parameters, string? outputDir, CancellationToken cancel)
    {
        AudioClip audio = LoadAudioClip(prompt) ?? throw new FileNotFoundException($"Input audio not found: {prompt}");
        string mode = (parameters.Get("mode") ?? "separate").ToLowerInvariant();
        if (mode == "enhance")
        {
            FxEnhanceRequest request = new FxEnhanceRequest
            {
                Audio = audio,
                Lambd = parameters.GetDoubleOrNull("lambda"),
                Tau = parameters.GetDoubleOrNull("tau"),
                Seed = parameters.GetInt("seed", -1),
            };
            AudioResult result = await engine.Fx.EnhanceAsync(spec, request, cancel).ConfigureAwait(false);
            return AudioArtifact(result, "enhanced");
        }
        // Leave Model null so the service falls back to the selector's variant (parsed from "demucs:htdemucs_6s"
        // style catalog ids) rather than the raw requested token.
        FxSeparateRequest separateRequest = new FxSeparateRequest { Audio = audio };
        StemsResult stems = await engine.Fx.SeparateAsync(spec, separateRequest, cancel).ConfigureAwait(false);
        return StemsArtifact(stems, outputDir, Path.GetFileNameWithoutExtension(prompt.Trim().Trim('"')));
    }

    /// <summary>Text-to-music through <see cref="IMusicService"/>.</summary>
    private static async Task<GeneratedArtifact> MusicAsync(
        IInferenceEngine engine, ModelSpec spec, string prompt, ParamState parameters, bool quiet, CancellationToken cancel)
    {
        // Everything past the first four is optional: a null means "the model applies its own default", so an
        // unset flag must NOT be materialised into a value here.
        MusicRequest request = new MusicRequest
        {
            Vram = ParseVramOverrides(parameters),
            Prompt = prompt,
            Genre = parameters.Get("genre"),
            Duration = parameters.GetIntOrNull("duration"),
            Seed = Math.Max(0, parameters.GetInt("seed", 0)),
            InferSteps = parameters.GetIntOrNull("steps"),
            CfgScale = parameters.GetDoubleOrNull("cfg-scale"),
            Temperature = parameters.GetDoubleOrNull("temperature"),
            TopK = parameters.GetIntOrNull("top-k"),
            TopP = parameters.GetDoubleOrNull("top-p"),
            RepetitionPenalty = parameters.GetDoubleOrNull("repetition-penalty"),
            Yue2Cot = parameters.Get("cot"),
            Yue2Abc = parameters.Get("abc"),
            Yue2AbcTemperature = parameters.GetDoubleOrNull("abc-temperature"),
            Yue2AbcTopP = parameters.GetDoubleOrNull("abc-top-p"),
            Yue2AbcTopK = parameters.GetIntOrNull("abc-top-k"),
            Yue2AbcRepetitionPenalty = parameters.GetDoubleOrNull("abc-repetition-penalty"),
            Yue2AbcMaxTokens = parameters.GetIntOrNull("abc-max-tokens"),
            Yue2PenaltyWindow = parameters.GetIntOrNull("penalty-window"),
            Yue2MinTokens = parameters.GetIntOrNull("min-tokens"),
        };
        // The score is the editable half of the song and costs seconds against minutes for the render, so it is
        // worth asking for alone: write it out, edit it, feed it back with --abc.
        if (parameters.GetBool("score-only", false))
        {
            ScorePlanResult plan = await engine.Music.PlanScoreAsync(spec, request, cancel).ConfigureAwait(false);
            GeneratedArtifact score = new GeneratedArtifact
            {
                Kind = ArtifactKind.Text,
                Text = plan.Abc,
                Extension = "abc",
            };
            score.Meta["score tokens"] = plan.ScoreTokens.ToString(CultureInfo.InvariantCulture);
            score.Meta["prefix tokens"] = plan.PrefixTokens.ToString(CultureInfo.InvariantCulture);
            score.Meta["audio budget"] = $"{plan.BudgetSeconds:0.0}s";
            if (plan.Truncated)
            {
                score.Meta["truncated"] = "the score hit its token ceiling";
            }
            return score;
        }
        ConsoleStepProgress? progress = quiet ? null : new ConsoleStepProgress("generate");
        AudioResult result;
        try
        {
            result = await engine.Music.GenerateAsync(spec, request, progress, cancel).ConfigureAwait(false);
        }
        finally
        {
            progress?.Finish();
        }
        return AudioArtifact(result, "music");
    }

    /// <summary>Speech-to-text through <see cref="ITranscribeService"/>; "prompt" is the audio path, loaded into <see cref="AudioClip"/>.</summary>
    private static async Task<GeneratedArtifact> TranscribeAsync(
        IInferenceEngine engine, ModelSpec spec, string prompt, ParamState parameters, CancellationToken cancel)
    {
        string path = prompt.Trim().Trim('"');
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"Audio file not found: {path}");
        }
        AudioRequest request = new AudioRequest
        {
            Audio = new AudioClip
            {
                Data = await File.ReadAllBytesAsync(path, cancel).ConfigureAwait(false),
                Format = Path.GetExtension(path).TrimStart('.').ToLowerInvariant(),
            },
            Language = parameters.Get("language") is { Length: > 0 } language ? language : "en",
            Translate = parameters.GetBool("translate", false),
            WordTimestamps = parameters.GetBool("timestamps", false),
        };
        TranscriptResult result = await engine.Transcribe.RunAsync(spec, request, cancel).ConfigureAwait(false);

        GeneratedArtifact artifact = new GeneratedArtifact
        {
            Kind = ArtifactKind.Text,
            // Words is populated only when the request asked for word timestamps or diarization: one
            // "[start --> end]" line per word or segment then, the plain transcript otherwise.
            Text = result.Words is { Count: > 0 } timedWords ? FormatTimestamps(timedWords) : result.Text.Trim(),
            Extension = "txt",
        };
        artifact.Meta["audio"] = path;
        artifact.Meta["language"] = result.Language;
        if (result.Words is { Count: > 0 } words)
        {
            artifact.Meta["segments"] = words.Count.ToString(CultureInfo.InvariantCulture);
        }
        return artifact;
    }

    /// <summary>One <c>[start --> end]  text</c> line per word/segment, seconds to 3 decimals.</summary>
    internal static string FormatTimestamps(IReadOnlyList<WordSegment> words)
    {
        StringBuilder sb = new();
        foreach (WordSegment w in words)
        {
            string word = w.Word.Trim();
            if (word.Length == 0)
            {
                continue;
            }
            sb.Append('[').Append(w.Start.ToString("F3", CultureInfo.InvariantCulture)).Append("s --> ")
              .Append(w.End.ToString("F3", CultureInfo.InvariantCulture)).Append("s]  ").Append(word).Append('\n');
        }
        return sb.ToString().TrimEnd('\n');
    }

    /// <summary>Embed / detect / segment through <see cref="IVisionService"/>; the CLI's "prompt" is the image path.</summary>
    /// <remarks>Segment masks have no meaningful text form, so they self-write as PNGs under <paramref name="outputDir"/>
    /// (mirroring the video frame-sequence pattern) — otherwise the mask is invisible to inspection.</remarks>
    private static async Task<GeneratedArtifact> VisionAsync(
        IInferenceEngine engine, ModelSpec spec, string prompt, ParamState parameters, string? outputDir, CancellationToken cancel)
    {
        VisionMode visionMode = ParseVisionMode(parameters.Get("mode") ?? "embed");
        string inputPath = prompt.Trim().Trim('"');
        if (Directory.Exists(inputPath))
        {
            return await VisionBatchAsync(engine, spec, inputPath, visionMode, parameters, outputDir, cancel).ConfigureAwait(false);
        }
        ImageData image = LoadImage(prompt);
        VisionRequest request = new VisionRequest
        {
            Image = image,
            Mode = visionMode,
            Prompt = parameters.Get("query"),
            Threshold = parameters.GetFloat("confidence", 0.25f),
            TargetWidth = parameters.GetIntOrNull("width"),
            TargetHeight = parameters.GetIntOrNull("height"),
        };
        VisionRequestValidator.Validate(request);
        VisionResult result = await engine.Vision.RunAsync(spec, request, cancel).ConfigureAwait(false);

        if (VisionOutputWriter.HasStructured(result))
        {
            return StructuredVisionArtifact(result, outputDir, inputPath);
        }

        if (result.Masks is { Count: > 0 } masks)
        {
            byte[][] pixels = new byte[masks.Count][];
            for (int i = 0; i < masks.Count; i++)
            {
                pixels[i] = masks[i].Rgb;
            }
            string dir = FrameWriter.WriteFrames(pixels, masks[0].Width, masks[0].Height, outputDir ?? RepoPaths.OutputRoot(), prompt);
            GeneratedArtifact maskArtifact = new GeneratedArtifact
            {
                Kind = ArtifactKind.Data,
                Extension = "png",
                Text = $"{masks.Count} mask(s) at {masks[0].Width}x{masks[0].Height} → {dir}",
                PreviewRgb = masks[0].Rgb,
                PreviewWidth = masks[0].Width,
                PreviewHeight = masks[0].Height,
                SelfWritten = true,
            };
            maskArtifact.Meta["masks"] = masks.Count.ToString(CultureInfo.InvariantCulture);
            return maskArtifact;
        }

        if (result.Image is { } single)
        {
            GeneratedArtifact imageArtifact = new GeneratedArtifact
            {
                Kind = ArtifactKind.Image,
                // Alpha-aware: a background-removal result writes an RGBA PNG, everything else stays RGB.
                FileBytes = PngEncoder.Encode(single),
                Extension = "png",
                Text = $"{single.Width}x{single.Height} image{(single.HasAlpha ? " with alpha" : "")}",
                PreviewRgb = single.Rgb,
                PreviewWidth = single.Width,
                PreviewHeight = single.Height,
            };
            imageArtifact.Meta["size"] = $"{single.Width}x{single.Height}";
            return imageArtifact;
        }

        GeneratedArtifact artifact = new GeneratedArtifact
        {
            Kind = ArtifactKind.Data,
            Extension = "txt",
            Text = Describe(result),
        };
        if (result.Embedding is { Length: > 0 } embedding)
        {
            artifact.Meta["dim"] = embedding.Length.ToString(CultureInfo.InvariantCulture);
        }
        if (result.Detections is not null)
        {
            artifact.Meta["detections"] = result.Detections.Count.ToString(CultureInfo.InvariantCulture);
        }
        return artifact;
    }

    /// <summary>The CLI's <c>--mode</c> names for each <see cref="VisionMode"/>.</summary>
    internal static VisionMode ParseVisionMode(string mode) => mode.ToLowerInvariant() switch
    {
        "detect" => VisionMode.Detect,
        "segment" => VisionMode.Segment,
        "embed" => VisionMode.Embed,
        "depth" => VisionMode.Depth,
        "edge" => VisionMode.Edge,
        "lineart" => VisionMode.Lineart,
        "normal" => VisionMode.Normal,
        "segmap" => VisionMode.SegMap,
        "removebg" => VisionMode.BackgroundRemoval,
        "upscale" => VisionMode.Upscale,
        "metric-depth" => VisionMode.MetricDepth,
        "geometry" => VisionMode.Geometry,
        "pose" => VisionMode.Pose,
        "matting" => VisionMode.Matting,
        "face" => VisionMode.Face,
        "intrinsic" => VisionMode.Intrinsic,
        "track" => VisionMode.Track,
        "interpolate" => VisionMode.Interpolate,
        _ => throw new ArgumentException(
            $"Unknown vision mode '{mode}'. Use embed, detect, segment, depth, edge, lineart, normal, segmap, removebg, upscale, "
            + "metric-depth, geometry, pose, matting, face, intrinsic, track, or interpolate."),
    };

    /// <summary>A folder input: its images (sorted by name) become the frames of a track or interpolate run, or the views
    /// of a multi-view depth/geometry run. Frame runs need <c>--fps</c>; interpolation needs <c>--factor</c> or
    /// <c>--target-fps</c>; tracking takes its text prompt from <c>--prompt</c> on the first frame.</summary>
    private static async Task<GeneratedArtifact> VisionBatchAsync(IInferenceEngine engine, ModelSpec spec, string folder,
        VisionMode mode, ParamState parameters, string? outputDir, CancellationToken cancel)
    {
        List<ImageData> images = VisionBatchInputs(folder).Select(LoadImage).ToList();
        bool frames = VisionRequestValidator.IsSequenceOnly(mode);
        string? query = parameters.Get("query");
        VisionBatchRequest request = new VisionBatchRequest
        {
            Images = images,
            Mode = mode,
            Kind = frames ? VisionSequenceKind.Frames : VisionSequenceKind.Views,
            Fps = frames ? parameters.GetDoubleOrNull("fps") : null,
            InterpolationFactor = parameters.GetIntOrNull("factor"),
            TargetFps = parameters.GetDoubleOrNull("target-fps"),
            Prompts = mode == VisionMode.Track && !string.IsNullOrWhiteSpace(query)
                ? [new VisionFramePrompt { FrameIndex = 0, ObjectId = 1, Text = query }]
                : null,
            Threshold = parameters.GetFloat("confidence", 0.25f),
        };
        VisionRequestValidator.Validate(request);
        string dir = VisionOutputWriter.NewRunDirectory(outputDir ?? RepoPaths.OutputRoot(), folder);
        if (!frames)
        {
            VisionResult result = await engine.Vision.RunViewsAsync(spec, request, cancel).ConfigureAwait(false);
            IReadOnlyList<string> files = VisionOutputWriter.WriteResult(result, dir);
            return SelfWrittenVisionArtifact($"{images.Count} views → {files.Count} file(s) in {dir}", files.Count);
        }
        int written = await VisionOutputWriter.WriteFramesAsync(engine.Vision.RunFramesAsync(spec, request, cancel), dir, cancel)
            .ConfigureAwait(false);
        return SelfWrittenVisionArtifact($"{images.Count} input frames → {written} output frame(s) in {dir}", written);
    }

    /// <summary>A folder of PNG/BMP frames as a timed video: name order, frame <c>i</c> at <c>i / fps</c> seconds.</summary>
    internal static IReadOnlyList<TimedImage> FolderFrames(string folder, double fps) =>
        [.. VisionBatchInputs(folder).Select((path, i) => new TimedImage { Image = LoadImage(path), TimestampSeconds = i / fps })];

    /// <summary>The images a folder input contributes, sorted by name: the formats <see cref="ImageIo.DecodeFile"/> reads
    /// (.png, .bmp). Other files are skipped rather than failing the batch.</summary>
    internal static string[] VisionBatchInputs(string folder)
    {
        string[] paths = Directory.EnumerateFiles(folder)
            .Where(p => p.EndsWith(".png", StringComparison.OrdinalIgnoreCase) || p.EndsWith(".bmp", StringComparison.OrdinalIgnoreCase))
            .Order(StringComparer.Ordinal)
            .ToArray();
        return paths.Length > 0 ? paths : throw new ArgumentException($"No .png or .bmp images in '{folder}'.");
    }

    /// <summary>Writes a single-image structured result (depth, geometry, pose, faces, matting, intrinsics) to a fresh run
    /// folder and reports it.</summary>
    private static GeneratedArtifact StructuredVisionArtifact(VisionResult result, string? outputDir, string inputPath)
    {
        string dir = VisionOutputWriter.NewRunDirectory(outputDir ?? RepoPaths.OutputRoot(), inputPath);
        IReadOnlyList<string> files = VisionOutputWriter.WriteResult(result, dir);
        return SelfWrittenVisionArtifact($"{files.Count} file(s) in {dir}", files.Count);
    }

    private static GeneratedArtifact SelfWrittenVisionArtifact(string text, int count)
    {
        GeneratedArtifact artifact = new GeneratedArtifact { Kind = ArtifactKind.Data, Extension = "json", Text = text, SelfWritten = true };
        artifact.Meta["files"] = count.ToString(CultureInfo.InvariantCulture);
        return artifact;
    }

    /// <summary>Text-to-video through <see cref="IVideoService"/>; frames stream out and are written as a numbered PNG sequence.</summary>
    private static async Task<GeneratedArtifact> VideoAsync(
        IInferenceEngine engine, ModelSpec spec, string prompt, ParamState parameters, string? outputDir, bool quiet, CancellationToken cancel)
    {
        VideoRequest request = new VideoRequest
        {
            Vram = ParseVramOverrides(parameters),
            Prompt = prompt,
            NegativePrompt = parameters.Get("negative") is { Length: > 0 } negative ? negative
                : "blurry, low quality, distorted, watermark",
            Width = parameters.GetIntOrNull("width"),
            Height = parameters.GetIntOrNull("height"),
            Steps = parameters.GetIntOrNull("steps"),
            CfgScale = parameters.GetFloatOrNull("cfg"),
            FlowShift = parameters.GetFloatOrNull("flow-shift"),
            AudioFlowShift = parameters.GetFloatOrNull("audio-flow-shift"),
            Sampler = parameters.GetStringOrNull("sampler"),
            Scheduler = parameters.GetStringOrNull("scheduler"),
            Frames = parameters.GetIntOrNull("frames"),
            ChainTotalFrames = ResolveChainTotalFrames(parameters),
            ChainContextFrames = parameters.GetIntOrNull("chain-context-frames")
                ?? MiniMaxH3ChainPlanner.DefaultContextFrames,
            Fps = parameters.GetIntOrNull("fps"),
            Seed = parameters.GetInt("seed", -1),
            InitImage = parameters.GetStringOrNull("init-image") is { } initPath ? LoadImage(initPath) : null,
            VideoEndFrame = parameters.GetStringOrNull("end-frame") is { } endPath ? LoadImage(endPath) : null,
            DrivingVideo = parameters.GetStringOrNull("driving-video") is { } drivingPath ? LoadVideoClip(drivingPath) : null,
            DrivingPoseVideo = parameters.GetStringOrNull("pose-video") is { } posePath ? LoadVideoClip(posePath) : null,
            DrivingFaceVideo = parameters.GetStringOrNull("face-video") is { } facePath ? LoadVideoClip(facePath) : null,
            DrivingAutoPreprocess = !parameters.GetBool("no-auto-preprocess", false),
            VideoSwapModel = parameters.GetStringOrNull("swap-model"),
            VideoSwapPercent = parameters.GetDoubleOrNull("swap-percent"),
            ReferenceImages = SplitPaths(parameters.GetStringOrNull("ref-images"))?.Select(LoadImage).ToList(),
            ReferenceVideos = BuildReferenceVideos(parameters),
            Loras = BuildLoraStack(parameters),
            ReferenceAudios = SplitPaths(parameters.GetStringOrNull("ref-audios"))
                ?.Select(p => LoadAudioClip(p) ?? throw new FileNotFoundException($"Reference audio not found: {p}"))
                .ToList(),
            VideoAudioReference = parameters.GetStringOrNull("driving-audio") is string drivingAudio
                ? LoadAudioClip(drivingAudio)
                    ?? throw new FileNotFoundException($"Driving audio not found: {drivingAudio}")
                : null,
            Guides = VideoInputManifest.Guides(parameters),
            VideoDenoiseMask = VideoInputManifest.VideoMask(parameters),
            AudioDenoiseMask = VideoInputManifest.AudioMask(parameters),
            Controls = VideoInputManifest.Controls(parameters),
            SparseAttentionPolicy = ParseSparseAttentionPolicy(parameters.GetStringOrNull("sparse-attention")),
            Components = BuildVideoComponents(parameters),
        };

        ConsoleStepProgress? progress = quiet ? null
            : new ConsoleStepProgress("denoise", parameters.Get("preview-output"));
        VideoGenerationResult generated;
        try
        {
            generated = await engine.Video.GenerateAsync(spec, request, progress, cancel).ConfigureAwait(false);
        }
        finally
        {
            progress?.Finish();
        }
        List<VideoFrame> frames = [.. generated.Frames];
        AudioBuffer? soundtrack = generated.Audio;

        // Optional --restore chain: feed the generated frames straight into SeedVR2 (no encode round-trip),
        // reusing the loaded engine. Generate small, restore up.
        if (parameters.GetStringOrNull("restore") is { } restoreModel && frames.Count > 0)
        {
            ModelSpec restoreSpec = ModelResolver.Resolve(restoreModel, null, Modality.Restore);
            RestoreRequest restoreRequest = new RestoreRequest
            {
                Vram = ParseVramOverrides(parameters),
                Frames = frames.Select(f => new ImageData { Rgb = f.Rgb, Width = f.Width, Height = f.Height }).ToList(),
                TargetWidth = parameters.GetIntOrNull("restore-width"),
                TargetHeight = parameters.GetIntOrNull("restore-height"),
            };
            ConsoleStepProgress? restoreProgress = quiet ? null : new ConsoleStepProgress("restore");
            List<VideoFrame> restored = new List<VideoFrame>();
            try
            {
                await foreach (VideoFrame frame in engine.Restore.RestoreAsync(restoreSpec, restoreRequest, restoreProgress, cancel).ConfigureAwait(false))
                {
                    restored.Add(frame);
                }
            }
            finally
            {
                restoreProgress?.Finish();
            }
            frames = restored;
        }
        GeneratedArtifact artifact = FrameArtifact(frames, outputDir, prompt, "video", soundtrack, generated.Fps ?? 24);
        if (generated.Execution is { } execution)
        {
            artifact.Meta["model_profile"] = execution.ProfileId;
            artifact.Meta["task"] = execution.Task.ToString();
            artifact.Meta["seed"] = execution.Seed.ToString(CultureInfo.InvariantCulture);
            artifact.Meta["steps"] = execution.Steps.ToString(CultureInfo.InvariantCulture);
            if (execution.FlowShift is float flowShift)
            {
                artifact.Meta["flow_shift"] = flowShift.ToString(CultureInfo.InvariantCulture);
            }
            if (execution.AudioFlowShift is float audioFlowShift)
            {
                artifact.Meta["audio_flow_shift"] = audioFlowShift.ToString(CultureInfo.InvariantCulture);
            }
            artifact.Meta["acceleration"] = execution.Acceleration.ToString();
            artifact.Meta["attention"] = execution.Attention.ToString();
            artifact.Meta["execution_path"] = execution.ExecutionPath;
        }
        return artifact;
    }

    /// <summary>The image component overrides from <c>--vae</c>, <c>--text-encoder</c> and <c>--vision-encoder</c>, or null.</summary>
    private static ComponentOverrides? BuildImageComponents(ParamState parameters)
    {
        string? vae = parameters.GetStringOrNull("vae");
        string? textEncoder = parameters.GetStringOrNull("text-encoder");
        string? vision = parameters.GetStringOrNull("vision-encoder");
        return vae is null && textEncoder is null && vision is null
            ? null
            : new ComponentOverrides { Vae = vae, Qwen = textEncoder, TextEncoderVision = vision };
    }

    private static ComponentOverrides? BuildVideoComponents(ParamState parameters)
    {
        string? videoVae = parameters.GetStringOrNull("video-vae");
        string? audioVae = parameters.GetStringOrNull("audio-vae");
        return videoVae is null && audioVae is null
            ? null
            : new ComponentOverrides { VideoVae = videoVae, AudioVae = audioVae };
    }

    private static SparseAttentionPolicy ParseSparseAttentionPolicy(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return SparseAttentionPolicy.Auto;
        }
        if (Enum.TryParse(value, true, out SparseAttentionPolicy policy))
        {
            return policy;
        }
        throw new ArgumentException($"Unknown sparse-attention policy '{value}'. Expected auto, require, or disable.");
    }

    /// <summary>3D generation through <see cref="IMeshService"/>; the CLI's "prompt" is the input path: an image for image models, a scene mesh for cola3d, a mesh to rig for unirig.</summary>
    private static async Task<GeneratedArtifact> MeshAsync(
        IInferenceEngine engine, ModelSpec spec, string prompt, ParamState parameters, bool quiet, CancellationToken cancel)
    {
        string meshId = (spec.Catalog?.Id ?? spec.Requested ?? string.Empty).ToLowerInvariant();
        bool paint = MeshModelIds.PaintMesh.Contains(meshId);
        bool scene = MeshModelIds.SceneMesh.Contains(meshId) || MeshModelIds.RigMesh.Contains(meshId) || MeshModelIds.PartMesh.Contains(meshId) || paint;
        string? imagePath = parameters.GetStringOrNull("image");
        if (paint && imagePath is null)
            throw new ArgumentException($"3D model '{meshId}' textures a mesh from a reference image; pass the image with --image.");
        MeshRequest request = new MeshRequest
        {
            Vram = ParseVramOverrides(parameters),
            Image = paint ? LoadMeshImage(imagePath!) : scene ? null : LoadMeshImage(prompt),
            SceneMesh = scene ? LoadSceneMesh(prompt) : null,
            ScenePrompt = LoadScenePrompt(parameters),
            Steps = parameters.GetInt("steps", 0),
            GridResolution = parameters.GetInt("grid", 0),
            Seed = parameters.GetInt("seed", -1),
            FieldOfViewRadians = parameters.GetDoubleOrNull("fov"),
            Samples = parameters.GetInt("samples", 1),
            Rigged = parameters.GetBool("rigged", false),
            Colors = parameters.GetBool("no-colors", false) ? false : null,
            Chains = parameters.GetBool("no-chains", false) ? false : null,
            Texture = parameters.GetBool("texture", false),
            TextureSize = parameters.GetInt("texture-size", 0),
        };

        ConsoleStepProgress? progress = quiet ? null : new ConsoleStepProgress("mesh");
        MeshResult result;
        try
        {
            result = await engine.Mesh.GenerateAsync(spec, request, progress, cancel).ConfigureAwait(false);
        }
        finally
        {
            progress?.Finish();
        }

        GeneratedArtifact artifact = new GeneratedArtifact
        {
            Kind = ArtifactKind.Mesh,
            FileBytes = result.Data,
            Extension = result.Format,
            Text = $"{result.Format.ToUpperInvariant()} mesh ({result.Data.Length / 1024} KB)" + (result.Parts is { } found ? $", {found.Count} parts" : string.Empty),
            SlugHint = spec.Catalog?.Id ?? "mesh",
        };
        artifact.Meta["format"] = result.Format;
        if (result.Parts is { } parts)
        {
            artifact.Meta["parts"] = string.Join("; ", parts.Select(p =>
                $"part {p.Id}: {p.FaceCount} faces, box [{string.Join(", ", p.Box.Select(v => v.ToString("0.####", System.Globalization.CultureInfo.InvariantCulture)))}]"));
            if (!quiet)
            {
                foreach (MeshPart part in parts)
                    Console.WriteLine($"part {part.Id}: {part.FaceCount} faces, box [{string.Join(", ", part.Box.Select(v => v.ToString("0.####", System.Globalization.CultureInfo.InvariantCulture)))}]");
            }
        }
        return artifact;
    }

    /// <summary>World rollout through <see cref="IWorldService"/>: queues one action per frame, then drains the session's frame stream.</summary>
    /// <remarks>Actions come from <c>--actions</c> (cycled to fill the count) or a model-specific default demo plan.
    /// Oasis batches the queued plan into a single one-shot rollout; DIAMOND runs one real denoise per queued action
    /// (see <see cref="DiamondWorldSession"/>). A model that takes an action-script camera (SANA-WM) reads <c>--actions</c> as
    /// that script instead, and is sent no per-frame actions.</remarks>
    private static async Task<GeneratedArtifact> WorldAsync(
        IInferenceEngine engine, ModelSpec spec, string prompt, ParamState parameters, string? outputDir, bool quiet, CancellationToken cancel)
    {
        bool scriptCamera = spec.Catalog?.World?.CameraModes.Contains(WorldCameraMode.ActionScript) == true;
        string? actions = parameters.GetStringOrNull("actions");
        bool view = parameters.GetBool("view", false);
        bool interactive = parameters.GetBool("interactive", false);
        WorldStageTimings.Reset();
        if (view && interactive)
        {
            throw new ArgumentException("--view and --interactive both steer the run; use the window or the terminal, not both.");
        }
        WorldRequestOptions options = new WorldRequestOptions
        {
            Prompt = parameters.GetStringOrNull("world-prompt"),
            Frames = parameters.GetIntOrNull("frames"),
            Fps = parameters.GetIntOrNull("fps"),
            Steps = parameters.GetIntOrNull("steps"),
            Guidance = parameters.GetDoubleOrNull("cfg-scale"),
            FlowShift = parameters.GetDoubleOrNull("flow-shift"),
            Seed = parameters.GetInt("seed", -1),
            CameraPath = parameters.GetStringOrNull("camera"),
            CameraScript = scriptCamera ? actions : null,
            Intrinsics = parameters.GetStringOrNull("intrinsics"),
            SequentialCfg = parameters.GetBool("sequential-cfg", false),
            Refiner = parameters.GetBool("refiner", false),
            // An unset --streaming leaves the mode to the model: a streaming-only model streams, a batch model does not.
            Streaming = parameters.GetBool("streaming", false) || spec.Catalog?.World?.Streaming is { Only: true } ? true : null,
            Unbounded = parameters.GetBool("unbounded", false),
        };
        WorldRequest request = options.ToRequest(LoadImage(prompt), ParseVramOverrides(parameters));
        WorldPlan plan = WorldPlanResolver.Resolve(spec, request, engine.BackendSelector);

        // A live window steers from the keyboard, so it queues no demo plan: a per-step model is driven one action per frame.
        IWorldControlScheme? scheme = spec.Catalog?.World?.LiveControls;
        bool liveSteering = view && scheme is not null;
        if (liveSteering && scheme is { Cadence: WorldControlCadence.PerStep } && actions is not null)
        {
            throw new ArgumentException($"--actions does not apply to '{spec.Catalog?.Id}' with --view; steer it with the keys.");
        }

        ConsoleStepProgress? progress = quiet ? null : new ConsoleStepProgress("rollout");
        string slug = Path.GetFileNameWithoutExtension(prompt.Trim().Trim('"'));
        // A streaming session writes each frame to disk as it arrives, so a long or unbounded run holds none of its pixels and keeps
        // what it generated if it stops early. A batch rollout hands over all its frames at once, and they are collected.
        // Disposed on every exit: a frame the writer has accepted is written before the process can end, even on a failure.
        using StreamingFrameWriter? spool = plan.Streaming ? new StreamingFrameWriter(outputDir ?? RepoPaths.OutputRoot(), slug) : null;
        List<VideoFrame> frames = new List<VideoFrame>();
        int received = 0;
        using IWorldSession session = engine.World.Open(spec, plan.Request);
        if (!scriptCamera && !(liveSteering && actions is null))
        {
            string[] actionPlan = ResolveActionPlan(actions, spec.Catalog?.Id);
            for (int i = 0; i < plan.ActionCount; i++)
            {
                session.SendAction(actionPlan[i % actionPlan.Length]);
            }
        }
        // A streaming session can be ended from the keyboard; what it generated up to then is still saved.
        using CancellationTokenSource userStop = CancellationTokenSource.CreateLinkedTokenSource(cancel);
        Task? keyboard = plan.Streaming && interactive && !Console.IsInputRedirected
            ? Task.Run(() => DriveWorldFromKeyboard(session, userStop), CancellationToken.None)
            : null;
        WorldSteering steering = new WorldSteering(session, scheme);
        WorldPlayback? playback = null;
        WorldViewerHost? viewer = null;
        WorldRunPump? pump = null;
        GeneratedArtifact artifact;
        try
        {
            IAsyncEnumerable<VideoFrame> stream;
            if (view)
            {
                // The window plays frames at the plan's rate from a buffer: a full buffer holds generation back. Its size is the control
                // lag, so a per-step stream gets a short one.
                playback = new WorldPlayback(plan.Fps, WorldPlayback.DefaultBufferFrames(plan.Fps, scheme?.Cadence));
                viewer = WorldViewerHost.Start(steering, playback, userStop);
                stream = new WorldFrameStream(session, steering, plan.Unbounded ? 0 : plan.Frames).StreamAsync(userStop.Token);
            }
            else
            {
                stream = session.StreamAsync(userStop.Token);
            }
            pump = new WorldRunPump(steering, playback, viewer);
            try
            {
                await pump.RunAsync(stream, frame =>
                {
                    if (spool is not null)
                    {
                        long spoolStart = WorldStageTimings.Start();
                        spool.Add(frame.Rgb, frame.Width, frame.Height);
                        WorldStageTimings.Record(WorldStage.FrameSpoolHandoff, spoolStart);
                    }
                    else
                    {
                        frames.Add(frame);
                    }
                    received++;
                    progress?.Report(new StepPreview { Step = received, TotalSteps = plan.Unbounded ? received : plan.Frames });
                }, userStop.Token).ConfigureAwait(false);
            }
            catch (OperationCanceledException) when (received > 0 && userStop.IsCancellationRequested
                && (!cancel.IsCancellationRequested || plan.Unbounded))
            {
                // The stream was ended on purpose, so what it generated up to then is kept: from the keyboard (q), or with Ctrl+C when the
                // run has no frame count to reach (--unbounded, which is stopped that way), or by closing the window.
            }
            catch when (spool is { Count: > 0 })
            {
                // A bounded run that is interrupted or fails part-way still has every frame it wrote on disk; say where. The writer is
                // drained first, so the frames it accepted are on disk before the count is reported.
                spool.Dispose();
                Logs.Warning($"[World] The stream ended after {spool.Count} frames; they are saved in {spool.OutputDirectory}.");
                throw;
            }
            finally
            {
                // Keys pressed after this change nothing, so the session is never steered once it is being torn down.
                userStop.Cancel();
                if (keyboard is not null)
                {
                    await keyboard.ConfigureAwait(false);
                }
                progress?.Finish();
            }
            if (spool is not null)
            {
                if (spool.Count == 0)
                {
                    throw new InvalidOperationException("The world pipeline produced no frames.");
                }
                long muxStart = WorldStageTimings.Start();
                VideoOutputWriter.Written written = spool.Complete(null, plan.Fps);
                WorldStageTimings.Record(WorldStage.Mux, muxStart);
                artifact = DescribeFrames(spool.Count, spool.Width, spool.Height, spool.FirstFrame!, written, null);
            }
            else
            {
                artifact = FrameArtifact(frames, outputDir, slug, "world", null, plan.Fps);
            }
            // The files are written; a window still playing its buffer stays open until it is closed. A Ctrl+C here keeps the files
            // and closes the window rather than reporting the run as cancelled.
            await pump.WaitForCloseAsync(cancel).ConfigureAwait(false);
        }
        finally
        {
            // The window closes on every exit, and before userStop is disposed, since a window left open would cancel a disposed source.
            pump?.Dispose();
        }
        if (view || EngineKnobs.StageProfile.Value)
        {
            Logs.Info("[World] Where the time went:\n" + WorldStageTimings.Report());
        }
        artifact.Meta["fps"] = plan.Fps.ToString(CultureInfo.InvariantCulture);
        artifact.Meta["seed"] = plan.Seed.ToString(CultureInfo.InvariantCulture);
        // Planning adjustments (a truncated camera path, an expanded action script) are reported with the artifact.
        for (int i = 0; i < plan.Notes.Count; i++)
        {
            artifact.Meta[$"note{i + 1}"] = plan.Notes[i];
        }
        return artifact;
    }

    /// <summary>Steers a streaming world session from the keyboard until <paramref name="stop"/> is cancelled. A terminal reports
    /// key presses but not releases, so each of <c>wasdijkl</c> toggles its control (pressing one direction of an axis releases
    /// the opposite one), space or <c>x</c> releases everything, and <c>q</c> or Esc cancels <paramref name="stop"/>. Every change
    /// is sent as the key set now held, which the stream applies from its next chunk.</summary>
    private static void DriveWorldFromKeyboard(IWorldSession session, CancellationTokenSource stop)
    {
        const string Controls = "wasdijkl";
        const string Opposites = "sdwaklij";
        List<char> held = [];
        while (!stop.IsCancellationRequested)
        {
            if (!Console.KeyAvailable)
            {
                stop.Token.WaitHandle.WaitOne(20);
                continue;
            }
            ConsoleKeyInfo key = Console.ReadKey(intercept: true);
            char pressed = char.ToLowerInvariant(key.KeyChar);
            if (key.Key == ConsoleKey.Escape || pressed == 'q')
            {
                stop.Cancel();
                return;
            }
            int control = Controls.IndexOf(pressed);
            if (pressed == ' ' || pressed == 'x')
            {
                held.Clear();
            }
            else if (control >= 0)
            {
                if (!held.Remove(pressed))
                {
                    held.Remove(Opposites[control]);
                    held.Add(pressed);
                }
            }
            else
            {
                continue;
            }
            try
            {
                session.SendAction(held.Count == 0 ? "none" : new string(held.ToArray()));
            }
            catch (ObjectDisposedException)
            {
                return;
            }
        }
    }

    /// <summary>Resolves the action plan: an explicit comma-separated <paramref name="actionsOption"/>, else a per-model demo plan.</summary>
    /// <remarks>DIAMOND/Breakout fires then alternates left/right to show paddle response; everything else uses Oasis's canned
    /// forward walk.</remarks>
    private static string[] ResolveActionPlan(string? actionsOption, string? catalogId)
    {
        if (!string.IsNullOrWhiteSpace(actionsOption))
        {
            string[] actions = actionsOption.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
            // A value like "," names nothing, and a rollout indexes the plan modulo its length.
            if (actions.Length == 0)
            {
                throw new ArgumentException($"--actions \"{actionsOption}\" names no action; give a comma-separated list such as \"w-24,d-24\".");
            }
            return actions;
        }
        return string.Equals(catalogId, "diamond", StringComparison.OrdinalIgnoreCase)
            ? ["fire", "right", "right", "right", "right", "left", "left", "left", "left"] : ["forward"];
    }

    /// <summary>Video/image restoration through <see cref="IRestoreService"/>; video inputs are decoded ONCE here (ffmpeg subprocess) so the source fps is known for the MP4 remux.</summary>
    private static async Task<GeneratedArtifact> RestoreAsync(
        IInferenceEngine engine, ModelSpec spec, string prompt, ParamState parameters, string? outputDir,
        bool quiet, CancellationToken cancel)
    {
        string inputPath = prompt.Trim().Trim('"');
        if (!File.Exists(inputPath))
        {
            throw new FileNotFoundException($"Restore input not found: {inputPath}");
        }
        bool isImage = Path.GetExtension(inputPath).ToLowerInvariant() is ".png" or ".bmp" or ".jpg" or ".jpeg";

        double sourceFps = 25.0;
        RestoreRequest request;
        RestoreRequest baseRequest = new RestoreRequest
        {
            Vram = ParseVramOverrides(parameters),
            Device = parameters.GetStringOrNull("restore-device"),
            TargetWidth = parameters.GetIntOrNull("width"),
            TargetHeight = parameters.GetIntOrNull("height"),
            ClipFrames = parameters.GetIntOrNull("clip-frames"),
            Overlap = parameters.GetIntOrNull("overlap"),
            Strength = parameters.GetFloatOrNull("strength"),
            Fps = parameters.GetDoubleOrNull("fps"),
            Seed = parameters.GetIntOrNull("seed") is int s and >= 0 ? s : null,
        };
        if (isImage)
        {
            request = baseRequest with { Image = LoadImage(inputPath) };
        }
        else
        {
            HartsyInference.Video.Encoding.FfmpegProcessDecoder decoder = new();
            HartsyInference.Video.Encoding.FfmpegProcessDecoder.Result decoded =
                await decoder.DecodeFileAsync(inputPath, cancel).ConfigureAwait(false);
            sourceFps = decoded.Fps;
            request = baseRequest with
            {
                Frames = decoded.Frames
                    .Select(f => new ImageData { Rgb = f, Width = decoded.Width, Height = decoded.Height })
                    .ToList(),
            };
        }

        ConsoleStepProgress? progress = quiet ? null : new ConsoleStepProgress("restore");
        List<VideoFrame> frames = new List<VideoFrame>();
        try
        {
            await foreach (VideoFrame frame in engine.Restore.RestoreAsync(spec, request, progress, cancel).ConfigureAwait(false))
            {
                frames.Add(frame);
            }
        }
        finally
        {
            progress?.Finish();
        }

        string slug = Path.GetFileNameWithoutExtension(inputPath);
        GeneratedArtifact artifact = FrameArtifact(frames, outputDir, slug, "restore", null,
            (int)Math.Round(request.Fps ?? sourceFps));
        return artifact;
    }

    /// <summary>Writes a frame sequence to disk and describes it, previewing the first frame inline.</summary>
    private static GeneratedArtifact FrameArtifact(IReadOnlyList<VideoFrame> frames, string? outputDir, string slug, string label,
        AudioBuffer? audio = null, int fps = 24)
    {
        if (frames.Count == 0)
        {
            throw new InvalidOperationException($"The {label} pipeline produced no frames.");
        }
        int width = frames[0].Width;
        int height = frames[0].Height;
        byte[][] pixels = new byte[frames.Count][];
        for (int i = 0; i < frames.Count; i++)
        {
            pixels[i] = frames[i].Rgb;
        }
        VideoOutputWriter.Written written = VideoOutputWriter.Write(
            pixels, width, height, outputDir ?? RepoPaths.OutputRoot(), slug, audio, fps);
        return DescribeFrames(frames.Count, width, height, frames[0].Rgb, written, audio);
    }

    /// <summary>Describes a frame sequence already on disk, previewing its first frame inline.</summary>
    private static GeneratedArtifact DescribeFrames(int count, int width, int height, byte[] firstFrame,
        VideoOutputWriter.Written written, AudioBuffer? audio)
    {
        GeneratedArtifact artifact = new GeneratedArtifact
        {
            Kind = ArtifactKind.Video,
            Extension = "png",
            Text = $"{count} frames ({width}x{height}) → {written.Directory}"
                + (written.AudioPath is null ? "" : $" + {audio!.Seconds:0.0}s audio.wav")
                + (written.Mp4Path is null ? "" : " + video.mp4"),
            PreviewRgb = firstFrame,
            PreviewWidth = width,
            PreviewHeight = height,
            SelfWritten = true,
        };
        artifact.Meta["frames"] = count.ToString(CultureInfo.InvariantCulture);
        artifact.Meta["size"] = $"{width}x{height}";
        if (written.Mp4Path is not null)
        {
            artifact.Meta["mp4"] = written.Mp4Path;
        }
        return artifact;
    }

    /// <summary>Reads an audio file into an <see cref="AudioClip"/>, or null when <paramref name="path"/> is null/empty.</summary>
    /// <remarks>Throws if a path was given but does not exist.</remarks>
    internal static AudioClip? LoadAudioClip(string? path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return null;
        }
        string trimmed = path.Trim().Trim('"');
        if (!File.Exists(trimmed))
        {
            throw new FileNotFoundException($"Audio file not found: {trimmed}");
        }
        return new AudioClip { Data = File.ReadAllBytes(trimmed), Format = Path.GetExtension(trimmed).TrimStart('.').ToLowerInvariant() };
    }

    /// <summary>Writes each named stem as its own WAV into a fresh subdirectory since a stem set has no single-file artifact.</summary>
    /// <remarks>Mirrors <see cref="FrameWriter"/>'s "one call, N files" shape.</remarks>
    private static GeneratedArtifact StemsArtifact(StemsResult stems, string? outputDir, string slug)
    {
        if (stems.Stems.Count == 0)
        {
            throw new InvalidOperationException("The separation model produced no stems.");
        }
        string baseDir = outputDir ?? RepoPaths.OutputRoot();
        Directory.CreateDirectory(baseDir);
        string stemSlug = Slug.Make(slug);
        string dir = ArtifactWriter.NextAvailablePath(baseDir, i => $"{stemSlug}-stems-{i:D3}", Directory.Exists, () => $"{stemSlug}-stems-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        foreach (KeyValuePair<string, byte[]> stem in stems.Stems)
        {
            File.WriteAllBytes(Path.Combine(dir, $"{stem.Key}.{stems.Format}"), stem.Value);
        }
        return new GeneratedArtifact
        {
            Kind = ArtifactKind.Audio,
            Extension = stems.Format,
            Text = $"{stems.Stems.Count} stem(s) ({string.Join(", ", stems.Stems.Keys)}) @ {stems.SampleRate} Hz → {dir}",
            SelfWritten = true,
        };
    }

    /// <summary>Wraps an <see cref="AudioResult"/> as a saveable artifact with its duration/rate footer.</summary>
    private static GeneratedArtifact AudioArtifact(AudioResult result, string label)
    {
        GeneratedArtifact artifact = new GeneratedArtifact
        {
            Kind = ArtifactKind.Audio,
            FileBytes = result.Data,
            Extension = result.Format,
            Text = $"{result.DurationSeconds:F1}s of {label} ({result.SampleRate} Hz)",
        };
        artifact.Meta["seconds"] = result.DurationSeconds.ToString("F1", CultureInfo.InvariantCulture);
        artifact.Meta["sample_rate"] = result.SampleRate.ToString(CultureInfo.InvariantCulture);
        CopyMeta(result.Meta, artifact);
        return artifact;
    }

    /// <summary>Human-readable one-liner for a vision result, matching the mode that produced it.</summary>
    private static string Describe(VisionResult result)
    {
        if (result.Embedding is { Length: > 0 } embedding)
        {
            StringBuilder preview = new StringBuilder();
            int shown = Math.Min(8, embedding.Length);
            for (int i = 0; i < shown; i++)
            {
                preview.Append(i == 0 ? "" : ", ").Append(embedding[i].ToString("F4", CultureInfo.InvariantCulture));
            }
            return $"{embedding.Length}-dim embedding: [{preview}, …]";
        }
        if (result.Detections is { Count: > 0 } detections)
        {
            StringBuilder lines = new StringBuilder();
            foreach (Detection d in detections)
            {
                lines.Append(d.Label)
                    .Append(' ').Append(d.Score.ToString("F2", CultureInfo.InvariantCulture))
                    .Append("  (").Append(d.X).Append(',').Append(d.Y)
                    .Append(")-(").Append(d.X + d.Width).Append(',').Append(d.Y + d.Height).Append(")\n");
            }
            return lines.ToString().TrimEnd('\n');
        }
        if (result.Masks is { Count: > 0 } masks)
        {
            return $"{masks.Count} mask(s) at {masks[0].Width}x{masks[0].Height}";
        }
        return "(no results)";
    }

    /// <summary>Merges a service's free-form metadata into the artifact footer without clobbering CLI-set keys.</summary>
    private static void CopyMeta(IReadOnlyDictionary<string, string> meta, GeneratedArtifact artifact)
    {
        foreach (KeyValuePair<string, string> entry in meta)
        {
            artifact.Meta.TryAdd(entry.Key, entry.Value);
        }
    }

    /// <summary>The img2img init from <c>--init-image</c> / <c>--creativity</c> / <c>--init-image-mode</c>, or null for text-to-image.</summary>
    private static Img2Img? BuildImg2Img(ParamState parameters, ModelSpec spec)
    {
        string? initPath = parameters.GetStringOrNull("init-image");
        if (initPath is null)
        {
            return null;
        }
        Img2Img img2img = new Img2Img { InitImage = LoadImage(initPath) };
        double? creativity = parameters.GetDoubleOrNull("creativity");
        if (creativity is not null)
        {
            img2img = img2img with { Creativity = creativity.Value };
        }
        else if (parameters.GetStringOrNull("mask") is not null && Flux1Recipe.IsFillCatalogId(spec.Catalog?.Id))
        {
            // A Fill model conditions on the masked image and regenerates the masked area, so a mask starts it at full denoise.
            // Other models keep their own default: 0.6 keeps part of the masked content.
            img2img = img2img with { Creativity = 1.0 };
        }
        return parameters.GetStringOrNull("init-image-mode") switch
        {
            "denoise" => img2img with { Mode = Img2ImgMode.Denoise },
            "reference" => img2img with { Mode = Img2ImgMode.Reference },
            null or "auto" => img2img,
            string mode => throw new ArgumentException($"Unknown --init-image-mode '{mode}'; expected denoise, reference or auto."),
        };
    }

    /// <summary>The Redux Extra keys from <c>--style-model</c> and its strength knobs; empty when Redux is off.</summary>
    internal static IReadOnlyDictionary<string, object> BuildImageExtra(ParamState parameters)
    {
        Dictionary<string, object> extra = new Dictionary<string, object>(StringComparer.Ordinal);
        if (parameters.GetStringOrNull("style-model") is { Length: > 0 } styleModel)
        {
            extra[HartsyInference.Engine.Features.RequestExtras.ReduxStyleModel] = styleModel;
        }
        if (parameters.GetDoubleOrNull("redux-multiply") is { } reduxMultiply)
        {
            extra[HartsyInference.Engine.Features.RequestExtras.ReduxMultiply] = reduxMultiply;
        }
        if (parameters.GetDoubleOrNull("redux-merge") is { } reduxMerge)
        {
            extra[HartsyInference.Engine.Features.RequestExtras.ReduxMerge] = reduxMerge;
        }
        if (parameters.GetDoubleOrNull("redux-apply-start") is { } reduxApplyStart)
        {
            extra[HartsyInference.Engine.Features.RequestExtras.ReduxApplyStart] = reduxApplyStart;
        }
        (string Flag, string Key)[] slgLists =
        [
            ("slg-uncond-layers", HartsyInference.Engine.Features.RequestExtras.SlgUncondSkipLayers),
            ("slg-layers", HartsyInference.Engine.Features.RequestExtras.SlgLayers),
        ];
        foreach ((string flag, string key) in slgLists)
        {
            if (parameters.GetStringOrNull(flag) is { Length: > 0 } layers)
            {
                extra[key] = layers;
            }
        }
        (string Flag, string Key)[] slgNumbers =
        [
            ("slg-scale", HartsyInference.Engine.Features.RequestExtras.SlgScale),
            ("slg-start", HartsyInference.Engine.Features.RequestExtras.SlgStart),
            ("slg-stop", HartsyInference.Engine.Features.RequestExtras.SlgStop),
        ];
        foreach ((string flag, string key) in slgNumbers)
        {
            if (parameters.GetDoubleOrNull(flag) is { } value)
            {
                extra[key] = value;
            }
        }
        if (parameters.GetStringOrNull("reference-kv-cache") is { Length: > 0 } kvCache)
        {
            extra[HartsyInference.Engine.Features.RequestExtras.ReferenceKvCache] = kvCache;
        }
        if (parameters.GetIntOrNull("reference-max-pixels") is { } maxPixels)
        {
            extra[HartsyInference.Engine.Features.RequestExtras.ReferenceMaxPixels] = maxPixels;
        }
        if (parameters.GetStringOrNull("reference-layout") is { Length: > 0 } layout)
        {
            extra[HartsyInference.Engine.Features.RequestExtras.ReferenceLayout] = layout;
        }
        if (parameters.GetIntOrNull("reference-scale") is { } refScale)
        {
            extra[HartsyInference.Engine.Features.RequestExtras.ReferenceScale] = refScale;
        }
        if (parameters.GetIntOrNull("reference-vision-pixels") is { } visionPixels)
        {
            extra[HartsyInference.Engine.Features.RequestExtras.ReferenceVisionPixels] = visionPixels;
        }
        if (parameters.GetStringOrNull("reference-adapter") is { } adapter)
        {
            extra[HartsyInference.Engine.Features.RequestExtras.ReferenceAdapter] = adapter;
        }
        if (parameters.GetFloatOrNull("reference-strength") is { } adapterStrength)
        {
            extra[HartsyInference.Engine.Features.RequestExtras.ReferenceStrength] = adapterStrength;
        }
        if (parameters.GetFloatOrNull("reference-guidance") is { } referenceGuidance)
        {
            extra[HartsyInference.Engine.Features.RequestExtras.ReferenceGuidance] = referenceGuidance;
        }
        if (parameters.GetStringOrNull("reference-correction") is { } correction)
        {
            extra[HartsyInference.Engine.Features.RequestExtras.ReferenceCorrection] = correction;
        }
        if (parameters.GetFloatOrNull("reference-rescale-phi") is { } phi)
        {
            extra[HartsyInference.Engine.Features.RequestExtras.ReferenceRescalePhi] = phi;
        }
        if (parameters.GetFloatOrNull("reference-start") is { } start)
        {
            extra[HartsyInference.Engine.Features.RequestExtras.ReferenceStart] = start;
        }
        if (parameters.GetFloatOrNull("reference-end") is { } end)
        {
            extra[HartsyInference.Engine.Features.RequestExtras.ReferenceEnd] = end;
        }
        if (parameters.GetStringOrNull("reference-outside") is { } outside)
        {
            extra[HartsyInference.Engine.Features.RequestExtras.ReferenceOutside] = outside;
        }
        return extra;
    }

    /// <summary>The inpaint mask from <c>--mask</c> and its grow/blur knobs, or null when no mask was given.</summary>
    private static Inpaint? BuildInpaint(ParamState parameters)
    {
        string? maskPath = parameters.GetStringOrNull("mask");
        if (maskPath is null)
        {
            return null;
        }
        Inpaint inpaint = new Inpaint { Mask = LoadImage(maskPath) };
        int? grow = parameters.GetIntOrNull("mask-grow");
        int? blur = parameters.GetIntOrNull("mask-blur");
        int? shrinkGrow = parameters.GetIntOrNull("mask-shrink-grow");
        if (grow is not null)
        {
            inpaint = inpaint with { Grow = grow.Value };
        }
        if (blur is not null)
        {
            inpaint = inpaint with { Blur = blur.Value };
        }
        return shrinkGrow is null ? inpaint : inpaint with { ShrinkGrow = shrinkGrow.Value };
    }

    /// <summary>The LoRA stack from the newline-joined --lora/--lora-weight pair; null when none were given.</summary>
    private static LoraStack? BuildLoraStack(ParamState parameters)
    {
        string[]? names = SplitPaths(parameters.GetStringOrNull("loras"));
        if (names is null || names.Length == 0)
        {
            return null;
        }
        string[]? weights = SplitPaths(parameters.GetStringOrNull("lora-weights"));
        List<LoraEntry> entries = [];
        for (int i = 0; i < names.Length; i++)
        {
            double weight = weights is not null && i < weights.Length
                && double.TryParse(weights[i], NumberStyles.Float, CultureInfo.InvariantCulture, out double w) ? w : 1.0;
            entries.Add(new LoraEntry { Model = names[i], Weight = weight });
        }
        return new LoraStack { Entries = entries };
    }

    /// <summary>Builds the reference clips, pairing each soundtrack to the same-position clip; a missing entry leaves that clip silent rather than shifting the pairing.</summary>
    /// <summary>Total chain frames from either the frame or the seconds flag; seconds convert at H3's native rate.</summary>
    private static int? ResolveChainTotalFrames(ParamState parameters)
    {
        if (parameters.GetIntOrNull("chain-frames") is int frames)
        {
            return frames;
        }
        return parameters.GetDoubleOrNull("chain-seconds") is double seconds
            ? (int)Math.Round(seconds * MiniMaxH3Geometry.Fps)
            : null;
    }

    private static List<ReferenceVideo>? BuildReferenceVideos(ParamState parameters)
    {
        string[]? paths = SplitPaths(parameters.GetStringOrNull("ref-videos"));
        if (paths is null)
        {
            return null;
        }
        string[] soundtracks = SplitPaths(parameters.GetStringOrNull("ref-video-audios")) ?? [];
        List<ReferenceVideo> videos = new List<ReferenceVideo>(paths.Length);
        for (int i = 0; i < paths.Length; i++)
        {
            videos.Add(new ReferenceVideo
            {
                Video = LoadVideoClip(paths[i]),
                Audio = i < soundtracks.Length ? LoadAudioClip(soundtracks[i]) : null,
            });
        }
        return videos;
    }

    /// <summary>Reads a video container off disk as the Engine's encoded-video payload.</summary>
    private static VideoClip LoadVideoClip(string clipPath)
    {
        string path = clipPath.Trim().Trim('"');
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"Video file not found: {path}");
        }
        string extension = Path.GetExtension(path).TrimStart('.');
        return new VideoClip
        {
            Data = File.ReadAllBytes(path),
            Format = extension.Length == 0 ? null : extension.ToLowerInvariant(),
        };
    }

    /// <summary>Splits a newline-joined repeatable option back into its paths; null when the option was absent.</summary>
    private static string[]? SplitPaths(string? joined) =>
        string.IsNullOrWhiteSpace(joined) ? null : joined.Split('\n', StringSplitOptions.RemoveEmptyEntries);

    /// <summary>Reads the cola3d scene prompt from <c>--prompt-mask</c> or <c>--prompt-point</c>; null when neither is given.</summary>
    private static SceneMeshPrompt? LoadScenePrompt(ParamState parameters)
    {
        string[]? maskPaths = SplitPaths(parameters.GetStringOrNull("prompt-mask"));
        string[]? pointTexts = SplitPaths(parameters.GetStringOrNull("prompt-point"));
        if (maskPaths is null && pointTexts is null)
        {
            return null;
        }
        if (maskPaths is not null && pointTexts is not null)
        {
            throw new ArgumentException("Give --prompt-mask or --prompt-point, not both.");
        }
        if (maskPaths is not null)
        {
            return new SceneMeshPrompt { Masks = maskPaths.Select(LoadPromptMask).ToList() };
        }
        return new SceneMeshPrompt { Points = pointTexts!.Select(ParseScenePoint).ToList() };
    }

    /// <summary>Reads one mask PNG; the red channel is the coverage.</summary>
    private static ScenePromptMask LoadPromptMask(string path)
    {
        ImageData image = LoadImage(path);
        byte[] coverage = new byte[image.Width * image.Height];
        for (int i = 0; i < coverage.Length; i++)
        {
            coverage[i] = image.Rgb[i * 3];
        }
        return new ScenePromptMask { Coverage = coverage, Width = image.Width, Height = image.Height };
    }

    /// <summary>Parses <c>x,y,z</c> in the scene mesh's units.</summary>
    private static ScenePoint ParseScenePoint(string text)
    {
        string[] parts = text.Split(',', StringSplitOptions.TrimEntries);
        if (parts.Length != 3 || !float.TryParse(parts[0], NumberStyles.Float, CultureInfo.InvariantCulture, out float x)
            || !float.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out float y)
            || !float.TryParse(parts[2], NumberStyles.Float, CultureInfo.InvariantCulture, out float z))
        {
            throw new ArgumentException($"Prompt point '{text}' must be three numbers as x,y,z.");
        }
        return new ScenePoint(x, y, z);
    }

    /// <summary>Reads a scene mesh (obj or glb) for the cola3d route; the format comes from its extension.</summary>
    private static SceneMeshData LoadSceneMesh(string promptPath)
    {
        string path = promptPath.Trim().Trim('"');
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"Input scene mesh not found: {path}");
        }
        return new SceneMeshData { Data = File.ReadAllBytes(path), Format = SceneMeshReader.FormatOf(path) };
    }

    /// <summary>A mesh input image with its alpha plane and EXIF orientation applied; a non-PNG file goes through <see cref="LoadImage"/>.</summary>
    private static ImageData LoadMeshImage(string promptPath)
    {
        string path = promptPath.Trim().Trim('"');
        if (!path.EndsWith(".png", StringComparison.OrdinalIgnoreCase) || !File.Exists(path))
        {
            return LoadImage(promptPath);
        }

        DecodedPng png = PngDecoder.DecodeWithAlpha(File.ReadAllBytes(path));
        return new ImageData { Rgb = png.Rgb, Alpha = png.Alpha, Width = png.Width, Height = png.Height };
    }

    /// <summary>Reads an image file (PNG or BMP) into the engine's RGB24 <see cref="ImageData"/> contract.</summary>
    private static ImageData LoadImage(string promptPath)
    {
        string path = promptPath.Trim().Trim('"');
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"Input image not found: {path}");
        }
        (byte[] rgb, int width, int height) = ImageIo.DecodeFile(path);
        return new ImageData { Rgb = rgb, Width = width, Height = height };
    }

    /// <summary>The per-run VRAM override from <c>--vram-mode</c>, or null when the caller did not ask for one.</summary>
    /// <remarks>Null rather than an explicit Auto: null means "follow the engine", which is a different statement
    /// from "pin Auto" once a host has configured a backend policy.</remarks>
    private static VramOverrides? ParseVramOverrides(ParamState parameters)
    {
        string? mode = parameters.GetStringOrNull("vram-mode");
        if (string.IsNullOrWhiteSpace(mode))
        {
            return null;
        }
        if (!Enum.TryParse(mode.Trim(), ignoreCase: true, out VramTier tier))
        {
            Logs.Warning($"[VRAM] --vram-mode '{mode}' is not recognized; ignoring it. "
                + "Valid: Performance, Auto, Balanced, Aggressive, Maximum.");
            return null;
        }
        return new VramOverrides { Tier = tier };
    }

}
