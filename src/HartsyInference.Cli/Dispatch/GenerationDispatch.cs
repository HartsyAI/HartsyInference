using System.Globalization;
using System.Text;
using HartsyInference.Cli.Infra;
using HartsyInference.Engine.Services;
using HartsyInference.Vision.Codec;

namespace HartsyInference.Cli.Dispatch;

/// <summary>Maps the CLI's flag bag onto the engine's typed per-modality services and adapts each typed result back
/// into a <see cref="GeneratedArtifact"/> for presentation and persistence. This is the CLI's only generation entry
/// point: one <c>switch</c> over <see cref="Modality"/>, one typed request built per branch.</summary>
public static class GenerationDispatch
{
    /// <summary>Runs one generation for <paramref name="spec"/>. <paramref name="prompt"/> is the text prompt for the
    /// generative modalities and an input file path for transcribe / vision / 3d / world. Frame sequences (video,
    /// world) are written under <paramref name="outputDir"/> as they have no single-file artifact.</summary>
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
            Modality.Vision => await VisionAsync(engine, spec, prompt, parameters, cancel).ConfigureAwait(false),
            Modality.Video => await VideoAsync(engine, spec, prompt, parameters, outputDir, quiet, cancel).ConfigureAwait(false),
            Modality.Mesh => await MeshAsync(engine, spec, prompt, parameters, quiet, cancel).ConfigureAwait(false),
            Modality.World => await WorldAsync(engine, spec, prompt, parameters, outputDir, quiet, cancel).ConfigureAwait(false),
            Modality.VoiceConvert => await VoiceConvertAsync(engine, spec, prompt, parameters, cancel).ConfigureAwait(false),
            Modality.Fx => await FxAsync(engine, spec, prompt, parameters, outputDir, cancel).ConfigureAwait(false),
            _ => throw new NotSupportedException($"The '{Modalities.ToCliName(spec.Modality)}' modality has no CLI dispatch."),
        };
    }

    /// <summary>Text-to-image through <see cref="IImagesService"/>; the RGB result is encoded to PNG for saving and
    /// kept raw for the inline terminal preview.</summary>
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
            Sampler = parameters.GetStringOrNull("sampler"),
            Scheduler = parameters.GetStringOrNull("scheduler"),
            SigmaShift = parameters.GetDoubleOrNull("sigma-shift"),
            Seed = parameters.GetInt("seed", -1),
        };

        ConsoleStepProgress? progress = quiet ? null : new ConsoleStepProgress("denoise");
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
            FileBytes = PngEncoder.Encode(result.Rgb, result.Width, result.Height),
            Extension = "png",
            Text = $"{result.Width}x{result.Height} image (seed {result.Seed})",
            PreviewRgb = result.Rgb,
            PreviewWidth = result.Width,
            PreviewHeight = result.Height,
        };
        artifact.Meta["size"] = $"{result.Width}x{result.Height}";
        artifact.Meta["seed"] = result.Seed.ToString(CultureInfo.InvariantCulture);
        // The step count reported is the one the pipeline actually ran (its own meta), since an unset --steps resolves
        // to the family's official default inside the engine.
        CopyMeta(result.Meta, artifact);
        return artifact;
    }

    /// <summary>Chat completion through <see cref="ITextService"/>. Interactive runs consume the token stream so text
    /// still prints live; <c>--quiet</c> takes the one-shot path and lets the presenter print the final text.</summary>
    private static async Task<GeneratedArtifact> TextAsync(
        IInferenceEngine engine, ModelSpec spec, string prompt, ParamState parameters, bool quiet, CancellationToken cancel)
    {
        float temperature = parameters.GetFloat("temperature", 0.7f);
        List<ImageData> images = [];
        if (parameters.GetStringOrNull("image") is { Length: > 0 } imagePaths)
        {
            foreach (string path in imagePaths.Split('\n', StringSplitOptions.RemoveEmptyEntries))
                images.Add(LoadImage(path));
        }
        TextRequest request = new TextRequest
        {
            Messages = [new TextMessage { Role = TextRole.User, Content = prompt, Images = images.Count > 0 ? images : null }],
            SystemPrompt = parameters.GetStringOrNull("system"),
            MaxTokens = parameters.GetInt("max-tokens", 256),
            Temperature = temperature,
            TopP = parameters.GetFloat("top-p", 0.95f),
            TopK = parameters.GetIntOrNull("top-k"),
            MinP = parameters.GetDoubleOrNull("min-p"),
            RepetitionPenalty = parameters.GetDoubleOrNull("repetition-penalty"),
            Seed = parameters.GetInt("seed", -1),
            Greedy = temperature <= 0f,
            GraphDecode = parameters.GetBool("graph-decode", false) ? true : null,
            EnableThinking = parameters.GetStringOrNull("thinking") is { Length: > 0 } thinking ? bool.Parse(thinking) : null,
            // TextService.LoadInto only checks LowVramQuant for non-empty (a bool-shaped toggle, not an actual
            // target quant selector), so any non-empty sentinel works here — see its doc comment.
            LowVramQuant = parameters.GetBool("low-vram-quant", false) ? "on" : null,
            AlwaysFreeMemory = parameters.GetBool("always-free-memory", false) ? true : null,
        };

        StringBuilder text = new StringBuilder();
        StopReason stop = StopReason.Stop;
        if (quiet)
        {
            TextResult result = await engine.Text.GenerateAsync(spec, request, cancel).ConfigureAwait(false);
            text.Append(result.Text);
            stop = result.Stop;
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
        return artifact;
    }

    /// <summary>Text-to-speech through <see cref="ISpeechService"/>. Voice-cloning models read the optional
    /// <c>reference</c> path (and, for F5-style models that align against a transcript, <c>ref-text</c>).</summary>
    private static async Task<GeneratedArtifact> SpeechAsync(
        IInferenceEngine engine, ModelSpec spec, string prompt, ParamState parameters, CancellationToken cancel)
    {
        double speed = parameters.GetFloat("speed", 1f);
        SpeechRequest request = new SpeechRequest
        {
            Text = prompt,
            Voice = parameters.Get("voice"),
            Speed = speed > 0d ? speed : null,
            Reference = LoadAudioClip(parameters.GetStringOrNull("reference")),
            RefText = parameters.Get("ref-text") ?? "",
            Exaggeration = parameters.GetDoubleOrNull("exaggeration"),
            NfeStep = parameters.GetIntOrNull("nfe-step"),
            CfgScale = parameters.GetDoubleOrNull("cfg-scale"),
            Seed = Math.Max(0, parameters.GetInt("seed", 0)),
        };
        AudioResult result = await engine.Speech.SynthesizeAsync(spec, request, cancel).ConfigureAwait(false);
        return AudioArtifact(result, "speech");
    }

    /// <summary>Voice conversion through <see cref="IVoiceConversionService"/>; the CLI's "prompt" is the source audio
    /// path. An optional <c>target-path</c> tunable supplies the tone-color reference (OpenVoice); RVC ignores it.</summary>
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

    /// <summary>Audio effects (Demucs separation / Resemble-Enhance) through <see cref="IFxService"/>; the CLI's
    /// "prompt" is the input audio path and the <c>mode</c> tunable picks which of the two operations to run.
    /// Separation yields multiple stems, written directly to a fresh subfolder since there is no single-file result.</summary>
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
        MusicRequest request = new MusicRequest
        {
            Prompt = prompt,
            Duration = parameters.GetInt("duration", 10),
            Seed = Math.Max(0, parameters.GetInt("seed", 0)),
        };
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

    /// <summary>Speech-to-text through <see cref="ITranscribeService"/>; the CLI's "prompt" is the audio file path,
    /// read here into an <see cref="AudioClip"/> because the typed request takes bytes, not a path.</summary>
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
            Text = result.Text.Trim(),
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

    /// <summary>Embed / detect / segment through <see cref="IVisionService"/>; the CLI's "prompt" is the image path.</summary>
    private static async Task<GeneratedArtifact> VisionAsync(
        IInferenceEngine engine, ModelSpec spec, string prompt, ParamState parameters, CancellationToken cancel)
    {
        ImageData image = LoadImage(prompt);
        string mode = parameters.Get("mode") ?? "embed";
        VisionRequest request = new VisionRequest
        {
            Image = image,
            Mode = mode.ToLowerInvariant() switch
            {
                "detect" => VisionMode.Detect,
                "segment" => VisionMode.Segment,
                "embed" => VisionMode.Embed,
                _ => throw new ArgumentException($"Unknown vision mode '{mode}'. Use embed, detect, or segment."),
            },
            Prompt = parameters.Get("query"),
            Threshold = parameters.GetFloat("confidence", 0.25f),
        };
        VisionResult result = await engine.Vision.RunAsync(spec, request, cancel).ConfigureAwait(false);

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
        if (result.Masks is not null)
        {
            artifact.Meta["masks"] = result.Masks.Count.ToString(CultureInfo.InvariantCulture);
        }
        return artifact;
    }

    /// <summary>Text-to-video through <see cref="IVideoService"/>; frames are collected from the stream and written as
    /// a numbered PNG sequence, since there is no single-file video artifact yet.</summary>
    private static async Task<GeneratedArtifact> VideoAsync(
        IInferenceEngine engine, ModelSpec spec, string prompt, ParamState parameters, string? outputDir, bool quiet, CancellationToken cancel)
    {
        VideoRequest request = new VideoRequest
        {
            Prompt = prompt,
            NegativePrompt = parameters.Get("negative") is { Length: > 0 } negative
                ? negative
                : "blurry, low quality, distorted, watermark",
            Width = parameters.GetIntOrNull("width"),
            Height = parameters.GetIntOrNull("height"),
            Steps = parameters.GetIntOrNull("steps"),
            CfgScale = parameters.GetFloatOrNull("cfg"),
            Frames = parameters.GetIntOrNull("frames"),
            Fps = parameters.GetIntOrNull("fps"),
            Seed = parameters.GetInt("seed", -1),
        };

        ConsoleStepProgress? progress = quiet ? null : new ConsoleStepProgress("denoise");
        List<VideoFrame> frames = new List<VideoFrame>();
        try
        {
            await foreach (VideoFrame frame in engine.Video.GenerateAsync(spec, request, progress, cancel).ConfigureAwait(false))
            {
                frames.Add(frame);
            }
        }
        finally
        {
            progress?.Finish();
        }
        return FrameArtifact(frames, outputDir, prompt, "video");
    }

    /// <summary>Image-to-3D through <see cref="IMeshService"/>; the CLI's "prompt" is the input image path.</summary>
    private static async Task<GeneratedArtifact> MeshAsync(
        IInferenceEngine engine, ModelSpec spec, string prompt, ParamState parameters, bool quiet, CancellationToken cancel)
    {
        MeshRequest request = new MeshRequest
        {
            Image = LoadImage(prompt),
            Steps = parameters.GetInt("steps", 0),
            GridResolution = parameters.GetInt("grid", 0),
            Seed = parameters.GetInt("seed", -1),
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
            Text = $"{result.Format.ToUpperInvariant()} mesh ({result.Data.Length / 1024} KB)",
        };
        artifact.Meta["format"] = result.Format;
        return artifact;
    }

    /// <summary>World rollout through <see cref="IWorldService"/>: queues one "forward" action per requested frame,
    /// then drains the session's frame stream. The session batches the queued plan into a single rollout (the Oasis
    /// pipeline is one-shot), so this reproduces the canned forward walk the CLI has always driven.</summary>
    private static async Task<GeneratedArtifact> WorldAsync(
        IInferenceEngine engine, ModelSpec spec, string prompt, ParamState parameters, string? outputDir, bool quiet, CancellationToken cancel)
    {
        int totalFrames = Math.Max(2, parameters.GetInt("frames", 16));
        WorldRequest request = new WorldRequest
        {
            InitImage = LoadImage(prompt),
            Steps = parameters.GetInt("steps", 10),
            Seed = parameters.GetInt("seed", -1),
        };

        ConsoleStepProgress? progress = quiet ? null : new ConsoleStepProgress("rollout");
        List<VideoFrame> frames = new List<VideoFrame>();
        using IWorldSession session = engine.World.Open(spec, request);
        for (int i = 0; i < totalFrames - 1; i++)
        {
            session.SendAction("forward");
        }
        try
        {
            await foreach (VideoFrame frame in session.StreamAsync(cancel).ConfigureAwait(false))
            {
                frames.Add(frame);
                progress?.Report(new StepPreview { Step = frames.Count, TotalSteps = totalFrames });
            }
        }
        finally
        {
            progress?.Finish();
        }
        return FrameArtifact(frames, outputDir, Path.GetFileNameWithoutExtension(prompt.Trim().Trim('"')), "world");
    }

    /// <summary>Writes a frame sequence to disk and describes it, previewing the first frame inline.</summary>
    private static GeneratedArtifact FrameArtifact(IReadOnlyList<VideoFrame> frames, string? outputDir, string slug, string label)
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
        string dir = FrameWriter.WriteFrames(pixels, width, height, outputDir ?? RepoPaths.OutputRoot(), slug);

        GeneratedArtifact artifact = new GeneratedArtifact
        {
            Kind = ArtifactKind.Video,
            Extension = "png",
            Text = $"{frames.Count} frames ({width}x{height}) → {dir}",
            PreviewRgb = frames[0].Rgb,
            PreviewWidth = width,
            PreviewHeight = height,
            SelfWritten = true,
        };
        artifact.Meta["frames"] = frames.Count.ToString(CultureInfo.InvariantCulture);
        artifact.Meta["size"] = $"{width}x{height}";
        return artifact;
    }

    /// <summary>Reads an audio file into an <see cref="AudioClip"/>, or null when <paramref name="path"/> is null/empty
    /// (an unset optional reference). Throws if a path was given but does not exist.</summary>
    private static AudioClip? LoadAudioClip(string? path)
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

    /// <summary>Writes each named stem as its own WAV into a fresh subdirectory (mirroring <see cref="FrameWriter"/>'s
    /// "one call, N files" shape) since a stem set has no single-file artifact.</summary>
    private static GeneratedArtifact StemsArtifact(StemsResult stems, string? outputDir, string slug)
    {
        if (stems.Stems.Count == 0)
        {
            throw new InvalidOperationException("The separation model produced no stems.");
        }
        string baseDir = outputDir ?? RepoPaths.OutputRoot();
        Directory.CreateDirectory(baseDir);
        string dir = NextStemsDir(baseDir, Slug.Make(slug));
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

    private static string NextStemsDir(string baseDir, string slug)
    {
        for (int i = 1; i < 100000; i++)
        {
            string candidate = Path.Combine(baseDir, $"{slug}-stems-{i:D3}");
            if (!Directory.Exists(candidate))
            {
                return candidate;
            }
        }
        return Path.Combine(baseDir, $"{slug}-stems-{Guid.NewGuid():N}");
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

    /// <summary>Reads an image file (PNG or BMP) into the engine's RGB24 <see cref="ImageData"/> contract, since the
    /// typed requests take pixels rather than the file path the CLI accepts.</summary>
    private static ImageData LoadImage(string promptPath)
    {
        string path = promptPath.Trim().Trim('"');
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"Input image not found: {path}");
        }
        (byte[] rgb, int width, int height) = path.EndsWith(".bmp", StringComparison.OrdinalIgnoreCase)
            ? BmpEncoder.Decode(File.ReadAllBytes(path))
            : PngDecoder.DecodeFromFile(path);
        return new ImageData { Rgb = rgb, Width = width, Height = height };
    }
}
