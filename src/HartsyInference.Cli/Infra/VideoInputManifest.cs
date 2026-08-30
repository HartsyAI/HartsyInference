using System.Globalization;
using System.Text.Json;
using System.Text.Json.Serialization;
using HartsyInference.Core.Numerics;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Cli.Infra;

/// <summary>Parses CLI guide/control manifests and their repeatable shorthand flags into typed Engine inputs.</summary>
internal static class VideoInputManifest
{
    private static readonly JsonSerializerOptions JsonOptions = new JsonSerializerOptions
    {
        PropertyNameCaseInsensitive = true,
        Converters = { new JsonStringEnumConverter(JsonNamingPolicy.CamelCase) },
    };

    /// <summary>Builds and conflict-checks arbitrary visual/audio guides.</summary>
    internal static IReadOnlyList<VideoGuide>? Guides(ParamState parameters)
    {
        Dictionary<int, MutableGuide> guides = new Dictionary<int, MutableGuide>();
        AddGuideFlags(guides, parameters.GetStringOrNull("guide-images"), GuidePayload.Image);
        AddGuideFlags(guides, parameters.GetStringOrNull("guide-videos"), GuidePayload.Video);
        AddGuideFlags(guides, parameters.GetStringOrNull("guide-audios"), GuidePayload.Audio);
        string? manifestPath = parameters.GetStringOrNull("guides-manifest");
        if (!string.IsNullOrWhiteSpace(manifestPath))
        {
            foreach (GuideEntry entry in ReadEntries<GuideManifest, GuideEntry>(manifestPath, value => value.Guides,
                         ResolveGuidePaths))
            {
                MutableGuide target = Get(guides, entry.Frame);
                if (!string.IsNullOrWhiteSpace(entry.Image))
                {
                    Set(target, GuidePayload.Image, entry.Image, entry.Frame);
                }
                if (!string.IsNullOrWhiteSpace(entry.Video))
                {
                    Set(target, GuidePayload.Video, entry.Video, entry.Frame);
                }
                if (!string.IsNullOrWhiteSpace(entry.Audio))
                {
                    Set(target, GuidePayload.Audio, entry.Audio, entry.Frame);
                }
                if (!string.IsNullOrWhiteSpace(entry.Fit))
                {
                    target.Fit = ParseFit(entry.Fit);
                }
            }
        }
        if (guides.Count == 0)
        {
            return null;
        }
        List<VideoGuide> result = new List<VideoGuide>(guides.Count);
        foreach (KeyValuePair<int, MutableGuide> pair in guides.OrderBy(pair => pair.Key))
        {
            MutableGuide value = pair.Value;
            if (value.Image is null && value.Video is null && value.Audio is null)
            {
                throw new ArgumentException($"Guide at frame {pair.Key} has no image, video, or audio payload.");
            }
            result.Add(new VideoGuide
            {
                FrameIndex = pair.Key,
                Image = value.Image is null ? null : LoadImage(value.Image),
                Video = value.Video is null ? null : LoadVideo(value.Video),
                Audio = value.Audio is null ? null : LoadAudio(value.Audio),
                FitMode = value.Fit,
            });
        }
        return result;
    }

    /// <summary>Builds one shorthand control plus every control-manifest entry.</summary>
    internal static IReadOnlyList<VideoControl>? Controls(ParamState parameters)
    {
        List<ControlEntry> entries = new List<ControlEntry>();
        string? simpleModel = parameters.GetStringOrNull("control-model");
        string? simpleVideo = parameters.GetStringOrNull("control-video");
        if (simpleModel is not null || simpleVideo is not null)
        {
            if (string.IsNullOrWhiteSpace(simpleModel) || string.IsNullOrWhiteSpace(simpleVideo))
            {
                throw new ArgumentException("--control-model and --control-video must be supplied together.");
            }
            entries.Add(new ControlEntry
            {
                Model = simpleModel,
                Video = simpleVideo,
                Kind = parameters.GetStringOrNull("control-kind") ?? "custom",
                Strength = parameters.GetDoubleOrNull("control-strength") ?? 1.0,
                Start = parameters.GetDoubleOrNull("control-start") ?? 0.0,
                End = parameters.GetDoubleOrNull("control-end") ?? 1.0,
            });
            if (string.Equals(entries[^1].Kind, "inpaint", StringComparison.OrdinalIgnoreCase))
            {
                throw new ArgumentException(
                    "The simple control slot cannot describe H3 inpainting. Use --controls-manifest with visibilityMask and maskedSource paths.");
            }
        }
        string? manifestPath = parameters.GetStringOrNull("controls-manifest");
        if (!string.IsNullOrWhiteSpace(manifestPath))
        {
            entries.AddRange(ReadEntries<ControlManifest, ControlEntry>(manifestPath, value => value.Controls,
                ResolveControlPaths));
        }
        if (entries.Count == 0)
        {
            return null;
        }
        List<VideoControl> result = new List<VideoControl>(entries.Count);
        foreach (ControlEntry entry in entries)
        {
            if (string.IsNullOrWhiteSpace(entry.Model) || string.IsNullOrWhiteSpace(entry.Video))
            {
                throw new ArgumentException("Every control requires non-empty model and video paths.");
            }
            if (!Enum.TryParse(entry.Kind, true, out VideoControlKind kind))
            {
                throw new ArgumentException($"Unknown control kind '{entry.Kind}'.");
            }
            result.Add(new VideoControl
            {
                Model = ExistingPath(entry.Model),
                Video = LoadVideo(entry.Video),
                Kind = kind,
                Strength = entry.Strength,
                Start = entry.Start,
                End = entry.End,
                VisibilityMask = string.IsNullOrWhiteSpace(entry.VisibilityMask) ? null : LoadVideo(entry.VisibilityMask),
                MaskedSource = string.IsNullOrWhiteSpace(entry.MaskedSource) ? null : LoadVideo(entry.MaskedSource),
            });
        }
        return result;
    }

    /// <summary>Builds the video mask/source pair when either CLI path is present.</summary>
    internal static VideoDenoiseMask? VideoMask(ParamState parameters)
    {
        string? mask = parameters.GetStringOrNull("video-denoise-mask");
        string? source = parameters.GetStringOrNull("video-mask-source");
        if (mask is null && source is null)
        {
            return null;
        }
        if (string.IsNullOrWhiteSpace(mask))
        {
            throw new ArgumentException("--video-mask-source requires --video-denoise-mask.");
        }
        bool maskImage = IsImage(mask);
        bool sourceImage = source is not null && IsImage(source);
        return new VideoDenoiseMask
        {
            MaskImage = maskImage ? LoadImage(mask) : null,
            MaskVideo = maskImage ? null : LoadVideo(mask),
            SourceImage = sourceImage ? LoadImage(source!) : null,
            SourceVideo = source is null || sourceImage ? null : LoadVideo(source),
        };
    }

    /// <summary>Builds continuous audio mask samples from JSON or a comma/whitespace-delimited file.</summary>
    internal static AudioDenoiseMask? AudioMask(ParamState parameters)
    {
        string? mask = parameters.GetStringOrNull("audio-denoise-mask");
        string? source = parameters.GetStringOrNull("audio-mask-source");
        if (mask is null && source is null)
        {
            return null;
        }
        if (string.IsNullOrWhiteSpace(mask))
        {
            throw new ArgumentException("--audio-mask-source requires --audio-denoise-mask.");
        }
        string path = ExistingPath(mask);
        string text = File.ReadAllText(path);
        float[]? json = null;
        try
        {
            json = JsonSerializer.Deserialize<float[]>(text, JsonOptions);
        }
        catch (JsonException)
        {
            // Fall through to the intentionally simple delimited representation.
        }
        float[] values = json ?? text.Split([',', ' ', '\t', '\r', '\n'], StringSplitOptions.RemoveEmptyEntries)
            .Select(token => float.Parse(token, NumberStyles.Float, CultureInfo.InvariantCulture)).ToArray();
        if (values.Length == 0 || values.Any(value => !UnitInterval.Contains(value)))
        {
            throw new ArgumentException("Audio denoise mask values must be a non-empty finite sequence in [0,1].");
        }
        float rate = parameters.GetFloatOrNull("audio-mask-rate") ?? 40f;
        if (!(rate > 0f) || !float.IsFinite(rate))
        {
            throw new ArgumentException("--audio-mask-rate must be finite and greater than zero.");
        }
        return new AudioDenoiseMask
        {
            Values = values,
            Rate = rate,
            Source = source is null ? null : LoadAudio(source),
        };
    }

    private static void AddGuideFlags(Dictionary<int, MutableGuide> guides, string? joined, GuidePayload payload)
    {
        if (string.IsNullOrWhiteSpace(joined))
        {
            return;
        }
        foreach (string token in joined.Split('\n', StringSplitOptions.RemoveEmptyEntries))
        {
            int equals = token.IndexOf('=');
            if (equals <= 0 || !int.TryParse(token.AsSpan(0, equals), NumberStyles.Integer,
                    CultureInfo.InvariantCulture, out int frame) || equals == token.Length - 1)
            {
                throw new ArgumentException($"Guide '{token}' must use FRAME=PATH syntax.");
            }
            Set(Get(guides, frame), payload, token[(equals + 1)..], frame);
        }
    }

    private static MutableGuide Get(Dictionary<int, MutableGuide> guides, int frame)
    {
        if (!guides.TryGetValue(frame, out MutableGuide? guide))
        {
            guide = new MutableGuide();
            guides.Add(frame, guide);
        }
        return guide;
    }

    private static void Set(MutableGuide guide, GuidePayload payload, string path, int frame)
    {
        string? existing = payload switch
        {
            GuidePayload.Image => guide.Image,
            GuidePayload.Video => guide.Video,
            _ => guide.Audio,
        };
        if (existing is not null)
        {
            throw new ArgumentException($"Guide frame {frame} has duplicate {payload.ToString().ToLowerInvariant()} payloads.");
        }
        if ((payload == GuidePayload.Image && guide.Video is not null)
            || (payload == GuidePayload.Video && guide.Image is not null))
        {
            throw new ArgumentException($"Guide frame {frame} cannot contain both an image and video payload.");
        }
        switch (payload)
        {
            case GuidePayload.Image: guide.Image = path; break;
            case GuidePayload.Video: guide.Video = path; break;
            case GuidePayload.Audio: guide.Audio = path; break;
            default: throw new ArgumentOutOfRangeException(nameof(payload));
        }
    }

    private static VideoGuideFitMode ParseFit(string fit) => fit.Trim().ToLowerInvariant() switch
    {
        "cover" => VideoGuideFitMode.Cover,
        "contain" => VideoGuideFitMode.Contain,
        "stretch" => VideoGuideFitMode.Stretch,
        _ => throw new ArgumentException($"Unknown guide fit mode '{fit}'."),
    };

    private static IReadOnlyList<TEntry> ReadEntries<TManifest, TEntry>(string rawPath,
        Func<TManifest, IReadOnlyList<TEntry>?> select, Action<TEntry, string> resolvePaths)
    {
        string path = ExistingPath(rawPath);
        string baseDirectory = Path.GetDirectoryName(Path.GetFullPath(path))
            ?? throw new ArgumentException($"Manifest '{path}' has no parent directory.");
        string json = File.ReadAllText(path);
        using JsonDocument document = JsonDocument.Parse(json);
        IReadOnlyList<TEntry> entries;
        if (document.RootElement.ValueKind == JsonValueKind.Array)
        {
            TEntry[]? direct = JsonSerializer.Deserialize<TEntry[]>(json, JsonOptions);
            entries = direct ?? throw new ArgumentException($"Manifest '{path}' contains no entries.");
        }
        else
        {
            TManifest manifest = JsonSerializer.Deserialize<TManifest>(json, JsonOptions)
                ?? throw new ArgumentException($"Manifest '{path}' contains no entries.");
            entries = select(manifest)
                ?? throw new ArgumentException($"Manifest '{path}' contains no entries.");
        }
        foreach (TEntry entry in entries)
        {
            resolvePaths(entry, baseDirectory);
        }
        return entries;
    }

    private static void ResolveGuidePaths(GuideEntry entry, string baseDirectory)
    {
        entry.Image = ResolveManifestPath(entry.Image, baseDirectory);
        entry.Video = ResolveManifestPath(entry.Video, baseDirectory);
        entry.Audio = ResolveManifestPath(entry.Audio, baseDirectory);
    }

    private static void ResolveControlPaths(ControlEntry entry, string baseDirectory)
    {
        entry.Model = ResolveManifestPath(entry.Model, baseDirectory);
        entry.Video = ResolveManifestPath(entry.Video, baseDirectory);
        entry.VisibilityMask = ResolveManifestPath(entry.VisibilityMask, baseDirectory);
        entry.MaskedSource = ResolveManifestPath(entry.MaskedSource, baseDirectory);
    }

    private static string? ResolveManifestPath(string? raw, string baseDirectory)
    {
        if (string.IsNullOrWhiteSpace(raw))
        {
            return raw;
        }
        string path = raw.Trim().Trim('"');
        return Path.IsPathRooted(path) ? path : Path.GetFullPath(Path.Combine(baseDirectory, path));
    }

    private static bool IsImage(string path) => Path.GetExtension(path).ToLowerInvariant() is ".png" or ".bmp" or ".jpg" or ".jpeg";

    private static string ExistingPath(string raw)
    {
        string path = raw.Trim().Trim('"');
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"Video input file not found: {path}", path);
        }
        return path;
    }

    private static ImageData LoadImage(string raw)
    {
        string path = ExistingPath(raw);
        (byte[] rgb, int width, int height) = ImageIo.DecodeFile(path);
        return new ImageData { Rgb = rgb, Width = width, Height = height };
    }

    private static VideoClip LoadVideo(string raw)
    {
        string path = ExistingPath(raw);
        string extension = Path.GetExtension(path).TrimStart('.').ToLowerInvariant();
        return new VideoClip { Data = File.ReadAllBytes(path), Format = extension.Length == 0 ? null : extension };
    }

    private static AudioClip LoadAudio(string raw)
    {
        string path = ExistingPath(raw);
        string extension = Path.GetExtension(path).TrimStart('.').ToLowerInvariant();
        return new AudioClip { Data = File.ReadAllBytes(path), Format = extension };
    }

    private enum GuidePayload
    {
        Image,
        Video,
        Audio,
    }

    private sealed class MutableGuide
    {
        internal string? Image { get; set; }
        internal string? Video { get; set; }
        internal string? Audio { get; set; }
        internal VideoGuideFitMode Fit { get; set; } = VideoGuideFitMode.Cover;
    }

    private sealed class GuideManifest
    {
        public List<GuideEntry>? Guides { get; set; }
    }

    private sealed class GuideEntry
    {
        public int Frame { get; set; }
        public string? Image { get; set; }
        public string? Video { get; set; }
        public string? Audio { get; set; }
        public string? Fit { get; set; }
    }

    private sealed class ControlManifest
    {
        public List<ControlEntry>? Controls { get; set; }
    }

    private sealed class ControlEntry
    {
        public string? Model { get; set; }
        public string? Video { get; set; }
        public string Kind { get; set; } = "custom";
        public double Strength { get; set; } = 1.0;
        public double Start { get; set; }
        public double End { get; set; } = 1.0;
        public string? VisibilityMask { get; set; }
        public string? MaskedSource { get; set; }
    }
}
