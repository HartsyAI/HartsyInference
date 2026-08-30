using HartsyInference.Core.Logging;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>Locates MiniMax-H3's four component files, which ship in two incompatible layouts.</summary>
/// <remarks><para>The vendor (ModelScope / MiniMaxAI) publishes a <b>folder tree</b> — <c>transformer/</c>,
/// <c>video_vae/</c>, <c>audio_vae/</c>, <c>text_encoder/</c>, optionally nested under an <c>FL2VA</c>/<c>Ref2VA</c>
/// variant folder. Comfy-Org repackages the same weights <b>flat</b>, one file per component spread across
/// <c>diffusion_models/</c>, <c>vae/</c> and <c>text_encoders/</c>, and that is the layout SwarmUI's own H3 support
/// downloads into. A checkpoint path therefore identifies either a folder root or just the DiT file.</para>
/// <para>Flat-layout components are found by walking up from the DiT to a models root and searching known component
/// folders. Nothing is downloaded: SwarmUI has already fetched these files, and re-fetching them under the engine's
/// own model directory would duplicate ~5.8 GB of VAEs plus a 16-51 GB text encoder.</para></remarks>
public sealed record MiniMaxH3Assets
{
    /// <summary>The DiT weights.</summary>
    public required string Dit { get; init; }

    public required string VideoVae { get; init; }

    /// <summary>Null when the checkpoint ships no audio VAE — video still decodes, the soundtrack is dropped.</summary>
    public string? AudioVae { get; init; }

    public required string TextEncoder { get; init; }

    /// <summary>Folder holding <c>vocab.json</c> + <c>merges.txt</c>; null falls back to the embedded Qwen BPE, which the flat layout has no tokenizer files for.</summary>
    public string? TokenizerDir { get; init; }

    /// <summary>True when the components came from a vendor folder tree rather than the flat repack.</summary>
    public required bool IsFolderLayout { get; init; }

    private static readonly string[] _variantFolders = ["FL2VA", "Ref2VA"];
    // "vae"/"VAE" cover Comfy and SwarmUI. Do NOT add SwarmUI's "Video" folder: it holds video assets, not VAEs, and
    // a recursive name match there would hand a non-VAE file to LoadWeights.
    private static readonly string[] _vaeFolders = ["vae", "VAE"];
    private static readonly string[] _textEncoderFolders = ["text_encoders", "clip", "llm"];

    /// <summary>Resolves every component from whatever the caller pointed at.</summary>
    public static MiniMaxH3Assets Resolve(string checkpointPath) => Resolve(checkpointPath, null);

    /// <summary>Resolves the component set while allowing explicit request overrides to replace missing defaults.</summary>
    public static MiniMaxH3Assets Resolve(string checkpointPath, ComponentOverrides? components)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(checkpointPath);
        string? root = FolderRoot(checkpointPath);
        if (root is not null)
        {
            return FromFolder(root, components);
        }
        if (!File.Exists(checkpointPath))
        {
            throw new FileNotFoundException(
                $"MiniMax-H3 checkpoint not found at '{checkpointPath}' (expected a folder with a transformer/ "
                + "subfolder, or the DiT .safetensors file itself).");
        }
        return FromFlat(checkpointPath, components);
    }

    /// <summary>The vendor folder root for <paramref name="path"/>, or null when this is the flat layout.</summary>
    private static string? FolderRoot(string path)
    {
        if (File.Exists(path))
        {
            // <root>/transformer/dit.safetensors — only the transformer/ parent identifies the tree.
            string? dir = Path.GetDirectoryName(path);
            string? parent = dir is null ? null : Path.GetDirectoryName(dir);
            bool underTransformer = dir is not null
                && string.Equals(Path.GetFileName(dir), "transformer", StringComparison.OrdinalIgnoreCase);
            return underTransformer && parent is not null && Directory.Exists(Path.Combine(parent, "video_vae"))
                ? parent : null;
        }
        if (!Directory.Exists(path))
        {
            return null;
        }
        if (Directory.Exists(Path.Combine(path, "transformer")))
        {
            return path;
        }
        foreach (string variant in _variantFolders)
        {
            string candidate = Path.Combine(path, variant);
            if (Directory.Exists(Path.Combine(candidate, "transformer")))
            {
                return candidate;
            }
        }
        return null;
    }

    private static MiniMaxH3Assets FromFolder(string root, ComponentOverrides? components)
    {
        string? audioDir = Directory.Exists(Path.Combine(root, "audio_vae")) ? Path.Combine(root, "audio_vae") : null;
        string? tokenizer = null;
        foreach (string sub in new[] { "tokenizer", "processor", "text_encoder" })
        {
            string dir = Path.Combine(root, sub);
            if (File.Exists(Path.Combine(dir, "vocab.json")) && File.Exists(Path.Combine(dir, "merges.txt")))
            {
                tokenizer = dir;
                break;
            }
        }
        Logs.Info($"[MiniMaxH3Assets] Folder layout at {root}.");
        return new MiniMaxH3Assets
        {
            Dit = FirstSafetensors(Path.Combine(root, "transformer")),
            VideoVae = components?.VideoVae ?? components?.Vae ?? FirstSafetensors(Path.Combine(root, "video_vae")),
            AudioVae = components?.AudioVae ?? (audioDir is null ? null : FirstSafetensors(audioDir)),
            TextEncoder = components?.Qwen ?? FirstSafetensors(Path.Combine(root, "text_encoder")),
            TokenizerDir = tokenizer,
            IsFolderLayout = true,
        };
    }

    private static MiniMaxH3Assets FromFlat(string ditFile, ComponentOverrides? components)
    {
        string? ditDir = Path.GetDirectoryName(ditFile);
        List<string> roots = [];
        for (string? d = ditDir; d is not null && roots.Count < 4; d = Path.GetDirectoryName(d))
        {
            roots.Add(d);
        }
        string video = components?.VideoVae ?? components?.Vae
            ?? FindFlat(roots, _vaeFolders, ["video_vae"], preferQuantized: false)
            ?? throw new FileNotFoundException(
                $"MiniMax-H3 video VAE not found near '{ditFile}'. The flat layout expects a file whose name contains "
                + "'minimax_h3' and 'video_vae' under a vae/ folder (SwarmUI downloads it to Models/vae/MiniMaxH3/).");
        string? audio = components?.AudioVae ?? FindFlat(roots, _vaeFolders, ["audio_vae"]);
        string text = components?.Qwen ?? FindFlat(roots, _textEncoderFolders, ["qwen3vl", "qwen3_vl"])
            ?? throw new FileNotFoundException(
                $"MiniMax-H3 text encoder not found near '{ditFile}'. Expected a Qwen3-VL file whose name contains "
                + "'minimax_h3' under a text_encoders/ folder.");
        if (audio is null)
        {
            Logs.Warning("[MiniMaxH3Assets] No audio VAE found — video will decode without its soundtrack.");
        }
        Logs.Info($"[MiniMaxH3Assets] Flat layout: video VAE {video}, audio VAE {audio ?? "(none)"}, text {text}.");
        return new MiniMaxH3Assets
        {
            Dit = ditFile,
            VideoVae = video,
            AudioVae = audio,
            TextEncoder = text,
            TokenizerDir = null,
            IsFolderLayout = false,
        };
    }

    /// <summary>Searches each root's component folders for a <c>minimax_h3</c> file also matching one of
    /// <paramref name="hints"/>. Quantized text encoders remain the practical default, but a validation-pending
    /// quantized video VAE must be selected explicitly instead of displacing the proven FP16/BF16 component.</summary>
    private static string? FindFlat(List<string> roots, string[] folders, string[] hints,
        bool preferQuantized = true)
    {
        foreach (string root in roots)
        {
            foreach (string folder in folders)
            {
                string dir = Path.Combine(root, folder);
                if (!Directory.Exists(dir))
                {
                    continue;
                }
                string? best = Directory.EnumerateFiles(dir, "*.safetensors", SearchOption.AllDirectories)
                    .Where(f => Matches(Path.GetFileName(f), hints))
                    .OrderBy(f => FormatRank(Path.GetFileName(f), preferQuantized))
                    .ThenBy(f => new FileInfo(f).Length)
                    .FirstOrDefault();
                if (best is not null)
                {
                    return best;
                }
            }
        }
        return null;
    }

    /// <summary>Ranks quantized variants before or after dense precision according to the component's release
    /// status. File size breaks ties within the selected precision class.</summary>
    /// <remarks><c>int8_convrot</c> used to rank last because the engine could not load it at all; it is now a
    /// first-class resident format (<c>CudaBackend</c>'s int8 IMMA path), so it ranks with the other quantized
    /// builds and the size tie-break below decides between them.</remarks>
    private static int FormatRank(string name, bool preferQuantized)
    {
        string lower = name.ToLowerInvariant();
        bool quantized = lower.Contains("nvfp4") || lower.Contains("fp8") || lower.Contains("int8");
        return quantized == preferQuantized ? 0 : 1;
    }

    private static bool Matches(string name, string[] hints)
    {
        string lower = name.ToLowerInvariant();
        if (!lower.Contains("minimax") || !lower.Contains("h3"))
        {
            return false;
        }
        foreach (string hint in hints)
        {
            if (lower.Contains(hint))
            {
                return true;
            }
        }
        return false;
    }

    private static string FirstSafetensors(string dir)
    {
        if (!Directory.Exists(dir))
        {
            throw new FileNotFoundException($"MiniMax-H3 component folder not found: {dir}");
        }
        string[] files = Directory.GetFiles(dir, "*.safetensors", SearchOption.TopDirectoryOnly);
        if (files.Length == 0)
        {
            files = Directory.GetFiles(dir, "*.safetensors", SearchOption.AllDirectories);
        }
        return files.Length > 0 ? files.OrderBy(f => f, StringComparer.Ordinal).First()
            : throw new FileNotFoundException($"No .safetensors in {dir}.");
    }
}
