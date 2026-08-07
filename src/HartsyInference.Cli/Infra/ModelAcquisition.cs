using HartsyInference.Audio.Cache;
using HartsyInference.Engine.Audio;
using Spectre.Console;

namespace HartsyInference.Cli.Infra;

/// <summary>Ensures a selected catalog model's files are on disk before use: the SwarmUI "pick a model, it fetches everything" flow.</summary>
/// <remarks>If any preset asset (transformer / text encoder / VAE) is missing, it lists them, asks the user to confirm,
/// and downloads the complete set into the right folders. Returns the spec with its resolved local path once present;
/// leaves it unchanged (so the handler surfaces the normal not-found error) if the user declines.</remarks>
public static class ModelAcquisition
{
    /// <summary>Modalities whose models self-download through <see cref="AudioModelCache"/>, a separate cache keyed by repo.</summary>
    /// <remarks>Not the SwarmUI-style <c>Models/&lt;subdir&gt;</c> tree <see cref="ModelDownloader"/> targets. Cache lives at
    /// <c>~/.cache/hartsyinference/models/</c>. These need their own present-check against it instead.</remarks>
    private static readonly HashSet<Modality> AudioCacheModalities = new()
    {
        Modality.Speech, Modality.Music, Modality.Transcribe, Modality.VoiceConvert, Modality.Fx,
    };

    /// <summary>Music catalog ids exempt from <see cref="AudioCacheModalities"/>: ACE-Step/YuE use <see cref="ModelDownloader"/> instead.</summary>
    /// <remarks>Same as image models, landing under <c>Models/audio/music/{acestep,yue}/</c> via <see cref="AudioWeightsCatalog"/> —
    /// so they must fall through to the code below instead of the audio-cache branch.</remarks>
    private static readonly HashSet<string> StandardDownloadMusicIds = new(StringComparer.OrdinalIgnoreCase)
    {
        AudioWeightsCatalog.AceStepId, AudioWeightsCatalog.YueId,
    };

    /// <summary>Downloads <paramref name="spec"/>'s catalog assets if missing (with confirmation). No-op for models with no preset assets.</summary>
    /// <returns>The spec pointed at the now-present primary file.</returns>
    public static ModelSpec EnsurePresent(ModelSpec spec)
    {
        if (spec.Catalog is not { } cat || cat.Assets.Count == 0)
            return spec;

        if (UsesAudioCache(cat, spec.Modality))
        {
            EnsureAudioAssetsPresent(cat, spec.Modality);
            return spec;
        }

        // Folder-checkpoint families (YuE) load a whole variant DIRECTORY — the primary asset is just the first
        // shard, so setting LocalPath to it would hand the loader a file path where it expects a folder. Leave
        // LocalPath alone for these; MusicCatalog.ResolveLocalCheckpoint finds the folder itself via the same
        // AudioWeightsCatalog data once the files are on disk.
        bool isFolderFamily = spec.Modality == Modality.Music && AudioWeightsCatalog.IsFolderCheckpoint(cat.Id);

        IReadOnlyList<ModelAsset> missing = ModelDownloader.MissingAssets(cat);
        if (missing.Count == 0)
        {
            string? present = isFolderFamily ? null : ModelDownloader.PrimaryLocalPath(cat);
            return spec.LocalPath is null && present is not null ? spec with { LocalPath = present } : spec;
        }

        AnsiConsole.MarkupLine($"[{CliTheme.Accent}]{Markup.Escape(cat.DisplayName)}[/] [#9aa4af]needs {missing.Count} file(s) not on disk:[/]");
        foreach (ModelAsset a in missing)
            AnsiConsole.MarkupLine($"  [#9aa4af]{a.Role}:[/] {Markup.Escape(a.Repo)}/{Markup.Escape(a.RepoPath)} [#9aa4af]→ Models/{Markup.Escape(a.TargetSubdir)}/[/]");

        if (!InteractivePrompt.Confirm("Download these now?", defaultYes: false))
            return spec;

        try
        {
            AnsiConsole.Progress()
                .Columns(new TaskDescriptionColumn(), new ProgressBarColumn(), new PercentageColumn(), new SpinnerColumn())
                .Start(ctx =>
                {
                    Dictionary<string, ProgressTask> tasks = new(StringComparer.Ordinal);
                    foreach (ModelAsset a in missing)
                        tasks[a.FileName] = ctx.AddTask(Markup.Escape(a.FileName), maxValue: 1.0);
                    ModelDownloader.DownloadAsync(missing, (a, fraction) =>
                    {
                        if (tasks.TryGetValue(a.FileName, out ProgressTask? task))
                            task.Value = fraction;
                    }, CancellationToken.None).GetAwaiter().GetResult();
                });
        }
        catch (Exception ex)
        {
            CliErrors.Report(ex, spec.Modality);
            return spec;
        }

        string? primary = isFolderFamily ? null : ModelDownloader.PrimaryLocalPath(cat);
        AnsiConsole.MarkupLine("[green]✓ model files ready.[/]");
        return primary is not null ? spec with { LocalPath = primary } : spec;
    }

    /// <summary>True when this catalog entry's files live under <see cref="AudioModelCache"/> rather than the
    /// <c>Models/&lt;subdir&gt;</c> tree — these must never be pulled via <see cref="ModelDownloader"/>.</summary>
    internal static bool UsesAudioCache(CatalogEntry cat, Modality modality)
        => AudioCacheModalities.Contains(modality) && !StandardDownloadMusicIds.Contains(cat.Id);

    /// <summary>Present-check + confirm + download for the audio-cache-backed modalities (TTS/STT/Music/VoiceConvert/Fx).</summary>
    /// <remarks>Resolves each asset's real location under <see cref="AudioModelCache"/> (not <see cref="ModelDownloader"/>'s
    /// SwarmUI-style <c>Models/&lt;subdir&gt;</c> tree, which these models never read from) and fetches anything missing via
    /// the same <see cref="AudioModelCache.GetAsync"/> the Engine's own descriptors call internally, so the file lands
    /// exactly where the generation call will look for it.</remarks>
    /// <returns>True when all assets are present (or were downloaded); false on decline or failure.</returns>
    internal static bool EnsureAudioAssetsPresent(CatalogEntry cat, Modality modality, bool confirm = true, CancellationToken ct = default)
    {
        string category = AudioCategoryFor(modality);
        List<ModelAsset> missing = cat.Assets.Where(a => !File.Exists(AudioAssetPath(a, category))).ToList();
        if (missing.Count == 0)
            return true;

        AnsiConsole.MarkupLine($"[{CliTheme.Accent}]{Markup.Escape(cat.DisplayName)}[/] [#9aa4af]needs {missing.Count} file(s) not on disk:[/]");
        foreach (ModelAsset a in missing)
            AnsiConsole.MarkupLine($"  [#9aa4af]{a.Role}:[/] {Markup.Escape(a.Repo)}/{Markup.Escape(a.RepoPath)}");

        if (confirm && !InteractivePrompt.Confirm("Download these now?", defaultYes: false))
            return false;

        try
        {
            AnsiConsole.Progress()
                .Columns(new TaskDescriptionColumn(), new SpinnerColumn())
                .Start(ctx =>
                {
                    foreach (ModelAsset a in missing)
                    {
                        ProgressTask task = ctx.AddTask(Markup.Escape(a.FileName));
                        string baseDescription = Markup.Escape(a.FileName);
                        IProgress<long> progress = new Progress<long>(bytes => task.Description = $"{baseDescription} ({bytes / (1024 * 1024)} MB)");
                        AudioModelCache.GetAsync(a.Repo, a.RepoPath, category, progress: progress, ct: ct).GetAwaiter().GetResult();
                        if (!string.IsNullOrEmpty(a.Sha256))
                            AudioModelCache.VerifySha256(AudioAssetPath(a, category), a.Sha256);
                    }
                });
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex)
        {
            CliErrors.Report(ex, modality);
            return false;
        }

        AnsiConsole.MarkupLine("[green]✓ model files ready.[/]");
        return true;
    }

    /// <summary>The real on-disk path for an audio asset: <see cref="AudioModelCache"/>'s directory, not <see cref="ModelDownloader"/>'s.</summary>
    /// <remarks>Audio descriptors call <c>AudioModelCache.GetAsync</c> directly and never read <c>ModelSpec.LocalPath</c>
    /// for HF-backed models, so <c>TargetSubdir</c>/<c>TargetName</c> on these catalog entries are display-only.</remarks>
    private static string AudioAssetPath(ModelAsset a, string category) => Path.Combine(AudioModelCache.GetRepoDirectory(a.Repo, category), a.RepoPath);

    /// <summary>Maps a CLI modality to the <see cref="AudioModelCache"/> category folder — mirrors
    /// <c>AudioWeights.CategorySubfolder</c> (SwarmUI extension) for the audio-cache-backed modalities.</summary>
    private static string AudioCategoryFor(Modality modality) => modality switch
    {
        Modality.Speech => "tts",
        Modality.Transcribe => "stt",
        Modality.Music => "music",
        Modality.VoiceConvert => "clone",
        Modality.Fx => "fx",
        _ => "misc",
    };
}
