using HartsyInference.Engine;
using Spectre.Console;

namespace HartsyInference.Cli.Infra;

/// <summary>Ensures a selected catalog model's files are on disk before use: if any preset asset (transformer / text
/// encoder / VAE) is missing, it lists them, asks the user to confirm, and downloads the complete set into the right
/// folders — the SwarmUI "pick a model and it just fetches everything" flow. Returns the spec with its resolved local
/// path once present; leaves it unchanged (so the handler surfaces the normal not-found error) if the user declines.</summary>
public static class ModelAcquisition
{
    /// <summary>Downloads <paramref name="spec"/>'s catalog assets if missing (with confirmation), returning the spec
    /// pointed at the now-present primary file. No-op for models with no preset assets.</summary>
    public static ModelSpec EnsurePresent(ModelSpec spec)
    {
        if (spec.Catalog is not { } cat || cat.Assets.Count == 0)
            return spec;

        IReadOnlyList<ModelAsset> missing = ModelDownloader.MissingAssets(cat);
        if (missing.Count == 0)
        {
            string? present = ModelDownloader.PrimaryLocalPath(cat);
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

        string? primary = ModelDownloader.PrimaryLocalPath(cat);
        AnsiConsole.MarkupLine("[green]✓ model files ready.[/]");
        return primary is not null ? spec with { LocalPath = primary } : spec;
    }
}
