using System.ComponentModel;
using HartsyInference.Cli.Infra;
using HartsyInference.Core.Logging;
using HartsyInference.ModelAssets.Registry;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>Downloads a model from HuggingFace (or registers a local path) into the on-disk cache.</summary>
public sealed class PullCommand : AsyncCommand<PullCommand.Settings>
{
    /// <summary>Options for <c>hartsy pull</c>.</summary>
    public sealed class Settings : CommandSettings
    {
        /// <summary>HuggingFace repo id (e.g. "stabilityai/sdxl") or a local file/directory path.</summary>
        [CommandArgument(0, "<model>")]
        [Description("HuggingFace repo id or local path to download and cache.")]
        public string Model { get; init; } = "";
    }

    /// <inheritdoc/>
    public override async Task<int> ExecuteAsync(CommandContext context, Settings settings)
    {
        if (!CommandRunner.RequireNonEmpty(settings.Model, "A model repo id or local path is required.", out int exitCode))
            return exitCode;

        using CancellationTokenSource cts = new CancellationTokenSource();
        using IDisposable cancelBinding = CommandRunner.BindCancelKey(cts);

        // Catalog ids first ("chroma", "qwen-image", ...): pull the pinned asset set into the Models tree —
        // the same flow `hartsy image -m <id>` uses, minus the confirmation prompt (pull IS the confirmation).
        // Previously these fell through to the raw HF path and died on a 401 for a nonexistent bare repo id.
        CatalogEntry? catalogEntry = ModelCatalog.Find(settings.Model);
        if (catalogEntry is not null && catalogEntry.Assets.Count > 0)
        {
            return await PullCatalogAsync(catalogEntry, cts.Token).ConfigureAwait(false);
        }

        using ModelRegistry registry = new ModelRegistry();
        ModelCacheStore cache = new ModelCacheStore();

        try
        {
            LoadedModel? loaded = null;
            await AnsiConsole.Progress()
                .Columns(new ProgressColumn[]
                {
                    new TaskDescriptionColumn(),
                    new ProgressBarColumn(),
                    new PercentageColumn(),
                    new SpinnerColumn(),
                })
                .StartAsync(async ctx =>
                {
                    ProgressTask task = ctx.AddTask($"[{CliTheme.Accent}]pulling[/] {Markup.Escape(settings.Model)}", maxValue: 100);
                    Progress<double> progress = new Progress<double>(p => task.Value = Math.Clamp(p, 0, 1) * 100);
                    loaded = await registry.LoadAsync(settings.Model, cache, downloadProgress: progress, ct: cts.Token).ConfigureAwait(false);
                    task.Value = 100;
                }).ConfigureAwait(false);

            if (loaded is null)
            {
                AnsiConsole.MarkupLine("[red]Pull failed: no model returned.[/]");
                return 1;
            }

            string arch = loaded.Info.Architecture is { Length: > 0 } a ? a : "unknown";
            AnsiConsole.MarkupLine($"[green]✓[/] pulled [{CliTheme.Accent}]{Markup.Escape(settings.Model)}[/]");
            AnsiConsole.MarkupLine($"  [#9aa4af]architecture:[/] {Markup.Escape(arch)}");
            AnsiConsole.MarkupLine($"  [#9aa4af]path:[/] {Markup.Escape(loaded.Info.LocalPath)}");
            registry.Unload(settings.Model);
            return 0;
        }
        catch (OperationCanceledException)
        {
            AnsiConsole.MarkupLine("[yellow]Pull cancelled.[/]");
            return 130;
        }
        catch (Exception ex)
        {
            Logs.Error($"Pull failed for '{settings.Model}'", ex);
            AnsiConsole.MarkupLine($"[red]Pull failed:[/] {Markup.Escape(ex.Message)}");
            return 1;
        }
    }

    /// <summary>Downloads a catalog entry's missing assets (transformer + side models) into the Models tree.</summary>
    private static async Task<int> PullCatalogAsync(CatalogEntry cat, CancellationToken ct)
    {
        try
        {
            // Audio-cache-backed entries load via AudioModelCache at runtime — pulling them into the
            // Models tree would double-download. Same branch EnsurePresent takes, minus the confirm.
            if (ModelAcquisition.UsesAudioCache(cat, cat.Modality))
            {
                if (!ModelAcquisition.EnsureAudioAssetsPresent(cat, cat.Modality, confirm: false, ct))
                    return 1;
                AnsiConsole.MarkupLine($"[green]✓[/] pulled [{CliTheme.Accent}]{Markup.Escape(cat.DisplayName)}[/]");
                return 0;
            }

            IReadOnlyList<ModelAsset> missing = ModelDownloader.MissingAssets(cat);
            if (missing.Count == 0)
            {
                AnsiConsole.MarkupLine($"[green]✓[/] [{CliTheme.Accent}]{Markup.Escape(cat.DisplayName)}[/] already present.");
                string? path = ModelDownloader.PrimaryLocalPath(cat);
                if (path is not null)
                    AnsiConsole.MarkupLine($"  [#9aa4af]path:[/] {Markup.Escape(path)}");
                return 0;
            }

            AnsiConsole.MarkupLine($"[{CliTheme.Accent}]{Markup.Escape(cat.DisplayName)}[/] [#9aa4af]needs {missing.Count} file(s):[/]");
            ModelAcquisition.ListMissingAssets(missing, showTargetDir: true);
            await ModelAcquisition.DownloadWithProgressAsync(missing, ct).ConfigureAwait(false);

            AnsiConsole.MarkupLine($"[green]✓[/] pulled [{CliTheme.Accent}]{Markup.Escape(cat.DisplayName)}[/]");
            string? primary = ModelDownloader.PrimaryLocalPath(cat);
            if (primary is not null)
                AnsiConsole.MarkupLine($"  [#9aa4af]path:[/] {Markup.Escape(primary)}");
            return 0;
        }
        catch (OperationCanceledException)
        {
            AnsiConsole.MarkupLine("[yellow]Pull cancelled.[/]");
            return 130;
        }
        catch (Exception ex)
        {
            Logs.Error($"Catalog pull failed for '{cat.Id}'", ex);
            AnsiConsole.MarkupLine($"[red]Pull failed:[/] {Markup.Escape(ex.Message)}");
            return 1;
        }
    }
}
