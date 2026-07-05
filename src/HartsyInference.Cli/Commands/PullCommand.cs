using System.ComponentModel;
using HartsyInference.Cli.Infra;
using HartsyInference.Core.Logging;
using HartsyInference.ModelHandler.Registry;
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
        if (string.IsNullOrWhiteSpace(settings.Model))
        {
            AnsiConsole.MarkupLine("[red]A model repo id or local path is required.[/]");
            return 1;
        }

        using CancellationTokenSource cts = new CancellationTokenSource();
        ConsoleCancelEventHandler onCancel = (_, e) =>
        {
            e.Cancel = true;
            cts.Cancel();
        };
        Console.CancelKeyPress += onCancel;

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
            AnsiConsole.MarkupLine($"  [grey]architecture:[/] {Markup.Escape(arch)}");
            AnsiConsole.MarkupLine($"  [grey]path:[/] {Markup.Escape(loaded.Info.LocalPath)}");
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
        finally
        {
            Console.CancelKeyPress -= onCancel;
        }
    }
}
