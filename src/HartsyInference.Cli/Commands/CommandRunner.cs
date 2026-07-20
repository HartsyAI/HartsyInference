using HartsyInference.Cli.Dispatch;
using HartsyInference.Cli.Infra;
using Spectre.Console;

namespace HartsyInference.Cli.Commands;

/// <summary>Shared generation lifecycle for the modality commands: header, run via the engine's typed services,
/// artifact save, cancellation, and error handling. The engine owns backend + model load; this only adds the
/// CLI-side header, result presentation, and persistence.</summary>
public static class CommandRunner
{
    /// <summary>Runs one generation end to end through the engine facade.</summary>
    public static int Run(
        Modality modality,
        ModelSpec spec,
        string prompt,
        ParamState parameters,
        string backendSelector,
        bool quiet,
        string? outputDir,
        string headerLabel,
        bool showResponseRule)
    {
        using CancellationTokenSource cts = new CancellationTokenSource();
        ConsoleCancelEventHandler onCancel = (_, e) =>
        {
            e.Cancel = true;
            cts.Cancel();
        };
        Console.CancelKeyPress += onCancel;

        using InferenceEngine engine = new InferenceEngine(backendSelector);
        try
        {
            spec = ModelAcquisition.EnsurePresent(spec);

            if (!quiet)
            {
                AnsiConsole.MarkupLine($"[#9aa4af]model[/] [{CliTheme.Accent}]{Markup.Escape(headerLabel)}[/]   " +
                    $"[#9aa4af]backend[/] [{CliTheme.Accent}]{Markup.Escape(engine.BackendDescription)}[/]");
            }

            if (!quiet && showResponseRule)
                AnsiConsole.Write(new Rule($"[{CliTheme.Accent}]response[/]").LeftJustified().RuleStyle("grey"));

            GeneratedArtifact artifact = GenerationDispatch
                .RunAsync(engine, spec, prompt, parameters, outputDir, quiet, cts.Token)
                .GetAwaiter().GetResult();
            ResultPresenter.Present(artifact, quiet);

            string? saved = ArtifactWriter.Write(artifact, outputDir, prompt, force: outputDir is not null);
            if (saved is not null)
                AnsiConsole.MarkupLine($"[#9aa4af]saved[/] [{CliTheme.Accent}]{Markup.Escape(saved)}[/]");

            return 0;
        }
        catch (OperationCanceledException)
        {
            AnsiConsole.MarkupLine("\n[yellow]Cancelled.[/]");
            return 130;
        }
        catch (Exception ex)
        {
            return CliErrors.Report(ex, modality);
        }
        finally
        {
            Console.CancelKeyPress -= onCancel;
        }
    }
}
