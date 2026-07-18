using HartsyInference.Cli.Dispatch;
using HartsyInference.Cli.Infra;
using HartsyInference.Core.Backends;
using Spectre.Console;

namespace HartsyInference.Cli.Commands;

/// <summary>Shared generation lifecycle for the modality commands: header, backend construction, model load, run,
/// artifact save, cancellation, and error handling. Each command supplies only its result presentation.</summary>
public static class CommandRunner
{
    /// <summary>The process-wide dispatch registry, shared by every command and the REPL.</summary>
    public static ModalityDispatch Dispatch { get; } = new ModalityDispatch();

    /// <summary>Runs one generation end to end. <paramref name="present"/> renders the modality-specific result
    /// after <paramref name="prompt"/> is processed; artifact persistence and lifecycle are handled here.</summary>
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
        IProgressSink sink = quiet ? new NullProgressSink() : new ConsoleProgressSink();

        using CancellationTokenSource cts = new CancellationTokenSource();
        ConsoleCancelEventHandler onCancel = (_, e) =>
        {
            e.Cancel = true;
            cts.Cancel();
        };
        Console.CancelKeyPress += onCancel;

        IBackend? backend = null;
        IModalityRunner? runner = null;
        try
        {
            if (!quiet)
            {
                AnsiConsole.MarkupLine($"[#9aa4af]model[/] [{CliTheme.Accent}]{Markup.Escape(headerLabel)}[/]   " +
                    $"[#9aa4af]backend[/] [{CliTheme.Accent}]{Markup.Escape(BackendFactory.Describe(backendSelector))}[/]");
            }

            backend = BackendFactory.Create(backendSelector);
            IModalityHandler handler = Dispatch.Get(modality);
            runner = handler.Load(spec, backend, sink);

            if (!quiet && showResponseRule)
                AnsiConsole.Write(new Rule($"[{CliTheme.Accent}]response[/]").LeftJustified().RuleStyle("grey"));

            GeneratedArtifact artifact = handler.Run(runner, prompt, parameters, sink, cts.Token);
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
            runner?.Dispose();
            backend?.Dispose();
        }
    }
}
