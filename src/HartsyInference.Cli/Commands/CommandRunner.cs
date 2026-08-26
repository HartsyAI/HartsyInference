using System.Globalization;
using HartsyInference.Cli.Dispatch;
using HartsyInference.Cli.Infra;
using Spectre.Console;

namespace HartsyInference.Cli.Commands;

/// <summary>Shared generation lifecycle for the modality commands: header, run, artifact save, cancellation, error handling.</summary>
/// <remarks>The engine owns backend + model load; this only adds the CLI-side header, result presentation, and persistence.</remarks>
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
        bool showResponseRule,
        int? gpuOrdinal = null,
        EngineOptions? engineOptions = null)
    {
        using CancellationTokenSource cts = new CancellationTokenSource();
        using IDisposable cancelBinding = BindCancelKey(cts);
        using InferenceEngine engine = gpuOrdinal.HasValue || engineOptions is not null
            ? new InferenceEngine(backendSelector, gpuOrdinal ?? 0, engineOptions)
            : new InferenceEngine(backendSelector);
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
    }

    /// <summary>Wires Ctrl+C to cancel <paramref name="cts"/> instead of killing the process; dispose the result to unhook.</summary>
    internal static IDisposable BindCancelKey(CancellationTokenSource cts)
    {
        ConsoleCancelEventHandler onCancel = (_, e) =>
        {
            e.Cancel = true;
            cts.Cancel();
        };
        Console.CancelKeyPress += onCancel;
        return new CancelKeyBinding(onCancel);
    }

    /// <summary>Returns false and prints <paramref name="message"/> in red when <paramref name="value"/> is null/whitespace.</summary>
    internal static bool RequireNonEmpty(string? value, string message, out int exitCode)
    {
        if (!string.IsNullOrWhiteSpace(value))
        {
            exitCode = 0;
            return true;
        }
        AnsiConsole.MarkupLine($"[red]{message}[/]");
        exitCode = 1;
        return false;
    }

    /// <summary>Returns false when both <paramref name="model"/> and <paramref name="modelPath"/> are null/whitespace.</summary>
    internal static bool RequireModelOrPath(string? model, string? modelPath, string modelFlag, string modelPathFlag, out int exitCode)
    {
        if (!string.IsNullOrWhiteSpace(model) || !string.IsNullOrWhiteSpace(modelPath))
        {
            exitCode = 0;
            return true;
        }
        AnsiConsole.MarkupLine($"[red]Specify a model with[/] [#2ea5e0]{modelFlag}[/] [red]or[/] [#2ea5e0]{modelPathFlag}[/][red].[/]");
        exitCode = 1;
        return false;
    }

    /// <summary>Picks a display label: catalog id, else checkpoint file name, else the model token, else <paramref name="fallback"/>.</summary>
    internal static string ResolveLabel(ModelSpec spec, string model, string? modelPath = null, string? fallback = null)
    {
        if (spec.Catalog?.Id is { } catalogId)
            return catalogId;
        if (modelPath is { Length: > 0 } path)
            return Path.GetFileName(path);
        if (model.Length > 0)
            return model;
        return fallback ?? model;
    }

    /// <summary>Forwards a tunable only when the flag was passed; an omitted flag reaches the engine as null.</summary>
    internal static void PutIfSet(this ParamState parameters, string key, IFormattable? value)
    {
        if (value is not null)
        {
            parameters.Put(key, value.ToString(null, CultureInfo.InvariantCulture));
        }
    }

    /// <summary>Forwards a string tunable only when the user actually passed a non-empty value.</summary>
    internal static void PutIfSet(this ParamState parameters, string key, string? value)
    {
        if (!string.IsNullOrWhiteSpace(value))
        {
            parameters.Put(key, value);
        }
    }

    private sealed class CancelKeyBinding : IDisposable
    {
        private readonly ConsoleCancelEventHandler _handler;

        public CancelKeyBinding(ConsoleCancelEventHandler handler)
        {
            _handler = handler;
        }

        public void Dispose()
        {
            Console.CancelKeyPress -= _handler;
        }
    }
}
