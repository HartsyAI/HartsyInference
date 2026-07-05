using HartsyInference.Core.Logging;
using Spectre.Console;

namespace HartsyInference.Cli.Infra;

/// <summary>Uniform error handling for generation: expected user-facing errors print a clean message (no stack), while
/// unexpected failures are logged in full. Shared by the one-shot commands and the REPL.</summary>
public static class CliErrors
{
    /// <summary>Reports <paramref name="ex"/> for a generation in <paramref name="modality"/>. Returns the exit code
    /// (1 for a failure). Rethrows nothing.</summary>
    public static int Report(Exception ex, Modality modality)
    {
        if (IsExpected(ex))
        {
            Logs.Warning($"{Modalities.ToCliName(modality)}: {ex.Message}");
            AnsiConsole.MarkupLine($"[red]Error:[/] {Markup.Escape(ex.Message)}");
            return 1;
        }
        Logs.Error($"{Modalities.ToCliName(modality)} generation failed", ex);
        AnsiConsole.MarkupLine($"[red]Error:[/] {Markup.Escape(ex.Message)}");
        return 1;
    }

    /// <summary>True for exceptions that represent a user/config mistake rather than a bug (no stack dump needed).</summary>
    public static bool IsExpected(Exception ex) => ex is FileNotFoundException or DirectoryNotFoundException
        or NotSupportedException or ArgumentException or InvalidOperationException;
}
