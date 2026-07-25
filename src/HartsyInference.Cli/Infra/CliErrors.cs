using HartsyInference.Core.Logging;
using Spectre.Console;

namespace HartsyInference.Cli.Infra;

/// <summary>Uniform error handling: expected user-facing errors print a clean message, unexpected failures are logged in full.</summary>
/// <remarks>Shared by the one-shot commands and the REPL.</remarks>
public static class CliErrors
{
    /// <summary>Reports <paramref name="ex"/> for a generation in <paramref name="modality"/>. Never rethrows.</summary>
    /// <returns>The exit code (1 for a failure).</returns>
    public static int Report(Exception ex, Modality modality)
    {
        if (IsExpected(ex))
            Logs.Warning($"{Modalities.ToCliName(modality)}: {ex.Message}");
        else
            Logs.Error($"{Modalities.ToCliName(modality)} generation failed", ex);
        AnsiConsole.MarkupLine($"[red]Error:[/] {Markup.Escape(ex.Message)}");
        return 1;
    }

    /// <summary>True for exceptions that represent a user/config mistake rather than a bug (no stack dump needed).</summary>
    private static bool IsExpected(Exception ex) => ex is FileNotFoundException or DirectoryNotFoundException
        or NotSupportedException or ArgumentException or InvalidOperationException;
}
