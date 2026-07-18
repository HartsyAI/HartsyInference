using HartsyInference.Cli.Infra;
using Spectre.Console;

namespace HartsyInference.Cli.Dispatch;

/// <summary>Renders a <see cref="GeneratedArtifact"/> to the console uniformly for both the one-shot commands and the
/// REPL: streamed text is finalized, non-streamed text is printed, file artifacts get a summary, and metadata is a
/// grey footer.</summary>
public static class ResultPresenter
{
    /// <summary>Prints <paramref name="artifact"/>. In <paramref name="quiet"/> mode only the essential text is emitted
    /// (no rules, headers, or metadata).</summary>
    public static void Present(GeneratedArtifact artifact, bool quiet)
    {
        bool isText = artifact.Kind is ArtifactKind.Text or ArtifactKind.Data;

        if (quiet)
        {
            if (isText)
                AnsiConsole.WriteLine(artifact.Text ?? "");
            return;
        }

        if (isText)
        {
            if (artifact.Streamed)
            {
                AnsiConsole.WriteLine();
            }
            else
            {
                AnsiConsole.Write(new Rule().RuleStyle("grey"));
                AnsiConsole.WriteLine(artifact.Text ?? "");
            }
        }
        else
        {
            if (artifact.PreviewRgb is { Length: > 0 } rgb && TerminalImage.IsSupported)
            {
                AnsiConsole.WriteLine();
                TerminalImage.Render(rgb, artifact.PreviewWidth, artifact.PreviewHeight);
            }
            if (!string.IsNullOrEmpty(artifact.Text))
                AnsiConsole.MarkupLine($"[grey]{Markup.Escape(artifact.Text)}[/]");
        }

        string footer = string.Join(" · ", artifact.Meta
            .Where(kv => !kv.Key.Equals("model", StringComparison.OrdinalIgnoreCase))
            .Select(kv => $"{kv.Key} {kv.Value}"));
        if (footer.Length > 0)
            AnsiConsole.MarkupLine($"[grey]{Markup.Escape(footer)}[/]");
    }
}
