using Spectre.Console;

namespace HartsyInference.Cli.Dispatch;

/// <summary>Default progress sink: grey stage lines, an in-place step counter, and inline token streaming. Safe to
/// interleave with plain console writes because it does not hold a Spectre live display.</summary>
public sealed class ConsoleProgressSink : IProgressSink
{
    private string _stepLabel = "";
    private int _stepTotal;

    /// <inheritdoc/>
    public void Stage(string message)
    {
        if (string.IsNullOrEmpty(message))
            return;
        AnsiConsole.MarkupLine($"[#9aa4af]{Markup.Escape(message)}[/]");
    }

    /// <inheritdoc/>
    public void BeginSteps(string label, int totalSteps)
    {
        _stepLabel = label;
        _stepTotal = totalSteps;
    }

    /// <inheritdoc/>
    public void Step(int current, string? detail = null)
    {
        string counter = _stepTotal > 0 ? $"{current}/{_stepTotal}" : current.ToString();
        string line = detail is null ? $"  {_stepLabel} [{counter}]" : $"  {_stepLabel} [{counter}] {detail}";
        Console.Write("\r" + line.PadRight(Math.Min(Console.IsOutputRedirected ? line.Length : Console.WindowWidth - 1, 120)));
    }

    /// <inheritdoc/>
    public void Token(string text) => Console.Write(text);

    /// <inheritdoc/>
    public void EndSteps(string? summary = null)
    {
        Console.WriteLine();
        if (!string.IsNullOrEmpty(summary))
            AnsiConsole.MarkupLine($"[#9aa4af]{Markup.Escape(summary)}[/]");
    }
}
