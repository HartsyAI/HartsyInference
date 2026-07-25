using HartsyInference.Engine.Services;
using Spectre.Console;

namespace HartsyInference.Cli.Dispatch;

/// <summary>Renders a service's <see cref="StepPreview"/> ticks as an in-place step counter on the current line.</summary>
/// <remarks>Safe to interleave with plain console writes because it holds no Spectre live display; call <see cref="Finish"/>
/// once the run ends to close the line.</remarks>
public sealed class ConsoleStepProgress : IProgress<StepPreview>
{
    private readonly string _label;
    private bool _wrote;

    /// <summary>Creates a counter labelled <paramref name="label"/> (e.g. "denoise", "mesh", "rollout").</summary>
    public ConsoleStepProgress(string label) => _label = label;

    /// <inheritdoc/>
    public void Report(StepPreview value)
    {
        string counter = value.TotalSteps > 0 ? $"{value.Step}/{value.TotalSteps}" : value.Step.ToString();
        string line = $"  {_label} [{counter}]";
        int width = Console.IsOutputRedirected ? line.Length : Math.Min(Console.WindowWidth - 1, 120);
        Console.Write("\r" + line.PadRight(Math.Max(width, line.Length)));
        _wrote = true;
    }

    /// <summary>Terminates the counter line if anything was ever reported.</summary>
    public void Finish()
    {
        if (_wrote)
        {
            AnsiConsole.WriteLine();
            _wrote = false;
        }
    }
}
