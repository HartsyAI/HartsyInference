using HartsyInference.Engine.Services;
using Spectre.Console;

namespace HartsyInference.Cli.Dispatch;

/// <summary>Renders a service's <see cref="StepPreview"/> ticks as an in-place step counter on the current line.</summary>
/// <remarks>Safe to interleave with plain console writes because it holds no Spectre live display; call <see cref="Finish"/>
/// once the run ends to close the line.</remarks>
public sealed class ConsoleStepProgress : IProgress<StepPreview>
{
    private readonly string _label;
    private readonly string? _previewOutput;
    private bool _wrote;

    /// <summary>Creates a counter labelled <paramref name="label"/> (e.g. "denoise", "mesh", "rollout").</summary>
    /// <param name="previewOutput">Optional PNG path for the latest static preview. Temporal previews are written
    /// beside it as numbered PNG frames on every tick.</param>
    public ConsoleStepProgress(string label, string? previewOutput = null)
    {
        _label = label;
        _previewOutput = previewOutput;
    }

    /// <inheritdoc/>
    public void Report(StepPreview value)
    {
        string counter = value.TotalSteps > 0 ? $"{value.Step}/{value.TotalSteps}" : value.Step.ToString();
        string line = $"  {_label} [{counter}]";
        int width = Console.IsOutputRedirected ? line.Length : Math.Min(Console.WindowWidth - 1, 120);
        Console.Write("\r" + line.PadRight(Math.Max(width, line.Length)));
        _wrote = true;
        WritePreview(value);
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

    /// <summary>Writes an opt-in preview snapshot for terminals that cannot display pixels inline.</summary>
    private void WritePreview(StepPreview preview)
    {
        if (string.IsNullOrWhiteSpace(_previewOutput) || preview.PreviewRgb is null ||
            preview.PreviewWidth <= 0 || preview.PreviewHeight <= 0)
        {
            return;
        }

        string fullPath = Path.GetFullPath(_previewOutput);
        Directory.CreateDirectory(Path.GetDirectoryName(fullPath)!);
        File.WriteAllBytes(fullPath, PngEncoder.Encode(
            preview.PreviewRgb, preview.PreviewWidth, preview.PreviewHeight));
        if (preview.PreviewFramesRgb is not { Length: > 1 } frames)
        {
            return;
        }

        string directory = Path.GetDirectoryName(fullPath)!;
        string stem = Path.GetFileNameWithoutExtension(fullPath);
        for (int i = 0; i < frames.Length; i++)
        {
            string framePath = Path.Combine(directory, $"{stem}-{i:D4}.png");
            File.WriteAllBytes(framePath, PngEncoder.Encode(
                frames[i], preview.PreviewWidth, preview.PreviewHeight));
        }
    }
}
