using System.ComponentModel;
using System.Globalization;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.Quant;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>Writes a quantized copy of a checkpoint. Offline and explicit: the engine never quantizes at load, so
/// this is how a smaller file comes to exist, and what a later generation runs is that file rather than a runtime
/// decision about it.</summary>
public sealed class QuantizeCommand : Command<QuantizeCommand.Settings>
{
    /// <summary>Options for <c>hartsy quantize</c>.</summary>
    public sealed class Settings : CommandSettings
    {
        /// <summary>Checkpoint to read; any container the engine can open, including a GGUF being made smaller.</summary>
        [CommandOption("--model-path")]
        [Description("Checkpoint to quantize (.safetensors or .gguf).")]
        public string ModelPath { get; init; } = "";

        /// <summary>File to write.</summary>
        [CommandOption("-o|--out")]
        [Description("Output .gguf path.")]
        public string Out { get; init; } = "";

        /// <summary>Precision preset.</summary>
        [CommandOption("--quant")]
        [Description("Precision: Q8_0, Q6_K, Q5_K_M, Q4_K_M or Q4_K_S.")]
        public string Quant { get; init; } = "Q8_0";

        /// <summary>Value for the output's <c>general.architecture</c>, which is how a reader picks a key mapper.</summary>
        [CommandOption("--arch")]
        [Description("Architecture id written into the output metadata.")]
        public string? Architecture { get; init; }

        /// <summary>Whether an existing output may be replaced.</summary>
        [CommandOption("--overwrite")]
        [Description("Replace the output file if it already exists.")]
        public bool Overwrite { get; init; }
    }

    /// <summary>Named presets, matched case-insensitively so <c>q4_k_m</c> works as well as <c>Q4_K_M</c>.</summary>
    private static readonly Dictionary<string, GgufQuantPolicy> _policies = new(StringComparer.OrdinalIgnoreCase)
    {
        ["Q8_0"] = GgufQuantPolicy.Q8_0,
        ["Q6_K"] = GgufQuantPolicy.Q6_K,
        ["Q5_K_M"] = GgufQuantPolicy.Q5_K_M,
        ["Q4_K_M"] = GgufQuantPolicy.Q4_K_M,
        ["Q4_K_S"] = GgufQuantPolicy.Q4_K_S,
    };

    /// <inheritdoc/>
    public override int Execute(CommandContext context, Settings settings)
    {
        if (string.IsNullOrWhiteSpace(settings.ModelPath))
        {
            AnsiConsole.MarkupLine("[red]Error:[/] --model-path is required.");
            return 1;
        }
        if (string.IsNullOrWhiteSpace(settings.Out))
        {
            AnsiConsole.MarkupLine("[red]Error:[/] -o/--out is required.");
            return 1;
        }
        if (!_policies.TryGetValue(settings.Quant, out GgufQuantPolicy? policy))
        {
            AnsiConsole.MarkupLine(
                $"[red]Error:[/] unknown --quant '{settings.Quant}'. Known: {string.Join(", ", _policies.Keys)}.");
            return 1;
        }

        QuantizationJob job = new()
        {
            SourcePath = settings.ModelPath,
            OutputPath = settings.Out,
            Target = new QuantizationTarget(QuantizationTargetKind.Gguf, policy),
            Architecture = settings.Architecture,
            Overwrite = settings.Overwrite,
        };
        QuantizationReport report = CheckpointQuantizer.Quantize(job);
        AnsiConsole.MarkupLine(
            $"[green]Wrote[/] {settings.Out}\n"
            + $"  {report.TensorCount} tensors, {report.QuantizedCount} quantized\n"
            + $"  {Gib(report.SourceBytes)} → {Gib(report.OutputBytes)} "
            + $"({report.Ratio.ToString("P1", CultureInfo.InvariantCulture)} of source)");
        return 0;
    }

    private static string Gib(long bytes) =>
        (bytes / (1024.0 * 1024 * 1024)).ToString("F2", CultureInfo.InvariantCulture) + " GiB";
}
