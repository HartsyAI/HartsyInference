using System.ComponentModel;
using HartsyInference.Core.Backends;
using HartsyInference.ModelAssets.MoePack;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>Checks an expert pack's checksums and values against the GGUF it was packed from.</summary>
public sealed class MoeVerifyCommand : Command<MoeVerifyCommand.Settings>
{
    /// <summary>Options for <c>hartsy moe verify</c>.</summary>
    public sealed class Settings : CommandSettings
    {
        /// <summary>GGUF checkpoint the pack came from.</summary>
        [CommandArgument(0, "<gguf>")]
        [Description("GGUF checkpoint the pack was built from.")]
        public string Gguf { get; init; } = "";

        /// <summary>Pack directory to check.</summary>
        [CommandArgument(1, "<pack>")]
        [Description("Pack directory to verify.")]
        public string Pack { get; init; } = "";

        /// <summary>Topology fingerprint the pack must have been built for.</summary>
        [CommandOption("--expect-fingerprint")]
        [Description("Runtime topology fingerprint the pack must match. Omit to expect the GGUF-geometry fingerprint.")]
        public string? ExpectFingerprint { get; init; }
    }

    /// <inheritdoc/>
    public override int Execute(CommandContext context, Settings settings)
    {
        try
        {
            ExpertPackVerification result = GgufExpertPack.Verify(settings.Gguf, settings.Pack, settings.ExpectFingerprint);
            AnsiConsole.MarkupLine(
                $"Checked {result.Checked} experts: relative RMSE {result.RelativeRmse:G4}, max abs error {result.MaxAbsError:G4}.");
            if (result.Failures.Count == 0)
            {
                AnsiConsole.MarkupLine("[green]Verified.[/]");
                return 0;
            }
            AnsiConsole.MarkupLine($"[red]{result.Failures.Count} expert(s) failed their checksum:[/]");
            foreach (ExpertKey key in result.Failures)
                AnsiConsole.MarkupLine($"  layer {key.Layer} expert {key.Expert}");
            return 1;
        }
        catch (Exception ex) when (ex is InvalidDataException or IOException or ArgumentException)
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] {Markup.Escape(ex.Message)}");
            return 1;
        }
    }
}
