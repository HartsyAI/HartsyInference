using System.ComponentModel;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.MoePack;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>Packs a GGUF checkpoint's routed experts into a quantized expert pack, or verifies one against its source.</summary>
public sealed class MoePackCommand : Command<MoePackCommand.Settings>
{
    /// <summary>Options for <c>hartsy moe pack</c>.</summary>
    public sealed class Settings : CommandSettings
    {
        /// <summary>GGUF checkpoint to read.</summary>
        [CommandArgument(0, "<gguf>")]
        [Description("GGUF checkpoint with stacked expert tensors.")]
        public string Gguf { get; init; } = "";

        /// <summary>Directory to create the pack in.</summary>
        [CommandOption("-o|--out")]
        [Description("Pack directory to create. Refused if it already holds a completed pack.")]
        public string Out { get; init; } = "";

        /// <summary>Quantized dtype of every projection.</summary>
        [CommandOption("--dtype")]
        [Description("Pack dtype: Q8_0, Q4_K, Q5_K or Q6_K.")]
        public string DType { get; init; } = "Q8_0";

        /// <summary>Runtime topology fingerprint to bind the pack to.</summary>
        [CommandOption("--topology-fingerprint")]
        [Description("Runtime topology fingerprint (SparseModelTopology.Fingerprint) to bind the pack to. Omit for the GGUF-geometry fingerprint.")]
        public string? TopologyFingerprint { get; init; }
    }

    /// <inheritdoc/>
    public override int Execute(CommandContext context, Settings settings)
    {
        if (string.IsNullOrWhiteSpace(settings.Out))
        {
            AnsiConsole.MarkupLine("[red]Error:[/] -o/--out is required.");
            return 1;
        }
        if (!TryParseDType(settings.DType, out DType dtype))
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] '{settings.DType}' is not a pack dtype; use Q8_0, Q4_K, Q5_K or Q6_K.");
            return 1;
        }
        try
        {
            GgufExpertPack.Write(settings.Gguf, settings.Out, dtype, settings.TopologyFingerprint);
            AnsiConsole.MarkupLine($"[green]Packed[/] {Markup.Escape(settings.Gguf)} into {Markup.Escape(settings.Out)} as {dtype.Name}.");
            return 0;
        }
        catch (Exception ex) when (ex is InvalidDataException or InvalidOperationException or ArgumentException or IOException)
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] {Markup.Escape(ex.Message)}");
            return 1;
        }
    }

    internal static bool TryParseDType(string name, out DType dtype)
    {
        foreach (DType candidate in new[] { DType.Q8_0, DType.Q4_K, DType.Q5_K, DType.Q6_K })
        {
            if (string.Equals(candidate.Name, name, StringComparison.OrdinalIgnoreCase))
            {
                dtype = candidate;
                return true;
            }
        }
        dtype = DType.F32;
        return false;
    }
}
