using System.ComponentModel;
using System.Globalization;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Metadata;
using HartsyInference.ModelAssets.SafeTensors;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>Merges sharded safetensors into one file, streamed tensor by tensor. Offline: the merged file loads like any
/// other single-file checkpoint, and <c>hartsy quantize</c> takes it as a source.</summary>
public sealed class MergeCommand : Command<MergeCommand.Settings>
{
    /// <summary>Options for <c>hartsy merge</c>.</summary>
    public sealed class Settings : CommandSettings
    {
        /// <summary>The inputs to merge, in order.</summary>
        [CommandArgument(0, "<inputs>")]
        [Description("Shard files, a folder of .safetensors files, or a *.safetensors.index.json. A folder takes every .safetensors in it, so list the shard files when a folder holds others that must stay out.")]
        public string[] Inputs { get; init; } = [];

        /// <summary>File to write.</summary>
        [CommandOption("-o|--out")]
        [Description("Output .safetensors path.")]
        public string Out { get; init; } = "";

        /// <summary>Optional float cast.</summary>
        [CommandOption("--dtype")]
        [Description("Cast float tensors to bf16, fp16 or fp32. Omit to keep each tensor's stored dtype.")]
        public string? Dtype { get; init; }

        /// <summary>Catalog id whose identity the output carries.</summary>
        [CommandOption("--model")]
        [Description("Catalog id whose identity the output carries, e.g. iris-z-small. Without one SwarmUI cannot classify the file.")]
        public string? Model { get; init; }

        /// <summary>Upstream repository, recorded as provenance.</summary>
        [CommandOption("--source-repo")]
        [Description("Upstream repository recorded as provenance, e.g. speridlabs/iris-z.")]
        public string? SourceRepo { get; init; }

        /// <summary>Extra header entries.</summary>
        [CommandOption("--meta")]
        [Description("Extra header entry as KEY=VALUE. Repeatable; applied after the identity, so it wins on a clash.")]
        public string[] Meta { get; init; } = [];

        /// <summary>Whether an existing output may be replaced.</summary>
        [CommandOption("--overwrite")]
        [Description("Replace the output file if it already exists.")]
        public bool Overwrite { get; init; }
    }

    private static readonly Dictionary<string, DType> _dtypes = new(StringComparer.OrdinalIgnoreCase)
    {
        ["bf16"] = DType.BF16,
        ["fp16"] = DType.F16,
        ["fp32"] = DType.F32,
    };

    /// <inheritdoc/>
    public override int Execute(CommandContext context, Settings settings)
    {
        if (settings.Inputs.Length == 0 || settings.Inputs.Any(string.IsNullOrWhiteSpace))
        {
            AnsiConsole.MarkupLine("[red]Error:[/] at least one input is required: shard files, a folder or an index.");
            return 1;
        }
        if (string.IsNullOrWhiteSpace(settings.Out))
        {
            AnsiConsole.MarkupLine("[red]Error:[/] -o/--out is required.");
            return 1;
        }
        DType? cast = null;
        if (settings.Dtype is not null)
        {
            if (!_dtypes.TryGetValue(settings.Dtype, out DType dtype))
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] unknown --dtype '{Markup.Escape(settings.Dtype)}'. Known: {string.Join(", ", _dtypes.Keys)}.");
                return 1;
            }
            cast = dtype;
        }
        ArtifactIdentity? identity = null;
        if (!string.IsNullOrWhiteSpace(settings.Model))
        {
            identity = ModelIdentityCatalog.Find(settings.Model);
            if (identity is null)
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] unknown --model '{Markup.Escape(settings.Model)}'. Use a catalog id, e.g. iris-z-small.");
                return 1;
            }
        }
        Dictionary<string, string> meta = new(StringComparer.Ordinal);
        foreach (string pair in settings.Meta)
        {
            int equals = pair.IndexOf('=', StringComparison.Ordinal);
            if (equals <= 0)
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] --meta '{Markup.Escape(pair)}' should look like KEY=VALUE.");
                return 1;
            }
            meta[pair[..equals]] = pair[(equals + 1)..];
        }
        if (identity is null)
        {
            AnsiConsole.MarkupLine("[yellow]No --model:[/] the output carries no identity. SwarmUI cannot classify it; not for publishing.");
        }

        SafeTensorsMergeReport report;
        try
        {
            report = SafeTensorsMergeService.Merge(new SafeTensorsMergeJob
            {
                InputPaths = settings.Inputs,
                OutputPath = settings.Out,
                CastTo = cast,
                Identity = identity,
                SourceRepo = settings.SourceRepo,
                Metadata = meta,
                Overwrite = settings.Overwrite,
            });
        }
        catch (Exception ex) when (ex is HartsyInferenceException or InvalidDataException or IOException or UnauthorizedAccessException)
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] {Markup.Escape(ex.Message)}");
            return 1;
        }

        AnsiConsole.MarkupLine(
            $"[green]Wrote[/] {Markup.Escape(settings.Out)}\n"
            + $"  {report.TensorCount} tensors from {report.SourceFileCount} file(s), {Gib(report.OutputBytes)}\n"
            + $"  payload sha256 {report.PayloadSha256}");
        return 0;
    }

    private static string Gib(long bytes) =>
        (bytes / (1024.0 * 1024 * 1024)).ToString("F2", CultureInfo.InvariantCulture) + " GiB";
}
