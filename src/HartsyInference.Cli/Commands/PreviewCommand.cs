using System.ComponentModel;
using System.Globalization;
using HartsyInference.Cli.Infra;
using HartsyInference.Vision.Codec;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>Renders an existing image file (PNG or the CLI's own BMP artifacts) inline in the terminal using truecolor
/// half-blocks — a quick way to eyeball a generated image without leaving the shell.</summary>
public sealed class PreviewCommand : Command<PreviewCommand.Settings>
{
    /// <summary>Options for <c>hartsy preview</c>.</summary>
    public sealed class Settings : CommandSettings
    {
        /// <summary>Path to the image to display.</summary>
        [CommandArgument(0, "<image>")]
        [Description("Path to a PNG or BMP image to display in the terminal.")]
        public string Image { get; init; } = "";

        /// <summary>Preview width in terminal columns.</summary>
        [CommandOption("-w|--width")]
        [Description("Preview width in terminal columns (default 56).")]
        public int Width { get; init; } = 56;
    }

    /// <inheritdoc/>
    public override int Execute(CommandContext context, Settings settings)
    {
        if (!File.Exists(settings.Image))
        {
            AnsiConsole.MarkupLine($"[red]Image not found:[/] {Markup.Escape(settings.Image)}");
            return 1;
        }

        (byte[] rgb, int width, int height) = Decode(settings.Image);
        if (!TerminalImage.IsSupported)
        {
            AnsiConsole.MarkupLine("[yellow]Inline preview unavailable[/] [#9aa4af](stdout is redirected, NO_COLOR, or HARTSY_NO_IMAGE=1).[/]");
            AnsiConsole.MarkupLine($"[#9aa4af]{width}x{height} · {Markup.Escape(Path.GetFileName(settings.Image))}[/]");
            return 0;
        }

        AnsiConsole.WriteLine();
        TerminalImage.Render(rgb, width, height, maxCellWidth: Math.Max(8, settings.Width));
        AnsiConsole.MarkupLine($"[#9aa4af]{width.ToString(CultureInfo.InvariantCulture)}x{height.ToString(CultureInfo.InvariantCulture)} · {Markup.Escape(Path.GetFileName(settings.Image))}[/]");
        return 0;
    }

    private static (byte[] rgb, int width, int height) Decode(string path)
    {
        if (path.EndsWith(".bmp", StringComparison.OrdinalIgnoreCase))
            return BmpEncoder.Decode(File.ReadAllBytes(path));
        return PngDecoder.DecodeFromFile(path);
    }
}
