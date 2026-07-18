using System.Reflection;
using System.Text;
using HartsyInference.Vision.Codec;
using Spectre.Console;

namespace HartsyInference.Cli.Infra;

/// <summary>Shared Spectre styling: the Hartsy-blue palette, the H-logo header panel, and the status markup used by
/// every command.</summary>
public static class CliTheme
{
    /// <summary>Primary brand accent (Hartsy blue) as a Spectre markup color token.</summary>
    public const string Accent = "#2ea5e0";

    /// <summary>Primary brand accent as a Spectre <see cref="Color"/> (for borders, figlet, etc.).</summary>
    public static readonly Color AccentColor = new Color(46, 165, 224);

    /// <summary>The truecolor RGB of the accent, for raw-ANSI rendering in the line editor.</summary>
    public static readonly (byte R, byte G, byte B) AccentRgb = (46, 165, 224);

    // Top-lit vertical gradient echoing the blue-to-white swoosh in the Hartsy H mark.
    private static readonly string[] LogoShades = { "#bfe8fb", "#8cd2f5", "#5bbeef", "#2ea5e0", "#3f9fd6", "#2b8cc4", "#1c78b4" };

    private static readonly string[] LogoRows =
    {
        "██          ██",
        "██          ██",
        "██          ██",
        "██████████████",
        "██          ██",
        "██          ██",
        "██          ██",
    };

    // The exact background color baked into the embedded logo (see Assets/hartsy-h.png); rendered as terminal-default
    // so the logo's swoosh drops cleanly onto any terminal theme.
    private static readonly (byte R, byte G, byte B) LogoKey = (13, 17, 23);

    /// <summary>Renders the app header. On a real terminal this is the actual Hartsy H mark (the embedded logo drawn
    /// with truecolor half-blocks); when output can't show it (piped, NO_COLOR), it falls back to the framed
    /// block-glyph H. Both carry the wordmark and version / backend / working directory.</summary>
    public static void RenderBanner(string backendSelector)
    {
        string version = typeof(CliTheme).Assembly.GetName().Version?.ToString(3) ?? "dev";
        string backend = BackendFactory.Describe(backendSelector);
        string cwd = Directory.GetCurrentDirectory();

        if (TerminalImage.IsSupported && TryLoadLogo() is { } logo)
        {
            AnsiConsole.WriteLine();
            TerminalImage.Render(logo.Rgb, logo.Width, logo.Height, maxCellWidth: 30, indent: 3, transparentKey: LogoKey);
            AnsiConsole.MarkupLine($"   [bold {Accent}]HARTSY[/] [white]INFERENCE[/]  [grey]· pure-C# AI inference[/]");
            AnsiConsole.MarkupLine($"   [grey]v{version}  ·  backend[/] [{Accent}]{Markup.Escape(backend)}[/]  [grey]·  {Markup.Escape(cwd)}[/]");
            AnsiConsole.WriteLine();
            return;
        }

        StringBuilder body = new StringBuilder();
        for (int i = 0; i < LogoRows.Length; i++)
        {
            if (i > 0)
                body.Append('\n');
            body.Append($"[{LogoShades[i]}]{LogoRows[i]}[/]");
        }
        body.Append("\n\n");
        body.Append($"[bold {Accent}]HARTSY[/] [white]INFERENCE[/]  [grey]· pure-C# AI inference[/]\n");
        body.Append($"[grey]v{version}  ·  backend[/] [{Accent}]{Markup.Escape(backend)}[/]  [grey]·  {Markup.Escape(cwd)}[/]");

        Panel panel = new Panel(new Markup(body.ToString()))
            .Border(BoxBorder.Rounded)
            .BorderColor(AccentColor)
            .Padding(3, 1, 3, 1);
        AnsiConsole.Write(panel);
    }

    private static (byte[] Rgb, int Width, int Height)? _logo;
    private static bool _logoTried;

    /// <summary>Decodes the embedded Hartsy H logo once, or null if the resource is missing/undecodable.</summary>
    private static (byte[] Rgb, int Width, int Height)? TryLoadLogo()
    {
        if (_logoTried)
            return _logo;
        _logoTried = true;
        try
        {
            Assembly asm = typeof(CliTheme).Assembly;
            string? name = Array.Find(asm.GetManifestResourceNames(), n => n.EndsWith("hartsy-h.png", StringComparison.OrdinalIgnoreCase));
            if (name is null)
                return null;
            using Stream? stream = asm.GetManifestResourceStream(name);
            if (stream is null)
                return null;
            using MemoryStream ms = new MemoryStream();
            stream.CopyTo(ms);
            (byte[] rgb, int width, int height) = PngDecoder.Decode(ms.ToArray());
            _logo = (rgb, width, height);
        }
        catch (Exception ex)
        {
            Core.Logging.Logs.Warning($"CLI banner logo failed to load: {ex.Message}");
            _logo = null;
        }
        return _logo;
    }

    /// <summary>Spectre markup fragment for a status badge (emoji + colored label).</summary>
    public static string StatusMarkup(ModelStatus status) => status switch
    {
        ModelStatus.Verified => "[green]✅ verified[/]",
        ModelStatus.ValidationPending => "[yellow]🧪 validating[/]",
        ModelStatus.Structural => "[grey]🏗️ structural[/]",
        _ => "[grey]?[/]",
    };

    /// <summary>Short status word without color, for plain/JSON output.</summary>
    public static string StatusWord(ModelStatus status) => status switch
    {
        ModelStatus.Verified => "verified",
        ModelStatus.ValidationPending => "validating",
        ModelStatus.Structural => "structural",
        _ => "unknown",
    };
}
