using System.Text;
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

    /// <summary>Renders the framed app header: the H logo, the wordmark, and version / backend / working directory.</summary>
    public static void RenderBanner(string backendSelector)
    {
        StringBuilder body = new StringBuilder();
        for (int i = 0; i < LogoRows.Length; i++)
        {
            if (i > 0)
                body.Append('\n');
            body.Append($"[{LogoShades[i]}]{LogoRows[i]}[/]");
        }

        string version = typeof(CliTheme).Assembly.GetName().Version?.ToString(3) ?? "dev";
        string cwd = Directory.GetCurrentDirectory();
        body.Append("\n\n");
        body.Append($"[bold {Accent}]HARTSY[/] [white]INFERENCE[/]  [grey]· pure-C# AI inference[/]\n");
        body.Append($"[grey]v{version}  ·  backend[/] [{Accent}]{Markup.Escape(BackendFactory.Describe(backendSelector))}[/]  [grey]·  {Markup.Escape(cwd)}[/]");

        Panel panel = new Panel(new Markup(body.ToString()))
            .Border(BoxBorder.Rounded)
            .BorderColor(AccentColor)
            .Padding(3, 1, 3, 1);
        AnsiConsole.Write(panel);
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
