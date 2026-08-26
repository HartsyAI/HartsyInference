using System.Reflection;
using System.Text;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Logging;
using HartsyInference.Vision.Codec;
using Spectre.Console;

namespace HartsyInference.Cli.Infra;

/// <summary>Shared Spectre styling: the Hartsy-blue palette, the H-logo header panel, and the status markup used by every command.</summary>
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

    /// <summary>Whether the terminal appears to use a light background, so muted text can pick a contrasting grey.</summary>
    /// <remarks>Determined once from <c>HARTSY_THEME</c> (dark/light) then <c>COLORFGBG</c>, defaulting to dark — the
    /// common developer setting and what any terminal that doesn't advertise its theme is assumed to be.</remarks>
    public static bool IsLightBackground { get; } = DetectLightBackground();

    /// <summary>Secondary/label text color that keeps contrast on the detected background.</summary>
    /// <remarks>Prefer this over a hardcoded <c>grey</c> (which is mid-tone and washes out on dark themes) for captions, footers,
    /// and hints.</remarks>
    public static string Muted => IsLightBackground ? "#5b636b" : "#9aa4af";

    /// <summary>The truecolor RGB of <see cref="Muted"/>, for raw-ANSI rendering (e.g. the line editor prompt).</summary>
    /// <remarks>Palette greys like <c>\e[90m</c> are unreadable because dark themes render them near the background.</remarks>
    public static (byte R, byte G, byte B) MutedRgb => IsLightBackground ? ((byte)0x5b, (byte)0x63, (byte)0x6b) : ((byte)0x9a, (byte)0xa4, (byte)0xaf);

    private static bool DetectLightBackground()
    {
        string? forced = EngineKnobs.Theme.Value;
        if (string.Equals(forced, "light", StringComparison.OrdinalIgnoreCase))
            return true;
        if (string.Equals(forced, "dark", StringComparison.OrdinalIgnoreCase))
            return false;

        // COLORFGBG is "fg;bg" (some terminals add a middle field); the last field is the background ANSI index.
        // 0-6 and 8 are dark backgrounds; 7 and 9-15 are light. Absent on most terminals → assume dark.
        string? fgbg = Environment.GetEnvironmentVariable("COLORFGBG");
        if (!string.IsNullOrEmpty(fgbg))
        {
            string[] parts = fgbg.Split(';');
            if (parts.Length > 0 && int.TryParse(parts[^1], out int bg))
                return bg == 7 || bg >= 9;
        }
        return false;
    }

    /// <summary>Renders the app header: the Hartsy H mark on a real terminal, or a framed block-glyph fallback otherwise.</summary>
    /// <remarks>On a real terminal this is the embedded logo drawn with truecolor half-blocks; piped output or <c>NO_COLOR</c>
    /// falls back to the block-glyph H. Both carry the wordmark and version / backend / working directory.</remarks>
    public static void RenderBanner(string backendSelector)
    {
        string version = typeof(CliTheme).Assembly.GetName().Version?.ToString(3) ?? "dev";
        string backend = BackendFactory.Describe(backendSelector);
        string cwd = Directory.GetCurrentDirectory();

        if (TerminalImage.IsSupported && TryLoadLogo() is { } logo
            && TerminalImage.RenderToLines(logo.Rgb, logo.Width, logo.Height, maxCellWidth: 26, indent: 3,
                transparentKey: LogoKey, sharpen: 0.4f) is { Count: > 0 } lines)
        {
            (byte mr, byte mg, byte mb) = Hex(Muted);
            string accent = $"\x1b[38;2;{AccentRgb.R};{AccentRgb.G};{AccentRgb.B}m";
            string muted = $"\x1b[38;2;{mr};{mg};{mb}m";
            string wordmark = $"\x1b[1m{accent}HARTSY\x1b[39m INFERENCE\x1b[0m";
            string tagline = $"{muted}pure-C# AI inference engine\x1b[0m";
            int band = Math.Max(0, (lines.Count - 2) / 2);

            AnsiConsole.WriteLine();
            for (int i = 0; i < lines.Count; i++)
            {
                string line = lines[i];
                if (i == band)
                    line += "    " + wordmark;
                else if (i == band + 1)
                    line += "    " + tagline;
                Console.Out.Write(line);
                Console.Out.Write('\n');
            }
            Console.Out.Write($"   {muted}v{version}  ·  backend {accent}{backend}{muted}  ·  {cwd}\x1b[0m\n\n");
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
        body.Append($"[bold {Accent}]HARTSY[/] [bold]INFERENCE[/]  [{Muted}]· pure-C# AI inference[/]\n");
        body.Append($"[{Muted}]v{version}  ·  backend[/] [{Accent}]{Markup.Escape(backend)}[/]  [{Muted}]·  {Markup.Escape(cwd)}[/]");

        Panel panel = new Panel(new Markup(body.ToString()))
            .Border(BoxBorder.Rounded)
            .BorderColor(AccentColor)
            .Padding(3, 1, 3, 1);
        AnsiConsole.Write(panel);
    }

    private static (byte R, byte G, byte B) Hex(string hex)
    {
        ReadOnlySpan<char> h = hex.AsSpan().TrimStart('#');
        return (Convert.ToByte(new string(h[..2]), 16), Convert.ToByte(new string(h[2..4]), 16), Convert.ToByte(new string(h[4..6]), 16));
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
            Logs.Warning($"CLI banner logo failed to load: {ex.Message}");
            _logo = null;
        }
        return _logo;
    }

    /// <summary>Spectre markup fragment for a status badge (emoji + colored label).</summary>
    public static string StatusMarkup(ModelStatus status) => status switch
    {
        ModelStatus.Verified => "[green]✅ verified[/]",
        ModelStatus.ValidationPending => "[yellow]🧪 validating[/]",
        ModelStatus.Structural => "[#9aa4af]🏗️ structural[/]",
        _ => "[#9aa4af]?[/]",
    };

    /// <summary>Formats a byte count as a human-readable size (e.g. "4.2 GB"), or <paramref name="emptyMarkup"/> when ≤ 0.</summary>
    internal static string FormatBytes(long bytes, string emptyMarkup = "")
    {
        if (bytes <= 0)
            return emptyMarkup;
        string[] units = { "B", "KB", "MB", "GB", "TB" };
        double value = bytes;
        int unit = 0;
        while (value >= 1024 && unit < units.Length - 1)
        {
            value /= 1024;
            unit++;
        }
        return $"{value:0.#} {units[unit]}";
    }
}
