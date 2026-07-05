using Spectre.Console;

namespace HartsyInference.Cli.Infra;

/// <summary>Shared Spectre styling: the startup banner and consistent status/modality markup used by every command.</summary>
public static class CliTheme
{
    /// <summary>Spectre markup color for the brand accent.</summary>
    public const string Accent = "mediumpurple2";

    /// <summary>Renders the figlet banner with the version and the resolved backend for the interactive/no-arg entry.</summary>
    public static void RenderBanner(string backendSelector)
    {
        AnsiConsole.Write(new FigletText("hartsy").Color(Color.MediumPurple2));
        string version = typeof(CliTheme).Assembly.GetName().Version?.ToString(3) ?? "dev";
        AnsiConsole.MarkupLine($"[grey]HartsyInference[/] [{Accent}]v{version}[/]  ·  pure-C# AI inference");
        AnsiConsole.MarkupLine($"[grey]backend:[/] [{Accent}]{Markup.Escape(BackendFactory.Describe(backendSelector))}[/]");
        AnsiConsole.WriteLine();
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
