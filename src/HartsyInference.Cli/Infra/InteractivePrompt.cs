using Spectre.Console;

namespace HartsyInference.Cli.Infra;

/// <summary>Small interactive-prompt helpers for the REPL: an arrow-key selection list (with type-to-filter) plus text/confirm prompts.</summary>
/// <remarks>Each falls back to a plain numbered / line-read flow when input is redirected (piped, tests), so the same call works
/// interactively and non-interactively.</remarks>
public static class InteractivePrompt
{
    // Fall back to a plain numbered/line-read flow whenever the console can't drive Spectre's interactive prompts —
    // redirected streams, or a terminal Spectre judges non-interactive / non-ANSI (CI, dumb terminals, some PTYs).
    private static bool Plain =>
        Console.IsInputRedirected || Console.IsOutputRedirected || !AnsiConsole.Profile.Capabilities.Ansi
        || !AnsiConsole.Profile.Capabilities.Interactive;

    /// <summary>Shows <paramref name="choices"/> and returns the selected index, or -1 if the user cancels.</summary>
    public static int SelectIndex(string title, IReadOnlyList<string> choices)
    {
        if (choices.Count == 0)
            return -1;

        if (!Plain)
        {
            int chosen = AnsiConsole.Prompt(
                new SelectionPrompt<int>()
                    .Title($"[{CliTheme.Accent}]{Markup.Escape(title)}[/] [{CliTheme.Muted}](↑/↓, type to filter, Enter)[/]")
                    .PageSize(15)
                    .MoreChoicesText($"[{CliTheme.Muted}](scroll for more)[/]")
                    .EnableSearch()
                    .UseConverter(i => Markup.Escape(choices[i]))
                    .AddChoices(Enumerable.Range(0, choices.Count)));
            return chosen;
        }

        AnsiConsole.MarkupLine($"[{CliTheme.Accent}]{Markup.Escape(title)}[/]");
        for (int i = 0; i < choices.Count; i++)
            AnsiConsole.MarkupLine($"  [{CliTheme.Accent}]{i + 1}[/]  {Markup.Escape(choices[i])}");
        AnsiConsole.Markup($"[{CliTheme.Muted}]number (Enter to cancel): [/]");
        string? line = Console.ReadLine();
        if (int.TryParse(line?.Trim(), out int n) && n >= 1 && n <= choices.Count)
            return n - 1;
        return -1;
    }

    /// <summary>Prompts for a line of text, returning <paramref name="def"/> on empty Enter, or null when there is no default.</summary>
    public static string? Ask(string label, string? def = null)
    {
        if (!Plain)
        {
            TextPrompt<string> prompt = new TextPrompt<string>($"[{CliTheme.Accent}]{Markup.Escape(label)}[/]").AllowEmpty();
            if (def is not null)
                prompt.DefaultValue(def).ShowDefaultValue();
            string result = AnsiConsole.Prompt(prompt);
            return string.IsNullOrEmpty(result) ? def : result;
        }

        AnsiConsole.Markup(def is null ? $"[{CliTheme.Accent}]{Markup.Escape(label)}[/][{CliTheme.Muted}]: [/]"
            : $"[{CliTheme.Accent}]{Markup.Escape(label)}[/] [{CliTheme.Muted}][[{Markup.Escape(def)}]]: [/]");
        string? line = Console.ReadLine();
        return string.IsNullOrWhiteSpace(line) ? def : line.Trim();
    }

    /// <summary>Yes/no confirmation, defaulting to <paramref name="defaultYes"/> on a bare Enter.</summary>
    public static bool Confirm(string question, bool defaultYes)
    {
        if (!Plain)
            return AnsiConsole.Confirm($"[{CliTheme.Accent}]{Markup.Escape(question)}[/]", defaultYes);

        AnsiConsole.Markup($"[{CliTheme.Accent}]{Markup.Escape(question)}[/] [{CliTheme.Muted}]{(defaultYes ? "[[Y/n]]" : "[[y/N]]")}: [/]");
        string? line = Console.ReadLine()?.Trim();
        if (string.IsNullOrEmpty(line))
            return defaultYes;
        return line.StartsWith('y') || line.StartsWith('Y');
    }
}
