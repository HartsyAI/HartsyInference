using System.Text;
using HartsyInference.Cli.Infra;

namespace HartsyInference.Cli.Repl;

/// <summary>An interactive prompt reader with live <c>/</c>-command autocomplete: as the user types a slash command,
/// a filtered dropdown of matching commands (with descriptions) is drawn below the input and navigated with the arrow
/// keys, Tab completes, Enter runs. Falls back to a plain read when input/output is not an interactive terminal.</summary>
public static class LineEditor
{
    private const int MaxSuggestions = 8;

    /// <summary>Reads one line for the given <paramref name="mode"/> prompt, offering <paramref name="commands"/> as
    /// autocomplete. Returns null on EOF or Ctrl+C (caller should exit).</summary>
    public static string? ReadLine(string mode, IReadOnlyList<SlashCommand> commands)
    {
        if (Console.IsInputRedirected || Console.IsOutputRedirected)
        {
            Console.Write($"hartsy ({mode}) > ");
            return Console.ReadLine();
        }

        bool previousCtrlC = Console.TreatControlCAsInput;
        Console.TreatControlCAsInput = true;
        try
        {
            return Interactive(mode, commands);
        }
        catch (IOException)
        {
            return Console.ReadLine();
        }
        finally
        {
            Console.TreatControlCAsInput = previousCtrlC;
        }
    }

    private static string? Interactive(string mode, IReadOnlyList<SlashCommand> commands)
    {
        StringBuilder buffer = new StringBuilder();
        int caret = 0;
        int selected = 0;

        Console.Write(BuildPrompt(mode));
        // Visible prompt width ("hartsy (" + mode + ") › "), computed rather than queried from the terminal so the
        // editor issues no cursor-position DSR queries (which hang/throw on some terminals).
        int promptColumn = 12 + mode.Length;

        while (true)
        {
            List<SlashCommand> matches = Matches(buffer.ToString(), commands);
            if (selected >= matches.Count)
                selected = Math.Max(0, matches.Count - 1);
            Render(promptColumn, buffer.ToString(), caret, matches, selected);

            ConsoleKeyInfo key = Console.ReadKey(intercept: true);

            if (key.Key == ConsoleKey.Enter)
            {
                if (matches.Count > 0 && ShowsDropdown(buffer.ToString()))
                    Complete(buffer, ref caret, matches[selected]);
                Finish(promptColumn, buffer.ToString());
                return buffer.ToString();
            }
            if (key.Key == ConsoleKey.C && (key.Modifiers & ConsoleModifiers.Control) != 0)
            {
                Finish(promptColumn, buffer.ToString());
                return null;
            }
            if (key.Key == ConsoleKey.Tab)
            {
                if (matches.Count > 0 && ShowsDropdown(buffer.ToString()))
                {
                    Complete(buffer, ref caret, matches[selected]);
                    selected = 0;
                }
                continue;
            }

            switch (key.Key)
            {
                case ConsoleKey.UpArrow:
                    if (matches.Count > 0)
                        selected = (selected - 1 + matches.Count) % matches.Count;
                    break;
                case ConsoleKey.DownArrow:
                    if (matches.Count > 0)
                        selected = (selected + 1) % matches.Count;
                    break;
                case ConsoleKey.LeftArrow:
                    if (caret > 0)
                        caret--;
                    break;
                case ConsoleKey.RightArrow:
                    if (caret < buffer.Length)
                        caret++;
                    break;
                case ConsoleKey.Home:
                    caret = 0;
                    break;
                case ConsoleKey.End:
                    caret = buffer.Length;
                    break;
                case ConsoleKey.Backspace:
                    if (caret > 0)
                    {
                        buffer.Remove(caret - 1, 1);
                        caret--;
                        selected = 0;
                    }
                    break;
                case ConsoleKey.Delete:
                    if (caret < buffer.Length)
                    {
                        buffer.Remove(caret, 1);
                        selected = 0;
                    }
                    break;
                default:
                    if (!char.IsControl(key.KeyChar))
                    {
                        buffer.Insert(caret, key.KeyChar);
                        caret++;
                        selected = 0;
                    }
                    break;
            }
        }
    }

    private static void Render(int promptColumn, string buffer, int caret, List<SlashCommand> matches, int selected)
    {
        // Return to the start of the buffer, erase it and anything below, then repaint.
        Console.Write($"\x1b[{promptColumn + 1}G\x1b[0J");
        Console.Write(buffer);

        int rows = 0;
        foreach (SlashCommand cmd in matches)
        {
            Console.Write("\n\r");
            WriteSuggestion(cmd, rows == selected);
            rows++;
        }
        if (rows > 0)
            Console.Write($"\x1b[{rows}A");
        Console.Write($"\x1b[{promptColumn + caret + 1}G");
    }

    private static void Finish(int promptColumn, string buffer)
    {
        Console.Write($"\x1b[{promptColumn + 1}G\x1b[0J");
        Console.Write(buffer);
        Console.WriteLine();
    }

    private static void WriteSuggestion(SlashCommand cmd, bool selected)
    {
        (byte r, byte g, byte b) = CliTheme.AccentRgb;
        string blue = $"\x1b[38;2;{r};{g};{b}m";
        string name = "/" + cmd.Name;
        if (selected)
            Console.Write($"\x1b[48;2;24;52;70m{blue} ▸ {name,-12}\x1b[0m\x1b[48;2;24;52;70m \x1b[38;2;170;190;205m{cmd.Description}\x1b[0m");
        else
            Console.Write($"{blue}   {name,-12}\x1b[0m \x1b[38;2;120;120;120m{cmd.Description}\x1b[0m");
    }

    private static string BuildPrompt(string mode)
    {
        (byte r, byte g, byte b) = CliTheme.AccentRgb;
        string blue = $"\x1b[38;2;{r};{g};{b}m";
        const string grey = "\x1b[90m";
        const string reset = "\x1b[0m";
        return $"{blue}hartsy{reset} {grey}({mode}){reset} {blue}›{reset} ";
    }

    private static bool ShowsDropdown(string text) => text.StartsWith('/') && !text.Contains(' ');

    private static List<SlashCommand> Matches(string text, IReadOnlyList<SlashCommand> commands)
    {
        if (!ShowsDropdown(text))
            return new List<SlashCommand>();
        string query = text.TrimStart('/');
        return commands
            .Where(c => c.Name.StartsWith(query, StringComparison.OrdinalIgnoreCase))
            .Take(MaxSuggestions)
            .ToList();
    }

    private static void Complete(StringBuilder buffer, ref int caret, SlashCommand cmd)
    {
        buffer.Clear();
        buffer.Append('/').Append(cmd.Name).Append(' ');
        caret = buffer.Length;
    }
}
