using HartsyInference.Core.Configuration;
using Spectre.Console;

namespace HartsyInference.Cli.Infra;

/// <summary>Turns <c>--profile</c> / <c>--set</c> into a <see cref="KnobProfile"/>, and prints <c>--list-settings</c>.</summary>
/// <remarks>Settings are applied as a scoped profile rather than by exporting environment variables, so one run's
/// overrides cannot leak into another process or outlive the command.</remarks>
public static class KnobCli
{
    /// <summary>Builds the profile for this run, or null when neither flag was passed. Throws with the operator's typo named.</summary>
    public static KnobProfile? Build(string? profileName, string[]? settings)
    {
        KnobProfile? profile = null;
        if (!string.IsNullOrWhiteSpace(profileName))
        {
            profile = KnobProfiles.ByName(profileName)
                ?? throw new ArgumentException(
                    $"Unknown profile '{profileName}'. Known profiles: {string.Join(", ", KnobProfiles.Names)}.");
        }
        if (settings is not { Length: > 0 })
        {
            return profile;
        }
        profile ??= KnobProfile.Create("cli");
        foreach (string entry in settings)
        {
            int eq = entry.IndexOf('=', StringComparison.Ordinal);
            if (eq <= 0)
            {
                throw new ArgumentException($"--set expects id=value, got '{entry}'.");
            }
            string id = entry[..eq].Trim();
            string value = entry[(eq + 1)..];
            if (!profile.TrySet(id, value, out KnobProfile updated, out string? error))
            {
                throw new ArgumentException($"{error} Run --list-settings to see valid ids.");
            }
            profile = updated;
        }
        return profile;
    }

    /// <summary>Prints every declared setting grouped by domain.</summary>
    public static void ListSettings()
    {
        List<(string Id, string? Legacy, string Type, object? Default, KnobScope Scope, KnobDomain Domain, string Summary)> all =
            [.. KnobRegistry.All.Select(KnobRegistry.Describe)
                .Where(k => !k.Id.StartsWith("test.", StringComparison.Ordinal))
                .OrderBy(k => k.Domain).ThenBy(k => k.Id, StringComparer.Ordinal)];

        foreach (IGrouping<KnobDomain, (string Id, string? Legacy, string Type, object? Default, KnobScope Scope, KnobDomain Domain, string Summary)> group
            in all.GroupBy(k => k.Domain))
        {
            Table table = new Table().Border(TableBorder.Rounded).Title($"[bold]{group.Key}[/]");
            table.AddColumn("setting");
            table.AddColumn("type");
            table.AddColumn("default");
            table.AddColumn("scope");
            table.AddColumn("what it does");
            foreach ((string id, _, string type, object? def, KnobScope scope, _, string summary) in group)
            {
                table.AddRow(
                    Markup.Escape(id),
                    Markup.Escape(type),
                    Markup.Escape(def?.ToString() ?? "—"),
                    scope == KnobScope.Construction ? "[yellow]load[/]" : "run",
                    Markup.Escape(summary));
            }
            AnsiConsole.Write(table);
        }
        AnsiConsole.MarkupLine($"[grey]{all.Count} settings. Override with --set id=value, or --profile "
            + $"{string.Join('|', KnobProfiles.Names)}. 'load' scope binds when a model is built.[/]");
    }
}
