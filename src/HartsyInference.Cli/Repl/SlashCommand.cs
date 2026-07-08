namespace HartsyInference.Cli.Repl;

/// <summary>A REPL slash command surfaced in the <c>/</c> autocomplete dropdown.</summary>
public sealed record SlashCommand(string Name, string Description);
