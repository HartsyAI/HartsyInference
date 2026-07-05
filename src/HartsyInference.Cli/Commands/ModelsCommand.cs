using HartsyInference.Cli.Infra;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>Shows the models that have been downloaded into the local cache.</summary>
public sealed class ModelsCommand : Command<ModelsCommand.Settings>
{
    /// <summary>Options for <c>hartsy models</c> (none yet).</summary>
    public sealed class Settings : CommandSettings
    {
    }

    /// <inheritdoc/>
    public override int Execute(CommandContext context, Settings settings)
    {
        CacheView.Render();
        return 0;
    }
}
