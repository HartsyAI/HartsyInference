using System.Reflection;
using HartsyInference.Cli.Commands;
using Spectre.Console.Cli;
using Xunit;

namespace HartsyInference.Cli.Tests;

public sealed class VideoCommandContractTests
{
    [Fact]
    public void EngineProfileAndModelProfileRemainDistinctOptions()
    {
        PropertyInfo engineProfile = typeof(VideoCommand.Settings).GetProperty(nameof(VideoCommand.Settings.Profile))!;
        PropertyInfo modelProfile = typeof(VideoCommand.Settings).GetProperty(nameof(VideoCommand.Settings.ModelProfile))!;

        CommandOptionAttribute engineOption = engineProfile.GetCustomAttribute<CommandOptionAttribute>()!;
        CommandOptionAttribute modelOption = modelProfile.GetCustomAttribute<CommandOptionAttribute>()!;
        Assert.Contains("profile", engineOption.LongNames);
        Assert.Contains("model-profile", modelOption.LongNames);
        Assert.DoesNotContain("model-profile", engineOption.LongNames);
    }

    [Theory]
    [InlineData("h3-pdd")]
    [InlineData("h3-controlnet")]
    public void H3ConverterDispatchPublishesGeneratedHelp(string converter)
    {
        int exitCode = Program.Main(["convert", converter, "--help"]);

        Assert.Equal(0, exitCode);
    }
}
