using Spectre.Console.Cli;
using Xunit;

namespace HartsyInference.Cli.Tests;

/// <summary>An option no command declares must fail the run.
///
/// <para>Spectre's default is to collect an unrecognized option as a "remaining" argument, which nothing here reads.
/// A typo therefore did not fail — it ran the generation with a DIFFERENT setting than the one asked for and
/// reported success. Two real cases: <c>--cfgscale 2</c> (the option is <c>--cfg</c>) silently generated at the
/// model's default guidance, and <c>--detect</c> (it is <c>--mode detect</c>) silently fell through to CLIP and
/// died several layers down in a model loader complaining about a missing text-encoder weight.</para>
///
/// <para>This is built against the same <see cref="CommandApp"/> configuration <c>Program</c> applies, since the
/// setting is what is under test rather than any one command's option list.</para></summary>
public sealed class StrictOptionParsingTests
{
    private sealed class NoopSettings : CommandSettings
    {
        [CommandOption("--real-option")]
        public string? Real { get; init; }
    }

    private sealed class NoopCommand : Command<NoopSettings>
    {
        public override int Execute(CommandContext context, NoopSettings settings) => 0;
    }

    private static CommandApp BuildApp(bool strict)
    {
        CommandApp app = new();
        app.Configure(config =>
        {
            config.PropagateExceptions();
            config.Settings.StrictParsing = strict;
            config.AddCommand<NoopCommand>("noop");
        });
        return app;
    }

    [Theory]
    [InlineData("--totally-bogus-flag")]
    [InlineData("--cfgscale")]
    public void An_Undeclared_Option_Is_Refused(string option)
    {
        CommandApp app = BuildApp(strict: true);
        Assert.ThrowsAny<Exception>(() => app.Run(["noop", option, "2"]));
    }

    /// <summary>The setting is load-bearing: without it the same run succeeds, which is the behaviour that let a
    /// typo through.</summary>
    [Fact]
    public void Without_Strict_Parsing_The_Same_Run_Succeeds()
    {
        CommandApp app = BuildApp(strict: false);
        Assert.Equal(0, app.Run(["noop", "--totally-bogus-flag", "2"]));
    }

    [Fact]
    public void A_Declared_Option_Still_Parses()
    {
        CommandApp app = BuildApp(strict: true);
        Assert.Equal(0, app.Run(["noop", "--real-option", "value"]));
    }
}
