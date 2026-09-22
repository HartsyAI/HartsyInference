using Xunit;

namespace HartsyInference.Cli.Tests;

/// <summary>The <c>--profile</c> / <c>--set</c> pairs must not reach the command parser.
///
/// <para>Both are read by <c>Program.Main</c> before any command exists, so no command declares them. With
/// <c>StrictParsing</c> on — see <see cref="StrictOptionParsingTests"/> for why it is — anything the parser does not
/// recognize fails the run. Leaving them in therefore broke <em>every</em> invocation that used one with
/// <c>Unexpected option 'set'</c>, which is how the feature shipped and stayed broken: nothing exercised it.</para></summary>
public sealed class KnobArgStrippingTests
{
    [Fact]
    public void StripsEachFlagAndItsValue_LeavingTheCommandLineIntact()
    {
        string[] args = ["--set", "numerics.fp8Native=false", "--profile", "reference", "--set",
            "diagnostics.logLevel=Info", "video", "-m", "minimax-h3", "--steps", "4"];

        string[] kept = Program.WithoutKnobArgs(args);

        // The value has to go with the flag: leaving `numerics.fp8Native=false` behind as a bare token would be
        // read as the command name, which is a worse failure than the one this fixes.
        Assert.Equal(["video", "-m", "minimax-h3", "--steps", "4"], kept);
    }

    [Fact]
    public void LeavesEverythingAloneWhenNeitherFlagIsPresent()
    {
        string[] args = ["image", "a castle", "--model-path", "sdxl.safetensors", "--steps", "30"];

        Assert.Equal(args, Program.WithoutKnobArgs(args));
    }

    [Theory]
    [InlineData("--set")]
    [InlineData("--profile")]
    public void KeepsATrailingFlagThatHasNoValue_SoTheParserStillRejectsIt(string flag)
    {
        // ArgValues cannot have read a flag with nothing after it, so the setting the operator asked for was never
        // applied. Dropping it here would run the generation anyway and report success — the exact silent-wrong-config
        // failure StrictParsing exists to prevent. It stays, and the parser names it.
        string[] kept = Program.WithoutKnobArgs(["video", "-m", "minimax-h3", flag]);

        Assert.Equal(["video", "-m", "minimax-h3", flag], kept);
    }

    [Fact]
    public void DoesNotConsumeATokenThatMerelyContainsTheFlagName()
    {
        // Matching is exact, not substring: an option like --settings-file must keep its own value.
        string[] args = ["video", "--settings-file", "/etc/hartsy.json", "--set", "vram.stepCache=true"];

        Assert.Equal(["video", "--settings-file", "/etc/hartsy.json"], Program.WithoutKnobArgs(args));
    }
}
