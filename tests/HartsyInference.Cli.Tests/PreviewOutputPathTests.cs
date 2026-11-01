using HartsyInference.Cli.Commands;
using Xunit;

namespace HartsyInference.Cli.Tests;

/// <summary>--preview-output takes one PNG path. A directory is refused before the model loads, instead of failing at the first preview write.</summary>
public sealed class PreviewOutputPathTests
{
    [Fact]
    public void Unset_Preview_Path_Is_Accepted()
    {
        Assert.True(CommandRunner.RequireFilePath(null, "--preview-output", out int exitCode));
        Assert.Equal(0, exitCode);
    }

    [Fact]
    public void File_Path_Is_Accepted_Before_The_File_Exists()
    {
        using TestDirectory dir = new();

        Assert.True(CommandRunner.RequireFilePath(dir.PathOf("latest.png"), "--preview-output", out int exitCode));
        Assert.Equal(0, exitCode);
    }

    [Fact]
    public void Existing_Directory_Is_Refused()
    {
        using TestDirectory dir = new();

        Assert.False(CommandRunner.RequireFilePath(dir.PathOf(string.Empty), "--preview-output", out int exitCode));
        Assert.Equal(1, exitCode);
    }

    [Fact]
    public void Trailing_Separator_Is_Refused_Even_When_The_Directory_Does_Not_Exist()
    {
        string path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"hartsy-missing-{Guid.NewGuid():N}") + System.IO.Path.DirectorySeparatorChar;

        Assert.False(CommandRunner.RequireFilePath(path, "--preview-output", out int exitCode));
        Assert.Equal(1, exitCode);
    }
}
