using HartsyInference.ModelAssets.MiniMaxH3;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

public sealed class MiniMaxH3ConverterPathTests : IDisposable
{
    private readonly string _directory = Path.Combine(
        Path.GetTempPath(), "h3-converter-paths-" + Guid.NewGuid().ToString("N"));

    public MiniMaxH3ConverterPathTests()
    {
        Directory.CreateDirectory(_directory);
    }

    [Fact]
    public void PddConversionRejectsOutputSymlinkToInput()
    {
        if (OperatingSystem.IsWindows())
        {
            // Windows runners do not consistently grant the symbolic-link privilege required by this test.
            return;
        }

        (string first, string second, string third, string output) = CreateInputsAndOutputAlias();

        ArgumentException error = Assert.Throws<ArgumentException>(() =>
            MiniMaxH3PddPrunedConverter.Convert(first, second, third, output,
                MiniMaxH3PddTask.Fl2Va));

        Assert.Contains("must not overwrite", error.Message, StringComparison.Ordinal);
        Assert.Equal("first", File.ReadAllText(first));
    }

    [Fact]
    public void ControlNetConversionRejectsOutputSymlinkToInput()
    {
        if (OperatingSystem.IsWindows())
        {
            // Windows runners do not consistently grant the symbolic-link privilege required by this test.
            return;
        }

        (string first, string second, string third, string output) = CreateInputsAndOutputAlias();

        ArgumentException error = Assert.Throws<ArgumentException>(() =>
            MiniMaxH3ControlNetPrunedConverter.Convert(first, second, third, output));

        Assert.Contains("must not overwrite", error.Message, StringComparison.Ordinal);
        Assert.Equal("first", File.ReadAllText(first));
    }

    public void Dispose()
    {
        try
        {
            Directory.Delete(_directory, recursive: true);
        }
        catch (IOException)
        {
            // Best-effort cleanup must not hide the assertion result.
        }
        catch (UnauthorizedAccessException)
        {
            // Best-effort cleanup must not hide the assertion result.
        }
    }

    private (string First, string Second, string Third, string Output) CreateInputsAndOutputAlias()
    {
        string suffix = Guid.NewGuid().ToString("N");
        string first = Path.Combine(_directory, "first-" + suffix + ".safetensors");
        string second = Path.Combine(_directory, "second-" + suffix + ".safetensors");
        string third = Path.Combine(_directory, "third-" + suffix + ".safetensors");
        string output = Path.Combine(_directory, "output-" + suffix + ".safetensors");
        File.WriteAllText(first, "first");
        File.WriteAllText(second, "second");
        File.WriteAllText(third, "third");
        File.CreateSymbolicLink(output, first);
        return (first, second, third, output);
    }
}
