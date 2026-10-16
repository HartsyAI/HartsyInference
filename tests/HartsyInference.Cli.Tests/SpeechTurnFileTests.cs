using HartsyInference.Cli.Dispatch;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.Cli.Tests;

/// <summary>The <c>speak</c> turn files: references name their clips relative to the file, and a saved take reads back as the
/// history that revises it.</summary>
public sealed class SpeechTurnFileTests
{
    [Fact]
    public void References_ReadTheirClipsRelativeToTheFile()
    {
        using TestDirectory files = new();
        files.WriteBytes("alice.wav", [1, 2, 3]);
        string path = files.WriteText("refs.json", """[{"text": "<|speaker:0|> Hello there.", "audio": "alice.wav"}]""");

        SpeechTurn reference = Assert.Single(SpeechTurnFile.Load(path)!);

        Assert.Equal("<|speaker:0|> Hello there.", reference.Text);
        Assert.Equal([1, 2, 3], reference.Audio!.Data);
        Assert.Equal("wav", reference.Audio.Format);
        Assert.Null(reference.Codes);
    }

    [Fact]
    public void SavedTurns_ReadBackAsTheSameHistory()
    {
        using TestDirectory files = new();
        string path = files.PathOf("take.json");
        SpeechTurn[] take =
        [
            new() { Text = "[whispering] Come closer.", Codes = [[5, 6, 7], [1, 2, 3]] },
            new() { Text = "Closer still.", Codes = [[8], [4]] },
        ];

        SpeechTurnFile.Save(path, take);
        IReadOnlyList<SpeechTurn> history = SpeechTurnFile.Load(path)!;

        Assert.Equal(take.Select(static t => t.Text), history.Select(static t => t.Text));
        Assert.Equal(take.Select(static t => t.Codes), history.Select(static t => t.Codes));
        Assert.All(history, static t => Assert.Null(t.Audio));
        Assert.DoesNotContain("\"audio\"", File.ReadAllText(path), StringComparison.Ordinal);
    }

    [Fact]
    public void NoPath_IsNoTurns() => Assert.Null(SpeechTurnFile.Load(" "));

    [Fact]
    public void ATurnWithoutSound_IsRefused()
    {
        using TestDirectory files = new();
        string path = files.WriteText("bad.json", """[{"text": "Silent."}]""");
        Assert.Throws<ArgumentException>(() => SpeechTurnFile.Load(path));
    }
}
