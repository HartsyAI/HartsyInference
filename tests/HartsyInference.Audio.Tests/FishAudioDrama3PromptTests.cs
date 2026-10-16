using HartsyInference.Audio.Models.FishAudio;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Drama 3 prompts are written literal by literal, as fish-speech's <c>build_drama3_prompt</c> writes them (the cases of
/// <c>tests/test_fish_omni_generate.py</c>): a recording tokenizer turns each literal into two tokens and remembers it.</summary>
public sealed class FishAudioDrama3PromptTests
{
    private readonly List<string> _parts = [];

    [Fact]
    public void PlainText_WritesTheTtsInstructionAndAnOpenVoiceTurn()
    {
        int[,] rows = Prompt().Build("hello", [], []);
        Assert.Equal(
            ["<|im_start|>system\n", "convert the provided text to speech", "<|im_end|>\n", "<|im_start|>user\n",
             "<|speaker:0|> hello", "<|im_end|>\n", "<|im_start|>assistant\n<|voice|>"], _parts);
        Assert.Equal(14, rows.GetLength(0));
        Assert.Equal(5, rows.GetLength(1));
        for (int r = 0; r < rows.GetLength(0); r++)
        {
            for (int c = 1; c < 5; c++) Assert.Equal(0, rows[r, c]);
        }
    }

    [Fact]
    public void References_CarryTheirCodes_AndMoveTheTagInFrontOfLeadingSpace()
    {
        int[,] codes = { { 1, 2 }, { 3, 4 }, { 5, 6 }, { 7, 0 } };
        int[,] rows = Prompt().Build("<|speaker:0|>hello", [new FishAudioDrama3Turn("  <|speaker:0|> reference", codes)], []);
        Assert.Contains("convert the provided text to speech reference to the following:", _parts);
        Assert.Contains("\n\n[Reference 1]\nText: <|speaker:0|>   reference\nSpeech: ", _parts);
        Assert.Contains("<|speaker:0|>hello", _parts);
        List<int[]> audio = AudioRows(rows);
        Assert.Equal(2, audio.Count);
        Assert.Equal([41, 1, 3, 5, 7], audio[0]);
        Assert.Equal([42, 2, 4, 6, 0], audio[1]);
    }

    [Fact]
    public void MultipleReferences_AreNumberedInOrder()
    {
        int[,] codes = { { 1 }, { 2 }, { 3 }, { 4 } };
        Prompt().Build("hi", [new FishAudioDrama3Turn("<|speaker:0|> first", codes), new FishAudioDrama3Turn("<|speaker:1|> second", codes)], []);
        Assert.Contains("\n\n[Reference 1]\nText: <|speaker:0|> first\nSpeech: ", _parts);
        Assert.Contains("\n\n[Reference 2]\nText: <|speaker:1|> second\nSpeech: ", _parts);
    }

    [Fact]
    public void History_IsWrittenAsCompletedTurns()
    {
        int[,] codes = { { 1, 2 }, { 3, 4 }, { 5, 6 }, { 7, 0 } };
        int plain = Prompt().Build("now", [], []).GetLength(0);
        _parts.Clear();
        int[,] rows = Prompt().Build("now", [], [new FishAudioDrama3Turn("before", codes)]);
        Assert.Equal(
            ["<|im_start|>user\n", "<|speaker:0|> before", "<|im_end|>\n", "<|im_start|>assistant\n<|voice|>", "<|im_end|>\n",
             "<|im_start|>user\n", "<|speaker:0|> now", "<|im_end|>\n", "<|im_start|>assistant\n<|voice|>"], _parts[^9..]);
        // Five two-token literals plus the two audio frames.
        Assert.Equal(12, rows.GetLength(0) - plain);
        Assert.Equal([[41, 1, 3, 5, 7], [42, 2, 4, 6, 0]], AudioRows(rows));
    }

    [Fact]
    public void Speakers0To15_AreKeptEverywhere_And16IsRefused()
    {
        int[,] codes = { { 1 }, { 2 }, { 3 }, { 4 } };
        string text = "<|speaker:15|> hi\n<|speaker:0|> yes";
        Prompt().Build(text, [new FishAudioDrama3Turn("<|speaker:2|>ref", codes)], [new FishAudioDrama3Turn("<|speaker:3|> before", codes)]);
        Assert.Contains(text, _parts);
        Assert.Contains("<|speaker:3|> before", _parts);
        Assert.Contains("\n\n[Reference 1]\nText: <|speaker:2|>ref\nSpeech: ", _parts);
        ArgumentException refused = Assert.Throws<ArgumentException>(
            () => Prompt().Build("hi", [], [new FishAudioDrama3Turn("<|speaker:16|> no", codes)]));
        Assert.Contains("history 1", refused.Message);
    }

    [Theory]
    [InlineData(" ")]
    [InlineData("hello <|voice|>")]
    [InlineData("<|speaker:16|> hello")]
    [InlineData("<|speaker:00|> hi")]
    [InlineData("<|speaker:01|> hi")]
    [InlineData("<|speaker:x|> hi")]
    public void InvalidText_IsRefused(string text) => Assert.Throws<ArgumentException>(() => Prompt().Build(text, [], []));

    [Fact]
    public void InvalidCodes_AreRefused()
    {
        Assert.Throws<ArgumentException>(() => Prompt().Build("hi", [new FishAudioDrama3Turn("ref", new int[2, 3])], []));
        Assert.Throws<ArgumentException>(() => Prompt().Build("hi", [new FishAudioDrama3Turn("ref", new int[4, 0])], []));
        int[,] outOfRange = new int[4, 2];
        outOfRange[1, 1] = 8;
        Assert.Throws<ArgumentException>(() => Prompt().Build("hi", [new FishAudioDrama3Turn("ref", outOfRange)], []));
    }

    [Fact]
    public void TooLongPrompts_DropTheChunkHistoryFirst()
    {
        int[,] codes = new int[4, FishAudioDrama3Config.MaxPromptRows - 10];
        FishAudioDrama3History history = new();
        history.Push(new FishAudioDrama3Turn("first", codes));
        int[,] rows = Prompt().BuildWithHistory("now", [], [], history);
        Assert.True(rows.GetLength(0) < 40);
        Assert.Throws<FishAudioDrama3PromptTooLongException>(() => Prompt().Build("now", [], [new FishAudioDrama3Turn("first", codes)]));
    }

    [Fact]
    public void AModelWithLessRoom_DropsTheChunkHistoryAtItsOwnLimit()
    {
        // The history fits upstream's 8192 rows but not a 600-row limit, so that prompt is built without it.
        int[,] codes = new int[4, 1000];
        FishAudioDrama3History history = new();
        history.Push(new FishAudioDrama3Turn("first", codes));
        Assert.True(Prompt().BuildWithHistory("now", [], [], history).GetLength(0) > 1000);
        Assert.True(Prompt().BuildWithHistory("now", [], [], history, maxRows: 600).GetLength(0) < 40);
        FishAudioDrama3PromptTooLongException tooLong = Assert.Throws<FishAudioDrama3PromptTooLongException>(
            () => Prompt().Build("now", [], [new FishAudioDrama3Turn("first", codes)], maxRows: 600));
        Assert.Equal(600, tooLong.Limit);
    }

    private FishAudioDrama3Prompt Prompt() => new(text =>
    {
        _parts.Add(text);
        return [3, 4];
    }, Config());

    private static List<int[]> AudioRows(int[,] rows)
    {
        List<int[]> audio = [];
        for (int r = 0; r < rows.GetLength(0); r++)
        {
            if (rows[r, 0] is < 40 or > 47) continue;
            audio.Add([rows[r, 0], rows[r, 1], rows[r, 2], rows[r, 3], rows[r, 4]]);
        }
        return audio;
    }

    // The tiny fixture's geometry: four codebooks of eight codes, the band 40..47, eos 2.
    private static FishAudioDrama3Config Config() => FishAudioDrama3Config.FromJson(
        File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", "FishAudioDrama3", "config.json")));
}
