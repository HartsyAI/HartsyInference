using System.Text.Json;
using HartsyInference.LLM.Decision.Clef;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Records encoded by the official <c>encode_record</c> with the real Clef tokenizer
/// (<c>tools/clef/encode_reference.py</c>) must encode identically. Needs <c>HARTSY_CLEF_DIR</c> (a directory holding the
/// release's <c>tokenizer.json</c>); without it the test does nothing.</summary>
public sealed class ClefRecordEncoderTests
{
    [Fact]
    public void Encoding_MatchesOfficialEncodeRecord()
    {
        string? dir = Environment.GetEnvironmentVariable("HARTSY_CLEF_DIR");
        if (string.IsNullOrEmpty(dir))
        {
            return;
        }
        using FileStream stream = File.OpenRead(Path.Combine(dir, "tokenizer.json"));
        ClefRecordEncoder encoder = new(HfTokenizerJson.LoadByteLevelBpe(stream));
        using JsonDocument golden = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", "Clef",
            "clef_encode_golden.json")));
        foreach (JsonElement item in golden.RootElement.EnumerateArray())
        {
            ClefEncodedRecord got = encoder.Encode(item.GetProperty("record"));
            int[] want = item.GetProperty("input_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            Assert.Equal(want, got.InputIds);
            JsonElement[] questions = item.GetProperty("questions").EnumerateArray().ToArray();
            Assert.Equal(questions.Length, got.Questions.Length);
            for (int i = 0; i < questions.Length; i++)
            {
                Assert.Equal(questions[i].GetProperty("id").GetString(), got.Questions[i].QuestionId);
                Assert.Equal(questions[i].GetProperty("type").GetInt32(), got.Questions[i].QuestionType);
                JsonElement span = questions[i].GetProperty("question");
                Assert.Equal((span[0].GetInt32(), span[1].GetInt32()), got.Questions[i].QuestionSpan);
                Assert.Equal(questions[i].GetProperty("options").EnumerateArray().Select(s => (s[0].GetInt32(), s[1].GetInt32())),
                    got.Questions[i].OptionSpans);
                Assert.Equal(questions[i].GetProperty("optionIds").EnumerateArray().Select(s => s.GetString()), got.Questions[i].OptionIds);
            }
        }
    }

    [Theory]
    [InlineData(1250.0, "1250.0")]
    [InlineData(1e-7, "1e-07")]
    [InlineData(3e20, "3e+20")]
    [InlineData(0.0001, "0.0001")]
    [InlineData(1e15, "1000000000000000.0")]
    public void PythonFloatFormatting(double value, string expected) => Assert.Equal(expected, ClefJson.PythonFloat(value));
}
