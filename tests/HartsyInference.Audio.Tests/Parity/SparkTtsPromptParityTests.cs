using System.Text.Json;
using HartsyInference.Audio.Pipelines;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Spark-TTS prompt parity against upstream <c>SparkTTS.process_prompt</c> / <c>process_prompt_control</c>
/// tokenized by HF <c>AutoTokenizer</c> (fixture from <c>tools/parity/spark_prompt_dump.py</c>). The prompt strings
/// are checked without weights; the token ids need <c>HARTSY_SPARK_DIR</c> naming a Spark-TTS-0.5B root whose
/// <c>LLM/</c> folder holds the tokenizer files.</summary>
public sealed class SparkTtsPromptParityTests
{
    private static readonly string FixturePath =
        Path.Combine(AppContext.BaseDirectory, "Fixtures", "SparkTts", "spark_prompt_ids.json");

    [Fact]
    public void PromptStrings_MatchUpstream()
    {
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(FixturePath));
        foreach (JsonElement c in doc.RootElement.GetProperty("cases").EnumerateArray())
        {
            Assert.Equal(c.GetProperty("prompt").GetString(), BuildPrompt(c));
        }
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void PromptIds_MatchUpstreamTokenizer()
    {
        string? root = Environment.GetEnvironmentVariable("HARTSY_SPARK_DIR");
        if (string.IsNullOrEmpty(root) || !Directory.Exists(Path.Combine(root, "LLM")))
        {
            Assert.False(Environment.GetEnvironmentVariable("HARTSY_REQUIRE_REAL_WEIGHTS") == "1",
                "HARTSY_SPARK_DIR is not set to a Spark-TTS-0.5B root.");
            return;
        }
        SparkTtsTokenizer tokenizer = SparkTtsTokenizer.FromDirectory(Path.Combine(root, "LLM"));
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(FixturePath));
        List<string> mismatches = [];
        int total = 0;
        foreach (JsonElement c in doc.RootElement.GetProperty("cases").EnumerateArray())
        {
            total++;
            int[] expected = c.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            int[] actual = tokenizer.Encode(c.GetProperty("prompt").GetString()!);
            if (!expected.AsSpan().SequenceEqual(actual))
            {
                mismatches.Add(c.GetProperty("mode").GetString() + " \"" + c.GetProperty("text").GetString()
                    + "\": expected [" + string.Join(",", expected) + "] got [" + string.Join(",", actual) + "]");
            }
        }
        Assert.True(mismatches.Count == 0,
            mismatches.Count + "/" + total + " prompts differ:\n" + string.Join("\n", mismatches));
    }

    private static string BuildPrompt(JsonElement c)
    {
        string text = c.GetProperty("text").GetString()!;
        if (c.GetProperty("mode").GetString() == "controllable")
        {
            return SparkTtsPipeline.BuildControllablePrompt(text, c.GetProperty("gender").GetString()!,
                c.GetProperty("pitch").GetString()!, c.GetProperty("speed").GetString()!);
        }
        int[] global = c.GetProperty("global_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        string? promptText = c.TryGetProperty("prompt_text", out JsonElement pt) ? pt.GetString() : null;
        int[]? semantic = c.TryGetProperty("semantic_ids", out JsonElement s)
            ? s.EnumerateArray().Select(e => e.GetInt32()).ToArray() : null;
        return SparkTtsPipeline.BuildZeroShotPrompt(text, global, promptText, semantic);
    }
}
