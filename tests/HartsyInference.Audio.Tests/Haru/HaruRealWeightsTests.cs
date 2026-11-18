using System.Text.Json;
using HartsyInference.Audio.Models.Haru;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests.Haru;

/// <summary>The released Haru-4B (<c>higurashi-ai/Haru-4B</c> @ 8d6e83bc) on the CPU backend against the upstream PyTorch model with the same
/// BF16 weights widened to F32 (<c>tools/haru/real_reference.py</c>). <c>HARU_REAL_DIR</c> names a folder with <c>model.safetensors</c>,
/// <c>config.json</c>, <c>tokenizer.json</c>, <c>real_lm_expected.safetensors</c>, <c>real_session_expected.safetensors</c> and
/// <c>real_session_expected.json</c>. It needs the 10.5 GB checkpoint and a few minutes of CPU per test.</summary>
[Trait("Category", "Integration")]
public sealed class HaruRealWeightsTests(ITestOutputHelper output)
{
    private static string? Directory()
    {
        string? dir = Environment.GetEnvironmentVariable("HARU_REAL_DIR");
        if (!string.IsNullOrEmpty(dir) && File.Exists(Path.Combine(dir, "model.safetensors")) && File.Exists(Path.Combine(dir, "real_lm_expected.safetensors")))
        {
            return dir;
        }
        Assert.False(Environment.GetEnvironmentVariable("HARTSY_REQUIRE_REAL_WEIGHTS") == "1", "HARU_REAL_DIR does not hold the released checkpoint and its reference outputs.");
        return null;
    }

    [Fact]
    public void LanguageModelAndHead_MatchUpstreamOnReleasedWeights()
    {
        if (Directory() is not string dir)
        {
            return;
        }
        using HaruReferenceModel model = new(Path.Combine(dir, "model.safetensors"), Path.Combine(dir, "config.json"), capacity: 64);
        string expected = Path.Combine(dir, "real_lm_expected.safetensors");
        (float hidden, float logits, float stream) = model.CompareLanguageModel(expected);
        float head = model.CompareHead(expected);
        output.WriteLine($"hidden states {hidden:E3}, text logits {logits:E3}, PAD/EOP logits {stream:E3}, head latents {head:E3}");
        Assert.True(hidden < 2e-4f, $"hidden states: relative error {hidden:E3}");
        Assert.True(logits < 2e-4f, $"text logits: relative error {logits:E3}");
        Assert.True(stream < 2e-4f, $"PAD/EOP logits: relative error {stream:E3}");
        Assert.True(head < 2e-4f, $"head latents: relative error {head:E3}");
    }

    [Fact]
    public void Conversation_MatchesUpstreamOnReleasedWeights()
    {
        if (Directory() is not string dir)
        {
            return;
        }
        string expected = Path.Combine(dir, "real_session_expected.safetensors");
        if (!File.Exists(expected))
        {
            Assert.False(Environment.GetEnvironmentVariable("HARTSY_REQUIRE_REAL_WEIGHTS") == "1", "real_session_expected.safetensors is missing.");
            return;
        }
        using HaruReferenceModel model = new(Path.Combine(dir, "model.safetensors"), Path.Combine(dir, "config.json"), capacity: 256);
        using FileStream tokenizerJson = File.OpenRead(Path.Combine(dir, "tokenizer.json"));
        HaruTextCodec text = HaruTextCodec.From(HaruTokenizer.Load(tokenizerJson, model.Config));
        using JsonDocument json = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "real_session_expected.json")));
        string card = json.RootElement.GetProperty("card").GetString()!;
        string[] wantText = [.. json.RootElement.GetProperty("text").EnumerateArray().Select(static e => e.GetString()!)];

        (bool tokens, bool sameText, float audio, int leftover, int[] ids) = model.ReplayConversation(expected, "run", card, "in.voice", wantText, text,
            new HaruDuplexOptions { Temperature = 0f, Seed = 1 });
        output.WriteLine($"tokens {string.Join(' ', ids)}; audio rel={audio:E3}; leftover draws {leftover}");
        Assert.True(tokens, "tokens differ from upstream");
        Assert.True(sameText, "text differs from upstream");
        Assert.Equal(0, leftover);
        Assert.True(audio < 5e-3f, $"audio relative error {audio:E3}");
    }
}
