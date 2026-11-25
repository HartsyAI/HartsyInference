using System.Text.Json;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.LLM.Tests.Maxwell;

/// <summary>The engine's tokenizer and chat template against the release's own tokenizer on probe strings from many scripts and a rendered chat
/// prompt (Fixtures/Maxwell/tokenizer_cases.json, written by tools/maxwell-reference/dump_tokenizer.py). Runs on the release's tokenizer.json,
/// which is not checked in: point MAXWELL_TOKENIZER_JSON at the downloaded file.</summary>
public sealed class MaxwellTokenizerParityTests
{
    private static readonly string FixtureDirectory = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Maxwell");

    [RealWeightsFact("MAXWELL_TOKENIZER_JSON")]
    [Trait("Category", "Integration")]
    public void Tokenizer_MatchesTheReleaseOnEveryProbe()
    {
        string directory = Path.GetDirectoryName(Environment.GetEnvironmentVariable("MAXWELL_TOKENIZER_JSON")!)!;
        using JsonDocument reference = JsonDocument.Parse(File.ReadAllText(Path.Combine(FixtureDirectory, "tokenizer_cases.json")));
        (string? bos, string? eos) = HfChatAssets.SpecialTokenLiterals(directory);
        List<int> stopIds = HfStopIds.Read(directory, configEos: 2);
        (ILlmTokenizer tokenizer, IChatTemplate template) = HfChatAssets.Load(directory, stopIds, bos, eos);

        Assert.Equal(reference.RootElement.GetProperty("bos_id").GetInt32(), tokenizer.BosId);
        Assert.Equal(reference.RootElement.GetProperty("eos_id").GetInt32(), tokenizer.EosId);

        List<string> mismatches = [];
        foreach (JsonElement probe in reference.RootElement.GetProperty("cases").EnumerateArray())
        {
            int[] expected = probe.GetProperty("ids").EnumerateArray().Select(static e => e.GetInt32()).ToArray();
            if (probe.TryGetProperty("text", out JsonElement text))
            {
                int[] actual = tokenizer.EncodeOrdinary(text.GetString()!);
                if (!actual.SequenceEqual(expected)) mismatches.Add($"text '{text.GetString()}': {actual.Length} vs {expected.Length} ids");
            }
            else
            {
                string question = probe.GetProperty("question").GetString()!;
                int[] actual = template.Encode(tokenizer, [new ChatMessage("user", question)], addGenerationPrompt: true, enableThinking: true);
                if (!actual.SequenceEqual(expected)) mismatches.Add($"chat prompt: {actual.Length} vs {expected.Length} ids");
            }
        }
        Assert.True(mismatches.Count == 0, string.Join("; ", mismatches));
    }
}
