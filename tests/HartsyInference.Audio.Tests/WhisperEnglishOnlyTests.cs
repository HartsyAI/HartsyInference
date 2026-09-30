using System.Diagnostics;
using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Real-weight CPU transcription of the JFK clip through an English-only checkpoint and through a
/// multilingual one. The <c>.en</c> checkpoints use a different special-token layout (SOT 50257, EOT 50256, no
/// language or task token in the prompt); feeding them the multilingual ids produced an empty transcript that ran to
/// the token limit. The multilingual run pins the decode so the layout fix cannot move it.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class WhisperEnglishOnlyTests
{
    private const string EnglishOnlyRepo = "openai/whisper-small.en";
    private const string MultilingualRepo = "openai/whisper-tiny";

    /// <summary>Every word of the JFK line except the leading "and so, my"; matched as whole words.</summary>
    private static readonly string[] JfkWords =
        ["fellow", "americans", "ask", "not", "what", "your", "country", "can", "do", "for", "you"];

    private readonly ITestOutputHelper _out;

    public WhisperEnglishOnlyTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public async Task SmallEn_TranscribesJfk_OnCpu()
    {
        string? heard = await TranscribeJfkAsync(EnglishOnlyRepo, new WhisperOptions());
        if (heard is null) return;
        AssertRecall(heard, minimum: 0.8);
    }

    [Fact]
    public async Task SmallEn_IgnoresLanguageAndTask_OnCpu()
    {
        // A caller that still sends the multilingual knobs must get the same English transcript: the prompt
        // carries no language or task slot on an English-only checkpoint.
        string? heard = await TranscribeJfkAsync(EnglishOnlyRepo, new WhisperOptions { Language = "en", Translate = true });
        if (heard is null) return;
        AssertRecall(heard, minimum: 0.8);
    }

    [Fact]
    public async Task Tiny_TranscribesJfk_OnCpu_Unchanged()
    {
        string? heard = await TranscribeJfkAsync(MultilingualRepo, new WhisperOptions { Language = "en" });
        if (heard is null) return;
        AssertRecall(heard, minimum: 0.8);
        // Pinned from the greedy F32 CPU decode on origin/main before the English-only layout fix.
        Assert.Equal(
            "and so my fellow americans ask not what your country can do for you ask what you can do for your country",
            Normalize(heard));
    }

    /// <summary>Loads the repo from the shared cache and transcribes the JFK clip on the CPU backend, or returns
    /// null after logging the skip when the staged weights or the clip are absent.</summary>
    private async Task<string?> TranscribeJfkAsync(string repo, WhisperOptions options)
    {
        string repoDir = AudioModelCache.GetRepoDirectory(repo, "stt");
        string jfk = Path.Combine(RepoRoot.Path, "tests", "python-reference", "silerovad_reference", "jfk.wav");
        if (!RealWeightGate.Require(_out.WriteLine,
                Path.Combine(repoDir, "model.safetensors"), Path.Combine(repoDir, "added_tokens.json"), jfk))
        {
            return null;
        }
        using IBackend backend = new CpuBackend();
        using WhisperPipeline pipeline = await WhisperPipeline.LoadAsync(repo);
        Stopwatch sw = Stopwatch.StartNew();
        string heard = pipeline.TranscribeWav(backend, jfk, options).Trim();
        sw.Stop();
        _out.WriteLine($"{repo} (CPU, {sw.Elapsed.TotalSeconds:0.0}s): \"{heard}\"");
        return heard;
    }

    private void AssertRecall(string heard, double minimum)
    {
        HashSet<string> words = new(Normalize(heard).Split(' ', StringSplitOptions.RemoveEmptyEntries), StringComparer.Ordinal);
        int hits = 0;
        foreach (string word in JfkWords)
        {
            if (words.Contains(word)) hits++;
        }
        double recall = hits / (double)JfkWords.Length;
        _out.WriteLine($"Content-word recall: {hits}/{JfkWords.Length} ({recall:P0})");
        Assert.True(recall >= minimum, $"recall {recall:P0} ({hits}/{JfkWords.Length}) on \"{heard}\"");
    }

    /// <summary>Lower-cases and keeps letters and single spaces only, so punctuation and casing cannot fail a pin.</summary>
    private static string Normalize(string text)
    {
        char[] buffer = new char[text.Length];
        int n = 0;
        bool pendingSpace = false;
        foreach (char c in text)
        {
            if (char.IsLetter(c))
            {
                if (pendingSpace && n > 0) buffer[n++] = ' ';
                pendingSpace = false;
                buffer[n++] = char.ToLowerInvariant(c);
            }
            else
            {
                pendingSpace = true;
            }
        }
        return new string(buffer, 0, n);
    }
}
