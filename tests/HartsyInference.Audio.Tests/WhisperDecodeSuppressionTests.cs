using HartsyInference.Audio.Pipelines;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>The checkpoint's <c>generation_config.json</c> suppression lists, as the decoder applies them. Self-contained:
/// no weights are loaded.</summary>
public sealed class WhisperDecodeSuppressionTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "whisper-suppression-" + Guid.NewGuid().ToString("N"));

    public WhisperDecodeSuppressionTests() => Directory.CreateDirectory(_dir);

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    private string Write(string json)
    {
        string path = Path.Combine(_dir, "generation_config.json");
        File.WriteAllText(path, json);
        return path;
    }

    [Fact]
    public void Load_ReadsBothListsFromGenerationConfig()
    {
        WhisperDecodeSuppression s = WhisperDecodeSuppression.Load(Write(
            """{"suppress_tokens": [1, 2, 931], "begin_suppress_tokens": [220, 50257], "max_length": 448}"""));

        Assert.Equal([1, 2, 931], s.SuppressTokens);
        Assert.Equal([220, 50257], s.BeginSuppressTokens);
    }

    [Fact]
    public void Load_MissingKeysYieldEmptyLists()
    {
        WhisperDecodeSuppression s = WhisperDecodeSuppression.Load(Write("""{"max_length": 448}"""));

        Assert.Empty(s.SuppressTokens);
        Assert.Empty(s.BeginSuppressTokens);
    }

    [Fact]
    public void Apply_MasksSuppressTokensAtEveryStep_AndBeginTokensOnlyFirst()
    {
        WhisperDecodeSuppression s = new([3], [4]);

        float[] later = [0f, 0f, 0f, 9f, 9f];
        s.Apply(later, firstStep: false);
        Assert.True(float.IsNegativeInfinity(later[3]));
        Assert.Equal(9f, later[4]);

        float[] first = [0f, 0f, 0f, 9f, 9f];
        s.Apply(first, firstStep: true);
        Assert.True(float.IsNegativeInfinity(first[3]));
        Assert.True(float.IsNegativeInfinity(first[4]));
    }

    [Fact]
    public void Apply_SuppressedTopScoreLosesToNextBest()
    {
        // Greedy argmax over the suppressed scores: a music-note id (here index 2) outscores every word, and the next
        // best wins once it is suppressed.
        WhisperDecodeSuppression s = new([2], []);
        float[] scores = [0.1f, 0.2f, 5.0f, 0.3f];
        s.Apply(scores, firstStep: false);

        int best = Array.IndexOf(scores, scores.Max());
        Assert.Equal(3, best);
    }

    [Fact]
    public void Apply_IgnoresIdsOutsideTheVocabulary()
    {
        WhisperDecodeSuppression s = new([-1, 99], [7]);
        float[] scores = [1f, 2f, 3f];

        s.Apply(scores, firstStep: true);

        Assert.Equal([1f, 2f, 3f], scores);
    }

    [Fact]
    public void None_SuppressesNothing()
    {
        float[] scores = [1f, 2f, 3f];

        WhisperDecodeSuppression.None.Apply(scores, firstStep: true);

        Assert.Equal([1f, 2f, 3f], scores);
    }
}
