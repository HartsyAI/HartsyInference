using HartsyInference.Engine.Audio;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Unit tests for <see cref="TtsChunkedSynthesis.Synthesize"/>: chunk ordering, the silence join between chunks,
/// and the empty-input short circuit.</summary>
public sealed class TtsChunkedSynthesisTests
{
    private const int SampleRate = 1000;

    [Fact]
    public void SingleChunk_ReturnsItsAudioUnchanged_WithNoGap()
    {
        float[] chunk = [0.25f, -0.5f, 0.75f];

        float[] samples = TtsChunkedSynthesis.Synthesize("one two", 40, SampleRate, (_, _) => chunk);

        Assert.Equal(chunk, samples);
    }

    [Fact]
    public void TwoChunks_AreJoinedWithOneGapOfZeros()
    {
        int gap = (int)Math.Round(SampleRate * TtsChunkedSynthesis.GapSeconds);
        float[] first = Enumerable.Repeat(0.5f, 100).ToArray();
        float[] second = Enumerable.Repeat(1.0f, 200).ToArray();

        float[] samples = TtsChunkedSynthesis.Synthesize("one two three four", 2, SampleRate,
            (index, _) => index == 0 ? first : second);

        Assert.Equal(100 + gap + 200, samples.Length);
        Assert.Equal(first, samples[..100]);
        Assert.All(samples[100..(100 + gap)], value => Assert.Equal(0f, value));
        Assert.Equal(second, samples[(100 + gap)..]);
    }

    [Fact]
    public void ChunksArriveInOrder_AndConcatenateBackToTheInputWords()
    {
        const string text = "one two three four five six seven";
        List<(int Index, string Text)> calls = [];

        TtsChunkedSynthesis.Synthesize(text, 3, SampleRate, (index, chunk) =>
        {
            calls.Add((index, chunk));
            return [0f];
        });

        Assert.Equal(Enumerable.Range(0, calls.Count), calls.Select(call => call.Index));
        Assert.True(calls.Count > 1);
        string[] words = string.Join(" ", calls.Select(call => call.Text))
            .Split(' ', StringSplitOptions.RemoveEmptyEntries);
        Assert.Equal(text.Split(' '), words);
    }

    [Fact]
    public void EmptyText_ReturnsEmptyArray_AndNeverCallsTheFunction()
    {
        int calls = 0;

        float[] samples = TtsChunkedSynthesis.Synthesize(string.Empty, 40, SampleRate, (_, _) =>
        {
            calls++;
            return [1f];
        });

        Assert.Empty(samples);
        Assert.Equal(0, calls);
    }
}
