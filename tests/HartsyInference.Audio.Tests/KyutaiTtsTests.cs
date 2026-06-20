using HartsyInference.Audio.Models.Kyutai;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Checkpoint-free tests for Kyutai TTS: the temporal/depformer presets, the per-step weight-set
/// schedule, and the per-codebook delay roll (the pure index bookkeeping that must be exact).</summary>
public sealed class KyutaiTtsTests
{
    [Fact]
    public void Tts1_6B_TemporalAndDepthPresets()
    {
        KyutaiTtsConfig c = KyutaiTtsConfig.Tts1_6B;
        Assert.Equal(2_048, c.Temporal.HiddenSize);
        Assert.Equal(16, c.Temporal.NumHiddenLayers);
        Assert.Equal(10_000f, c.Temporal.RopeTheta);     // TTS uses θ=10000 (STT uses 100000)
        Assert.False(c.Temporal.AttentionBias);
        Assert.Equal(1_024, c.Depth.Dim);
        Assert.Equal(4, c.Depth.NumLayers);
        Assert.Equal(32, c.Depth.DepQ);
        Assert.Equal(128, c.Depth.LowRankEmb);
    }

    [Fact]
    public void DepthSchedule_HasElevenSets_FirstEightUnique()
    {
        MoshiDepthConfig d = MoshiDepthConfig.Tts1_6B;
        Assert.Equal(11, d.NumWeightSets);
        for (int k = 0; k < 8; k++) Assert.Equal(k, d.WeightSet(k));   // 0..7 unique
        for (int k = 8; k < 16; k++) Assert.Equal(8, d.WeightSet(k));  // next 8 → set 8
        for (int k = 16; k < 24; k++) Assert.Equal(9, d.WeightSet(k)); // → set 9
        for (int k = 24; k < 32; k++) Assert.Equal(10, d.WeightSet(k));// → set 10
    }

    [Fact]
    public void Delays_TextAndSemanticZero_AcousticTwo()
    {
        KyutaiTtsConfig c = KyutaiTtsConfig.Tts1_6B;
        int[] d = c.BuildDelays();
        Assert.Equal(1 + 32, d.Length);
        Assert.Equal(0, d[0]);          // text
        Assert.Equal(0, d[1]);          // codebook 0 (semantic)
        Assert.Equal(2, d[2]);          // codebook 1 (acoustic)
        Assert.Equal(2, d[^1]);         // codebook 31
    }

    [Fact]
    public void MoshiDelay_ApplyThenRevertRoundTrips()
    {
        int[] delays = [0, 0, 2, 2];    // 4 streams
        int fill = 9999;
        int n = 4, t = 5;
        int[,] real = new int[n, t];
        for (int s = 0; s < n; s++)
            for (int j = 0; j < t; j++) real[s, j] = s * 10 + j;

        int[,] delayed = MoshiDelay.Apply(real, delays, fill);
        Assert.Equal(t + 2, delayed.GetLength(1));        // grid grows by max delay
        Assert.Equal(fill, delayed[2, 0]);                // acoustic stream's lead-in is fill
        Assert.Equal(real[2, 0], delayed[2, 2]);          // real content shifted by 2

        int[,] back = MoshiDelay.Revert(delayed, delays, t);
        for (int s = 0; s < n; s++)
            for (int j = 0; j < t; j++) Assert.Equal(real[s, j], back[s, j]);
    }
}
