using SharpInference.Audio.Models.Music;
using Xunit;

namespace SharpInference.Audio.Tests;

/// <summary>Checkpoint-free tests for MusicGen: the model presets and the delay-pattern staging math
/// (the one piece of MusicGen that is pure index bookkeeping and must be exactly right).</summary>
public sealed class MusicGenTests
{
    [Fact]
    public void Presets_MatchAudioCraftShapes()
    {
        Assert.Equal((1_024, 24, 16), Dims(MusicGenConfig.Small));
        Assert.Equal((1_536, 48, 24), Dims(MusicGenConfig.Medium));
        Assert.Equal((2_048, 48, 32), Dims(MusicGenConfig.Large));
        Assert.Equal(32_000, MusicGenConfig.Small.CodecSampleRate);
        Assert.Equal(16_000, MusicGenConfig.AudioGen.CodecSampleRate);
        Assert.Equal(4, MusicGenConfig.Small.NumCodebooks);
        Assert.Equal(2_048, MusicGenConfig.Small.SpecialToken);   // == codebook size
        Assert.Equal(64, MusicGenConfig.Small.HeadDim);           // 1024 / 16
    }

    [Fact]
    public void Delay_RevertUndoesApply()
    {
        int[] delay = [0, 1, 2, 3];
        int special = 2_048;
        int t = 5, k = 4;
        int[,] real = new int[t, k];
        for (int j = 0; j < t; j++)
            for (int c = 0; c < k; c++) real[j, c] = j * 10 + c;   // distinct, non-special values

        int[,] delayed = MusicGenDelay.Apply(real, delay, special);
        Assert.Equal(t + 3, delayed.GetLength(0));                 // grid grows by maxDelay
        int[,] back = MusicGenDelay.Revert(delayed, delay, t);

        for (int j = 0; j < t; j++)
            for (int c = 0; c < k; c++) Assert.Equal(real[j, c], back[j, c]);
    }

    [Fact]
    public void Delay_StaggersCodebooksWithSpecialLeadIn()
    {
        int[] delay = [0, 1, 2, 3];
        int special = 2_048;
        int[,] real = { { 7, 7, 7, 7 } };                          // single frame, all 7s
        int[,] delayed = MusicGenDelay.Apply(real, delay, special);

        // Codebook 0 is active at step 0; codebooks 1/2/3 are still in their special lead-in.
        Assert.Equal(7, delayed[0, 0]);
        Assert.Equal(special, delayed[0, 1]);
        Assert.Equal(special, delayed[0, 2]);
        Assert.Equal(special, delayed[0, 3]);
        // Each codebook's single real token lands on its delayed row.
        Assert.Equal(7, delayed[1, 1]);
        Assert.Equal(7, delayed[2, 2]);
        Assert.Equal(7, delayed[3, 3]);

        Assert.True(MusicGenDelay.IsActive(0, 0, delay));
        Assert.False(MusicGenDelay.IsActive(0, 3, delay));
        Assert.True(MusicGenDelay.IsActive(3, 3, delay));
    }

    private static (int, int, int) Dims(MusicGenConfig c) => (c.Hidden, c.NumLayers, c.NumHeads);
}
