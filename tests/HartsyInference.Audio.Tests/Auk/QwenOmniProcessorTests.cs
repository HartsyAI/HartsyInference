using Xunit;
using HartsyInference.Audio.Models.QwenOmni;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Tests.Auk;

/// <summary>Length formulas, mel front-end layout and the placeholder splice of the Qwen2.5-Omni processor.</summary>
public sealed unsafe class QwenOmniProcessorTests
{
    private static int FloorDiv(int a, int b) => (int)Math.Floor((double)a / b);

    [Theory]
    [InlineData(1, 0)]
    [InlineData(7, 2)]
    [InlineData(203, 51)]
    public void AudioTokens_FollowTheFloorDivisionFormula(int frames, int expected)
    {
        Assert.Equal(expected, QwenOmniProcessor.AudioTokens(frames));
        Assert.Equal(expected, FloorDiv(FloorDiv(frames - 1, 2) + 1 - 2, 2) + 1);
    }

    [Fact]
    public void ConvFrames_EqualsTheSumOverChunks_ForEveryLength()
    {
        for (int frames = 1; frames <= 1_000; frames++)
        {
            int sum = 0;
            for (int rest = frames; rest > 0; rest -= 200) sum += QwenOmniProcessor.ConvFrames(Math.Min(200, rest));
            Assert.Equal(sum, QwenOmniProcessor.ConvFrames(frames));
        }
    }

    [Fact]
    public void ComputeMel_ZeroPadsTo300Seconds_AndReportsValidFrames()
    {
        QwenOmniConfig cfg = QwenOmniConfig.Default;
        QwenOmniProcessor processor = new(cfg);
        Assert.Equal(30_000, processor.PaddedFrames);
        float[] audio = new float[16_000];
        for (int i = 0; i < audio.Length; i++) audio[i] = 0.3f * MathF.Sin(2 * MathF.PI * 440f * i / 16_000f);
        Tensor mel = new(new TensorShape(128, 30_000), DType.F32);

        int frames = processor.ComputeMel(audio, mel);

        Assert.Equal(100, frames);
        float* p = (float*)mel.DataPointer;
        float tone = 0;
        float silence = 0;
        for (int m = 0; m < 128; m++)
        {
            tone += p[m * 30_000 + 50];
            silence += p[m * 30_000 + 29_000];
        }
        Assert.True(float.IsFinite(tone) && float.IsFinite(silence));
        Assert.True(tone > silence);
        Assert.Throws<ArgumentException>(() => processor.ComputeMel(new float[cfg.MaxSamples + 1], mel));
        mel.Dispose();
    }

    [Fact]
    public void SpliceAudioRows_ReplacesPlaceholdersInOrder()
    {
        int[] ids = [5, 9, 9, 7, 9];
        float[] embeds = new float[ids.Length * 2];
        for (int i = 0; i < embeds.Length; i++) embeds[i] = i;
        float[] audio = [100, 101, 200, 201, 300, 301];

        QwenOmniProcessor.SpliceAudioRows(ids, 9, embeds, audio, 2);

        Assert.Equal([0f, 1f, 100f, 101f, 200f, 201f, 6f, 7f, 300f, 301f], embeds);
        Assert.Throws<ArgumentException>(() => QwenOmniProcessor.SpliceAudioRows(ids, 9, embeds, audio[..4], 2));
        Assert.Throws<ArgumentException>(() => QwenOmniProcessor.SpliceAudioRows(ids, 9, embeds, [.. audio, 1f, 2f], 2));
    }
}
