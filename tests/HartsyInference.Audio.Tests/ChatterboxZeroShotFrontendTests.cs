using HartsyInference.Audio.Models.Chatterbox;
using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>CPU-lane checks of the zero-shot conditioning pieces Chatterbox and CosyVoice 2 take from upstream: librosa's
/// silence trim, the voice encoder's partial windows, the flow solver's time grid, the flow's prompt trim, CosyVoice's prompt alignment and S3Gen's
/// output fade. Expected values come from librosa 0.11, chatterbox <c>voice_encoder.py</c> and torch.</summary>
public sealed class ChatterboxZeroShotFrontendTests
{
    [Theory]
    [InlineData(20f, 2048, 10240)]
    [InlineData(60f, 2048, 14336)]
    public void SilenceTrim_MatchesLibrosaTrim(float topDb, int start, int end)
    {
        float[] y = Signal(3000, 6000, 4000, 2500);
        Assert.Equal((start, end), SilenceTrim.Bounds(y, topDb));
    }

    [Fact]
    public void SilenceTrim_ShortAndSilentSignals_MatchLibrosa()
    {
        Assert.Equal((0, 700), SilenceTrim.Bounds(Signal(0, 700, 0, 0), 20f));
        Assert.Equal((0, 5000), SilenceTrim.Bounds(new float[5000], 20f));
        Assert.Equal((0, 0), SilenceTrim.Bounds([], 20f));
    }

    [Theory]
    [InlineData(150, 0)]
    [InlineData(200, 1)]
    [InlineData(300, 1)]
    [InlineData(480, 3)]
    public void S3TokenizerMel_ShortClipsFrameAsTorchStft(int samples, int frames)
    {
        // torch.stft(center=True) gives 1 + len // 160 frames and the last is dropped; a clip at or under the 200-sample
        // pad (where torch's reflect pad would refuse) still frames without throwing.
        float[] y = Signal(0, samples, 0, 0);
        float[,] mel = new MelSpectrogramExtractor(MelSpectrogramExtractor.WhisperConfig(128)).Compute(y);
        Assert.Equal(frames, mel.GetLength(1));
        foreach (float v in mel) Assert.True(float.IsFinite(v));
    }

    [Theory]
    [InlineData(1, 1, 160)]
    [InlineData(160, 1, 160)]
    [InlineData(200, 1, 160)]
    [InlineData(236, 2, 237)]
    [InlineData(238, 2, 237)]
    [InlineData(300, 3, 314)]
    [InlineData(315, 3, 314)]
    [InlineData(442, 5, 468)]
    [InlineData(1000, 12, 1007)]
    public void VoiceEncoderPartials_UseResembleRateStep(int frames, int count, int target) =>
        Assert.Equal((count, target), ChatterboxVoiceEncoder.PartialWindows(frames));

    [Fact]
    public void CfmTimeGrid_CosineMatchesTorchAndLinearStaysUniform()
    {
        float[] torchCosine = [0f, 0.012311637f, 0.04894346f, 0.10899347f, 0.190983f, 0.29289323f, 0.4122148f, 0.5460095f,
            0.69098306f, 0.8435655f, 1f];
        float[] cosine = ConditionalCfm.TimeGrid(10, cosine: true);
        Assert.Equal(torchCosine.Length, cosine.Length);
        for (int i = 0; i < cosine.Length; i++) Assert.Equal(torchCosine[i], cosine[i], 1e-6f);
        float[] linear = ConditionalCfm.TimeGrid(10, cosine: false);
        for (int i = 0; i < linear.Length; i++) Assert.Equal(i / 10f, linear[i], 1e-7f);
        Assert.True(CosyVoiceConfig.V2_0_5B.Flow.CosineTimeSchedule);
    }

    [Fact]
    public void FlowPromptFrames_FollowThePromptMelLength()
    {
        using Tensor mel = new(new TensorShape(1, 80, 221), DType.F32);
        Assert.Equal(221, CosyVoiceFlow.PromptFrames(110, mel));
        Assert.Equal(220, CosyVoiceFlow.PromptFrames(110, null));
    }

    [Fact]
    public void AlignPromptToTokens_CutsBothToTheShorter()
    {
        Tensor mel = new(new TensorShape(1, 2, 7), DType.F32);
        Span<float> m = mel.AsSpan<float>();
        for (int i = 0; i < m.Length; i++) m[i] = i;
        int[] tokens = [1, 2, 3, 4];
        using Tensor cut = S3GenReference.AlignPromptToTokens(ref tokens, mel);
        Assert.Equal([1, 2, 3], tokens);
        Assert.Equal(6, cut.Shape[2]);
        Assert.Equal([0f, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], cut.AsSpan<float>().ToArray());

        Tensor even = new(new TensorShape(1, 1, 8), DType.F32);
        int[] many = [1, 2, 3, 4, 5, 6];
        using Tensor same = S3GenReference.AlignPromptToTokens(ref many, even);
        Assert.Same(even, same);
        Assert.Equal(4, many.Length);
    }

    [Fact]
    public void TrimFade_MatchesUpstreamBuffer()
    {
        float[] audio = Enumerable.Repeat(1f, 1200).ToArray();
        ChatterboxPipeline.ApplyTrimFade(audio, 24_000);
        (int Index, float Want)[] torchValues =
            [(0, 0f), (479, 0f), (480, 0f), (481, 1.0758638e-05f), (600, 0.14702678f), (719, 0.4983603f), (959, 1f), (960, 1f), (1199, 1f)];
        foreach ((int index, float want) in torchValues) Assert.Equal(want, audio[index], 1e-6f);

        float[] tiny = Enumerable.Repeat(1f, 500).ToArray();
        ChatterboxPipeline.ApplyTrimFade(tiny, 24_000);
        Assert.Equal(0f, tiny[479]);
        Assert.Equal(0f, tiny[480]);
        Assert.Equal(1.0758638e-05f, tiny[481], 1e-6f);
    }

    /// <summary>Silence, a loud 440 Hz tone, a quiet 97 Hz tone, silence, at 16 kHz (the fixture the expected values
    /// were computed on).</summary>
    private static float[] Signal(int silenceHead, int loud, int quiet, int silenceTail)
    {
        float[] y = new float[silenceHead + loud + quiet + silenceTail];
        for (int i = 0; i < loud; i++) y[silenceHead + i] = (float)(0.5 * Math.Sin(2 * Math.PI * 440 * i / 16000.0));
        for (int i = 0; i < quiet; i++) y[silenceHead + loud + i] = (float)(0.004 * Math.Sin(2 * Math.PI * 97 * i / 16000.0));
        return y;
    }
}
