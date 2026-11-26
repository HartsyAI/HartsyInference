using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines.LongCat;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The 1.6 build's two-LoRA schedule: DMD (loraA) and OPSD (loraB) switched per step, as upstream <c>enable_lora_mix</c> does. The gate
/// states are read from inside the denoise loop, on the tiny config.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatLoraMixTests : IDisposable
{
    private const int Frames = 5, Height = 4, Width = 4, TextLength = 6, Steps = 8;
    private static readonly int Hw = Height * Width;
    private readonly CpuBackend _backend = new();
    private readonly LongCatConfig _config = LongCatTestData.TinyAvatar;
    private readonly Dictionary<string, Tensor> _weights;

    public LongCatLoraMixTests()
    {
        _weights = LongCatTestData.AvatarWeights(_config, 83);
    }

    public void Dispose()
    {
        LongCatTestData.DisposeAll(_weights);
        _backend.Dispose();
    }

    [Fact]
    public void TheCardsExampleRunsDmdOnTheFirstFourStepsAndOpsdOnTheLastFour()
    {
        LongCatAvatarLoraSchedule mix = new(new AdjunctGate(), new AdjunctGate(), Alpha: 1f, MixSteps: 4);
        for (int step = 0; step < 4; step++)
            Assert.Equal((1f, 0f), mix.ScalesAt(step));
        for (int step = 4; step < 8; step++)
            Assert.Equal((0f, 1f), mix.ScalesAt(step));
    }

    [Fact]
    public void AMixedAlphaBlendsTheTwoDuringTheMixedSteps()
    {
        LongCatAvatarLoraSchedule mix = new(new AdjunctGate(), new AdjunctGate(), Alpha: 0.25f, MixSteps: 2);
        Assert.Equal((0.25f, 0.75f), mix.ScalesAt(0));
        Assert.Equal((0.25f, 0.75f), mix.ScalesAt(1));
        Assert.Equal((0f, 1f), mix.ScalesAt(2));
    }

    [Fact]
    public void TheGatesAreSwitchedPerStepInsideTheDenoiseLoop()
    {
        LongCatAvatarLoraSchedule mix = new(new AdjunctGate(), new AdjunctGate(), Alpha: 1f, MixSteps: 4);
        List<(bool DmdOn, bool OpsdOn)> seen = [];
        RunDenoise(mix, progress: () => seen.Add((mix.Dmd.Enabled, mix.Opsd.Enabled)));

        Assert.Equal(Steps, seen.Count);
        for (int step = 0; step < Steps; step++)
            Assert.Equal(step < 4 ? (true, false) : (false, true), seen[step]);
    }

    [Fact]
    public void ABlendedMixEnablesBothGatesAtTheirScales()
    {
        LongCatAvatarLoraSchedule mix = new(new AdjunctGate(), new AdjunctGate(), Alpha: 0.5f, MixSteps: 3);
        List<(bool DmdOn, float DmdScale, bool OpsdOn, float OpsdScale)> seen = [];
        RunDenoise(mix, progress: () => seen.Add((mix.Dmd.Enabled, mix.Dmd.Scale, mix.Opsd.Enabled, mix.Opsd.Scale)));

        for (int step = 0; step < 3; step++)
            Assert.Equal((true, 0.5f, true, 0.5f), seen[step]);
        for (int step = 3; step < Steps; step++)
            Assert.Equal((false, 1f, true, 1f), seen[step]);
    }

    [Fact]
    public void NoGateIsOnAfterTheRun()
    {
        LongCatAvatarLoraSchedule mix = new(new AdjunctGate(), new AdjunctGate(), Alpha: 1f, MixSteps: 4);
        RunDenoise(mix, progress: () => { });
        Assert.False(mix.Dmd.Enabled);
        Assert.False(mix.Opsd.Enabled);
    }

    private void RunDenoise(LongCatAvatarLoraSchedule mix, Action progress)
    {
        Random rng = new(9);
        float[] noise = LongCatTestData.RandomValues(rng, _config.InChannels * Frames * Hw);
        float[] audio = LongCatTestData.RandomValues(rng, Frames * _config.AudioContextTokens * _config.AudioOutputDim);
        LongCatPrompt positive = new(LongCatTestData.RandomValues(rng, TextLength * _config.CaptionChannels), TextLength, TextLength);
        LongCatPrompt negative = new(LongCatTestData.RandomValues(rng, TextLength * _config.CaptionChannels), TextLength, TextLength);
        using LongCatTransformer transformer = new(_config);
        transformer.LoadWeights(_weights);
        LongCatAvatarRun run = new(_config.InChannels, Frames, Height, Width, noise, positive, negative, audio, null, Steps,
            LongCatAvatarSchedule.Kind.Distill8, 7f, 1f, 1f, LongCatAvatarMode.At2v, 0, null,
            OnProgress: _ => progress(), LoraMix: mix);
        new LongCatAvatarPipeline(_backend, transformer).Denoise(run);
    }
}
