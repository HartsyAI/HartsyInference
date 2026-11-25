using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines.LongCat;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The 8-step distilled schedule runs only with its DMD LoRA gate live. A run that asks for it without the gate is refused before
/// anything is computed, so the bare base is never run as the distilled model.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatDistilledGuardTests
{
    [Fact]
    public void ADistilled8StepRunWithoutItsLoraGateIsRefused()
    {
        using CpuBackend backend = new();
        using LongCatTransformer dit = new(LongCatTestData.Tiny);
        LongCatAvatarPipeline pipeline = new(backend, dit);
        LongCatPrompt prompt = new(new float[LongCatTestData.Tiny.CaptionChannels], 1, 1);
        LongCatAvatarRun run = new(LongCatTestData.Tiny.InChannels, 2, 2, 2, new float[LongCatTestData.Tiny.InChannels * 2 * 2 * 2],
            prompt, prompt, AudioTokens: [], UnconditionalAudioTokens: null, Steps: 8, Kind: LongCatAvatarSchedule.Kind.Distill8, Shift: 7f,
            TextGuidance: 1f, AudioGuidance: 1f, Mode: LongCatAvatarMode.At2v);

        InvalidOperationException refused = Assert.Throws<InvalidOperationException>(() => pipeline.Denoise(run));

        Assert.Contains("DMD LoRA gate", refused.Message, StringComparison.Ordinal);
    }
}
