using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

public sealed class MusicServicePromptTests
{
    private static readonly AudioClip Clip = new() { Data = [1, 2, 3] };

    private static ModelSpec Spec(string model) => new() { Requested = model, Modality = Modality.Music };

    [Fact]
    public void ControlFoley_AcceptsAVideoOrReferenceClipWithoutAPrompt()
    {
        Assert.True(MusicService.HasPromptOrConditioning(Spec("controlfoley"), new MusicRequest { Prompt = "", Video = new VideoClip { Data = [1] } }));
        Assert.True(MusicService.HasPromptOrConditioning(Spec("controlfoley"), new MusicRequest { Prompt = "", ReferenceAudio = Clip }));
    }

    [Fact]
    public void ControlFoley_StillNeedsSomethingToCondition()
    {
        Assert.False(MusicService.HasPromptOrConditioning(Spec("controlfoley"), new MusicRequest { Prompt = "" }));
    }

    [Fact]
    public void OtherModels_StillNeedAPromptOrGenre()
    {
        Assert.False(MusicService.HasPromptOrConditioning(Spec("acestep"), new MusicRequest { Prompt = "", ReferenceAudio = Clip }));
        Assert.True(MusicService.HasPromptOrConditioning(Spec("acestep"), new MusicRequest { Prompt = "", Genre = "jazz" }));
    }

    [Fact]
    public void ControlFoley_SelectorIsNormalisedAndMediaMustHaveBytes()
    {
        Assert.True(MusicService.HasPromptOrConditioning(Spec("  controlfoley  "), new MusicRequest { Prompt = "", ReferenceAudio = Clip }));
        Assert.False(MusicService.HasPromptOrConditioning(Spec("controlfoley"), new MusicRequest { Prompt = "", ReferenceAudio = new AudioClip { Data = [] } }));
        Assert.False(MusicService.HasPromptOrConditioning(Spec("controlfoley"), new MusicRequest { Prompt = "", Video = new VideoClip { Data = [] } }));
    }
}
