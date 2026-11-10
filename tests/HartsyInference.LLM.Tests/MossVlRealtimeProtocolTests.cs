using HartsyInference.LLM.Models.MossVl;

namespace HartsyInference.LLM.Tests;

/// <summary>MOSS-VL's prompt layouts, compared with the strings the released processor and realtime session build
/// (<c>processing_moss_vl.py</c> lines 688-694, <c>modeling_moss_vl.py</c> lines 3205-3228).</summary>
public sealed class MossVlRealtimeProtocolTests
{
    private static readonly MossVlRealtimeProtocol Protocol = new(imagePlaceholderId: 151655, silenceTokenId: 151671);

    [Fact]
    public void OfflineImagesAreBarePadsAndAVideoIsOneTimedBlock()
    {
        Assert.Equal("<|im_start|>user\n<|image_pad|><|image_pad|>What is this?<|im_end|>\n<|im_start|>assistant\n",
            Protocol.OfflinePrompt(2, [], "What is this?"));
        Assert.Equal(
            "<|im_start|>user\n<|vision_start|><|time_start|>0.0 seconds<|time_end|><|image_pad|>"
            + "<|time_start|>0.2 seconds<|time_end|><|image_pad|><|vision_end|>Describe.<|im_end|>\n<|im_start|>assistant\n",
            Protocol.OfflinePrompt(0, [0.0, 0.25], "\nDescribe."));
    }

    [Fact]
    public void LiveSegmentsMatchTheSession()
    {
        Assert.Equal("<|vision_start|><|time_start|>12.5 seconds<|time_end|><|image_pad|><|vision_end|>", Protocol.FrameSegment(12.5));
        Assert.Equal("<|im_end|>\n<|im_start|>user\nWhat changed?<|im_end|>\n<|im_start|>assistant\n", Protocol.PromptSegment("What changed?"));
        Assert.Equal("<|silence|>", Protocol.AfterPromptsWithFrames);
        Assert.Equal("<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n", Protocol.InitialPrompt("S", "U"));
        Assert.Equal(151671, Protocol.SilenceTokenId);
        Assert.Contains("<|silence|>", Protocol.DefaultSystemPrompt);
    }
}
