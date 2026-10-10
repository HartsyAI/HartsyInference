using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Requests;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Pins the Qwen-Image-Edit reference conditioning, where every failure mode is silent. A placeholder run that
/// does not match what the vision tower emits produces conditioning the model still happily denoises; references handed
/// to the VAE at dims the packer cannot divide throw deep inside the pipeline; and a reference presented in the wrong
/// slot makes "the second image" in a prompt point at the wrong picture with no error anywhere.</summary>
public sealed class QwenImageEditConditioningTests
{
    private static readonly QwenImageEditTemplate Plus = QwenImageEditTemplate.EditPlus;
    private static readonly QwenImageEditTemplate V1 = QwenImageEditTemplate.Edit;

    private static ImageData Image(int width, int height) =>
        new ImageData { Rgb = new byte[(long)width * height * 3], Width = width, Height = height };

    /// <summary>The init image is Picture 1 and the extra references follow in order — the order a prompt saying
    /// "the first image" / "the second image" is addressing.</summary>
    [Fact]
    public void Resolve_PresentsInitImageFirst()
    {
        using QwenImageEditConditioning.References? references =
            QwenImageEditConditioning.Resolve(Plus, Image(640, 480), [Image(480, 640)]);
        Assert.NotNull(references);
        Assert.Equal(2, references!.Latent.Count);
        Assert.Equal(2, references.Vision.Count);
        // Landscape first, portrait second: the aspect ratio is what identifies the slot here.
        Assert.True(references.Latent[0].Shape[3] > references.Latent[0].Shape[2]);
        Assert.True(references.Latent[1].Shape[3] < references.Latent[1].Shape[2]);
    }

    /// <summary>The VAE copy lands on the ~1 MP budget with both sides divisible by 16, which is what the packed
    /// reference path validates; the vision copy lands on the much smaller ~384² budget the tower was trained on.</summary>
    [Theory]
    [InlineData(333, 777)]
    public void Resolve_RescalesEachReferenceForItsConsumer(int width, int height)
    {
        using QwenImageEditConditioning.References? references = QwenImageEditConditioning.Resolve(Plus, Image(width, height), null);
        Assert.NotNull(references);
        Tensor latent = references!.Latent[0];
        Assert.Equal(0, latent.Shape[2] % 16);
        Assert.Equal(0, latent.Shape[3] % 16);
        long latentArea = latent.Shape[2] * latent.Shape[3];
        Assert.InRange(latentArea, (long)(QwenImageEditConditioning.LatentTargetArea * 0.9),
            (long)(QwenImageEditConditioning.LatentTargetArea * 1.1));
        Tensor vision = references.Vision[0];
        long visionArea = vision.Shape[2] * vision.Shape[3];
        Assert.InRange(visionArea, (long)(Plus.VisionTargetArea * 0.9),
            (long)(Plus.VisionTargetArea * 1.1));
    }

    /// <summary>One <c>Picture i:</c> block per reference, each with exactly as many image-pad placeholders as the tower
    /// will emit merged tokens, wrapped in the vision start/end markers.</summary>
    [Fact]
    public void BuildTokens_EmitsOnePaddedPictureBlockPerReference()
    {
        using Qwen3Tokenizer tokenizer = new Qwen3Tokenizer();
        (WeightedTokenSequence sequence, int dropIndex) =
            QwenImageEditConditioning.BuildTokens(tokenizer, "make it red", [7, 3], Plus);
        int[] tokens = sequence.Tokens;
        Assert.Equal(2, tokens.Count(t => t == Qwen25VlMultimodalEncoder.VisionStartId));
        Assert.Equal(2, tokens.Count(t => t == Qwen25VlMultimodalEncoder.VisionEndId));
        Assert.Equal(10, tokens.Count(t => t == Qwen25VlMultimodalEncoder.ImageTokenId));
        int firstStart = Array.IndexOf(tokens, Qwen25VlMultimodalEncoder.VisionStartId);
        for (int i = 1; i <= 7; i++)
        {
            Assert.Equal(Qwen25VlMultimodalEncoder.ImageTokenId, tokens[firstStart + i]);
        }
        Assert.Equal(Qwen25VlMultimodalEncoder.VisionEndId, tokens[firstStart + 8]);
        // Everything the model is asked to look at must survive the prefix drop.
        Assert.True(firstStart > dropIndex);
    }

    /// <summary>The drop index ends the system block plus the <c>user</c> header, matching ComfyUI's
    /// <c>template_end</c> scan (second <c>&lt;|im_start|&gt;</c>, then +3 for <c>user</c> and the newline).</summary>
    [Fact]
    public void BuildTokens_DropIndexEndsTheUserHeader()
    {
        using Qwen3Tokenizer tokenizer = new Qwen3Tokenizer();
        (WeightedTokenSequence sequence, int dropIndex) = QwenImageEditConditioning.BuildTokens(tokenizer, "x", [4], Plus);
        int[] tokens = sequence.Tokens;
        int secondImStart = Array.IndexOf(tokens, Qwen3Tokenizer.ImStartId,
            Array.IndexOf(tokens, Qwen3Tokenizer.ImStartId) + 1);
        Assert.Equal(secondImStart + 3, dropIndex);
    }

    /// <summary>Three references spend well past the text-to-image path's 512-token cap, so the edit template must not
    /// inherit that truncation — it would cut the prompt, or the placeholders, off the end.</summary>
    [Fact]
    public void BuildTokens_IsNotTruncatedToTheTextToImageWindow()
    {
        using Qwen3Tokenizer tokenizer = new Qwen3Tokenizer();
        (WeightedTokenSequence sequence, _) = QwenImageEditConditioning.BuildTokens(tokenizer, "compose these", [196, 196, 196], Plus);
        int[] tokens = sequence.Tokens;
        Assert.Equal(588, tokens.Count(t => t == Qwen25VlMultimodalEncoder.ImageTokenId));
        Assert.True(tokens.Length > 512);
        // The assistant header is the last thing emitted, so truncation anywhere would lose it.
        int[] assistantTail = [Qwen3Tokenizer.ImStartId, .. tokenizer.EncodeRaw("assistant\n")];
        Assert.Equal(assistantTail, tokens[^assistantTail.Length..]);
    }

    /// <summary>A reference the tower would emit no tokens for cannot be addressed by the template, and silently
    /// emitting an empty block would shift every later picture index.</summary>
    [Fact]
    public void BuildTokens_RejectsAnEmptyPlaceholderRun()
    {
        using Qwen3Tokenizer tokenizer = new Qwen3Tokenizer();
        Assert.Throws<ArgumentOutOfRangeException>(() => QwenImageEditConditioning.BuildTokens(tokenizer, "x", [4, 0], Plus));
    }

    /// <summary>Edit v1 (ComfyUI <c>TextEncodeQwenImageEdit</c>) was trained on one image: the rest are dropped, not
    /// silently presented as a second picture the model has no slot for.</summary>
    [Fact]
    public void Resolve_EditV1_TakesOneReferenceAtTheOneMegapixelVisionArea()
    {
        using QwenImageEditConditioning.References? references = QwenImageEditConditioning.Resolve(V1, Image(1920, 1080), [Image(512, 512)]);
        Assert.NotNull(references);
        Assert.Single(references!.Latent);
        long visionArea = references.Vision[0].Shape[2] * references.Vision[0].Shape[3];
        Assert.InRange(visionArea, (long)(V1.VisionTargetArea * 0.9), (long)(V1.VisionTargetArea * 1.1));
    }

    /// <summary>v1's template puts the vision block straight after the user header — ComfyUI's
    /// <c>llama_template_images</c> has no <c>Picture 1:</c> label, and a label the model never saw shifts the text.</summary>
    [Fact]
    public void BuildTokens_EditV1_EmitsAnUnlabelledVisionBlock()
    {
        using Qwen3Tokenizer tokenizer = new Qwen3Tokenizer();
        (WeightedTokenSequence sequence, int dropIndex) = QwenImageEditConditioning.BuildTokens(tokenizer, "make it red", [5], V1);
        int[] tokens = sequence.Tokens;
        Assert.Equal(Qwen25VlMultimodalEncoder.VisionStartId, tokens[dropIndex]);
        Assert.Equal(5, tokens.Count(t => t == Qwen25VlMultimodalEncoder.ImageTokenId));
        (WeightedTokenSequence labelled, int labelledDrop) = QwenImageEditConditioning.BuildTokens(tokenizer, "make it red", [5], Plus);
        Assert.NotEqual(Qwen25VlMultimodalEncoder.VisionStartId, labelled.Tokens[labelledDrop]);
    }

    /// <summary>More references than the template addresses is a caller bug, refused rather than emitted.</summary>
    [Fact]
    public void BuildTokens_RejectsMoreReferencesThanTheTemplate()
    {
        using Qwen3Tokenizer tokenizer = new Qwen3Tokenizer();
        Assert.Throws<ArgumentOutOfRangeException>(() => QwenImageEditConditioning.BuildTokens(tokenizer, "x", [4, 4], V1));
    }
}
