using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;
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
    private static ImageData Image(int width, int height) =>
        new ImageData { Rgb = new byte[(long)width * height * 3], Width = width, Height = height };

    /// <summary>The init image is Picture 1 and the extra references follow in order — the order a prompt saying
    /// "the first image" / "the second image" is addressing.</summary>
    [Fact]
    public void Resolve_PresentsInitImageFirst()
    {
        using QwenImageEditConditioning.References? references =
            QwenImageEditConditioning.Resolve(Image(640, 480), [Image(480, 640)]);
        Assert.NotNull(references);
        Assert.Equal(2, references!.Latent.Count);
        Assert.Equal(2, references.Vision.Count);
        // Landscape first, portrait second: the aspect ratio is what identifies the slot here.
        Assert.True(references.Latent[0].Shape[3] > references.Latent[0].Shape[2]);
        Assert.True(references.Latent[1].Shape[3] < references.Latent[1].Shape[2]);
    }

    /// <summary>Only the slots the edit-plus template was trained with are consumed; extras are dropped rather than
    /// silently extending a template the model never saw.</summary>
    [Fact]
    public void Resolve_CapsAtThreeReferences()
    {
        using QwenImageEditConditioning.References? references = QwenImageEditConditioning.Resolve(
            Image(512, 512), [Image(512, 512), Image(512, 512), Image(512, 512)]);
        Assert.NotNull(references);
        Assert.Equal(QwenImageEditConditioning.MaxReferences, references!.Latent.Count);
    }

    /// <summary>No init image and no references is text-to-image, not an empty edit.</summary>
    [Fact]
    public void Resolve_WithNothingAttached_ReturnsNull()
    {
        Assert.Null(QwenImageEditConditioning.Resolve(null, null));
        Assert.Null(QwenImageEditConditioning.Resolve(null, []));
    }

    /// <summary>The VAE copy lands on the ~1 MP budget with both sides divisible by 16, which is what the packed
    /// reference path validates; the vision copy lands on the much smaller ~384² budget the tower was trained on.</summary>
    [Theory]
    [InlineData(1920, 1080)]
    [InlineData(640, 480)]
    [InlineData(333, 777)]
    [InlineData(64, 64)]
    public void Resolve_RescalesEachReferenceForItsConsumer(int width, int height)
    {
        using QwenImageEditConditioning.References? references = QwenImageEditConditioning.Resolve(Image(width, height), null);
        Assert.NotNull(references);
        Tensor latent = references!.Latent[0];
        Assert.Equal(0, latent.Shape[2] % 16);
        Assert.Equal(0, latent.Shape[3] % 16);
        long latentArea = latent.Shape[2] * latent.Shape[3];
        Assert.InRange(latentArea, (long)(QwenImageEditConditioning.LatentTargetArea * 0.9),
            (long)(QwenImageEditConditioning.LatentTargetArea * 1.1));
        Tensor vision = references.Vision[0];
        long visionArea = vision.Shape[2] * vision.Shape[3];
        Assert.InRange(visionArea, (long)(QwenImageEditConditioning.VisionTargetArea * 0.9),
            (long)(QwenImageEditConditioning.VisionTargetArea * 1.1));
    }

    /// <summary>Aspect ratio survives both rescales — a stretched reference edits as a stretched subject.</summary>
    [Fact]
    public void Resolve_PreservesAspectRatio()
    {
        using QwenImageEditConditioning.References? references = QwenImageEditConditioning.Resolve(Image(1920, 1080), null);
        Assert.NotNull(references);
        foreach (Tensor reference in new[] { references!.Latent[0], references.Vision[0] })
        {
            double aspect = (double)reference.Shape[3] / reference.Shape[2];
            Assert.InRange(aspect, 1920.0 / 1080.0 - 0.05, 1920.0 / 1080.0 + 0.05);
        }
    }

    /// <summary>One <c>Picture i:</c> block per reference, each with exactly as many image-pad placeholders as the tower
    /// will emit merged tokens, wrapped in the vision start/end markers.</summary>
    [Fact]
    public void BuildTokens_EmitsOnePaddedPictureBlockPerReference()
    {
        using Qwen3Tokenizer tokenizer = new Qwen3Tokenizer();
        (int[] tokens, int dropIndex) = QwenImageEditConditioning.BuildTokens(tokenizer, "make it red", [7, 3]);
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
        (int[] tokens, int dropIndex) = QwenImageEditConditioning.BuildTokens(tokenizer, "x", [4]);
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
        (int[] tokens, _) = QwenImageEditConditioning.BuildTokens(tokenizer, "compose these", [196, 196, 196]);
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
        Assert.Throws<ArgumentOutOfRangeException>(() => QwenImageEditConditioning.BuildTokens(tokenizer, "x", [4, 0]));
    }
}
