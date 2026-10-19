using System.Runtime.InteropServices;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Requests;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Builds the Qwen-Image-Edit reference conditioning: the per-reference rescales the recipe needs and the
/// instruction template the Qwen2.5-VL tower is conditioned on. Mirrors ComfyUI's <c>TextEncodeQwenImageEdit</c> and
/// <c>TextEncodeQwenImageEditPlus</c> (<c>comfy_extras/nodes_qwen.py</c>), selected by a
/// <see cref="QwenImageEditTemplate"/> — each reference is presented twice, once to the vision tower at the template's
/// area and once to the VAE at ~1 MP, and the prompt is prefixed with one
/// <c>&lt;|vision_start|&gt;&lt;|image_pad|&gt;…&lt;|vision_end|&gt;</c> block per reference (labelled <c>Picture i:</c>
/// for Plus).</summary>
public static class QwenImageEditConditioning
{
    /// <summary>Pixel budget for the copy fed to the VAE as an in-context reference latent.</summary>
    public const int LatentTargetArea = 1024 * 1024;

    /// <summary>The edit system block, verbatim from ComfyUI's <c>llama_template_images</c> (shared by both nodes); it replaces the plain
    /// text-to-image "Describe the image…" block and is what makes the model read the prompt as an instruction.</summary>
    public const string SystemPrompt =
        "system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), "
        + "then explain how the user's text instruction should alter or modify the image. Generate a new image that "
        + "meets the user's requirements while maintaining consistency with the original input where appropriate.";

    /// <summary>Resolved reference tensors; owns every tensor it exposes.</summary>
    public sealed class References : IDisposable
    {
        /// <summary>VAE-contract references <c>[1, 3, H, W]</c> in <c>[-1, 1]</c>, dims divisible by 16, in Picture order.</summary>
        public required IReadOnlyList<Tensor> Latent { get; init; }

        /// <summary>Vision-tower references <c>[1, 3, H, W]</c> in <c>[0, 1]</c>, same order and count as <see cref="Latent"/>.</summary>
        public required IReadOnlyList<Tensor> Vision { get; init; }

        /// <summary>Frees every reference tensor.</summary>
        public void Dispose()
        {
            foreach (Tensor tensor in Latent)
            {
                tensor.Dispose();
            }
            foreach (Tensor tensor in Vision)
            {
                tensor.Dispose();
            }
        }
    }

    /// <summary>Resolves the reference set in presentation order — <paramref name="initImage"/> is Picture 1 when
    /// present, then <paramref name="extraReferences"/> — capped at <paramref name="template"/>'s reference count.
    /// Returns null when there is nothing to condition on. Caller disposes.</summary>
    public static References? Resolve(QwenImageEditTemplate template, ImageData? initImage, IReadOnlyList<ImageData>? extraReferences)
    {
        ArgumentNullException.ThrowIfNull(template);
        ReferenceImagePolicy policy = PolicyFor(template);
        IReadOnlyList<ImageData> ordered = ReferenceImageResolver.Collect(initImage, extraReferences, policy, "QwenImageEdit");
        ResolvedReferences? resolved = ReferenceImageResolver.Resolve(ordered, policy);
        return resolved is null ? null : new References { Latent = resolved.Latent, Vision = resolved.Vision };
    }

    /// <summary>The template's reference policy: the VAE copy at ~1 MP and the vision copy at the template's area, both
    /// scaled up or down as ComfyUI's encode nodes do, extra images dropped.</summary>
    public static ReferenceImagePolicy PolicyFor(QwenImageEditTemplate template)
    {
        ArgumentNullException.ThrowIfNull(template);
        return new ReferenceImagePolicy
        {
            MaxReferences = template.MaxReferences,
            LatentTargetArea = LatentTargetArea,
            LatentUpscale = true,
            // 16, not ComfyUI's 8: packing the reference latent needs the patch size to divide it too.
            LatentMultiple = 16,
            VisionTargetArea = template.VisionTargetArea,
            VisionUpscale = true,
            Overflow = ReferenceOverflow.Drop,
        };
    }

    /// <summary>Builds the templated edit-instruction token ids plus the prefix-drop index (the count of leading
    /// system-block and user-header tokens whose hidden states the pipeline discards). One vision block (labelled
    /// <c>Picture i:</c> when the template says so) is emitted per entry of <paramref name="visionTokenCounts"/>, each
    /// carrying that many <c>&lt;|image_pad|&gt;</c> placeholders for the tower's merged tokens.</summary>
    public static (WeightedTokenSequence Tokens, int DropIndex) BuildTokens(Qwen3Tokenizer tokenizer, string prompt,
        IReadOnlyList<int> visionTokenCounts, QwenImageEditTemplate template)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(visionTokenCounts);
        ArgumentNullException.ThrowIfNull(template);
        if (visionTokenCounts.Count > template.MaxReferences)
        {
            throw new ArgumentOutOfRangeException(nameof(visionTokenCounts),
                $"{visionTokenCounts.Count} references exceed the template's {template.MaxReferences}.");
        }
        List<int> prefix = new List<int>(1024);
        prefix.Add(Qwen3Tokenizer.ImStartId);
        prefix.AddRange(tokenizer.EncodeRaw(SystemPrompt));
        prefix.Add(Qwen3Tokenizer.ImEndId);
        prefix.AddRange(tokenizer.EncodeRaw("\n"));
        prefix.Add(Qwen3Tokenizer.ImStartId);
        prefix.AddRange(tokenizer.EncodeRaw("user\n"));
        int dropIndex = prefix.Count;
        // The vision blocks sit AFTER the drop index, so they reach the conditioning — but they are template, not
        // prompt, and carry no emphasis of their own.
        for (int i = 0; i < visionTokenCounts.Count; i++)
        {
            int count = visionTokenCounts[i];
            if (count <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(visionTokenCounts),
                    $"Reference {i + 1} reported {count} merged vision tokens; the template cannot address an empty image.");
            }
            if (template.LabelPictures)
            {
                prefix.AddRange(tokenizer.EncodeRaw($"Picture {i + 1}: "));
            }
            prefix.Add(Qwen25VlMultimodalEncoder.VisionStartId);
            for (int pad = 0; pad < count; pad++)
            {
                prefix.Add(Qwen25VlMultimodalEncoder.ImageTokenId);
            }
            prefix.Add(Qwen25VlMultimodalEncoder.VisionEndId);
        }
        List<int> suffix = new List<int>(8);
        suffix.Add(Qwen3Tokenizer.ImEndId);
        suffix.AddRange(tokenizer.EncodeRaw("\n"));
        suffix.Add(Qwen3Tokenizer.ImStartId);
        suffix.AddRange(tokenizer.EncodeRaw("assistant\n"));
        // Deliberately NOT truncated to the text-to-image path's 512 tokens: three references alone spend ~560
        // placeholder tokens, and ComfyUI's Qwen tokenizer imposes no limit on the edit template either.
        WeightedTokenSequence sequence = WeightedTokenBuilder.Build(
            prompt, tokenizer.EncodeRaw, CollectionsMarshal.AsSpan(prefix), CollectionsMarshal.AsSpan(suffix));
        return (sequence, dropIndex);
    }
}
