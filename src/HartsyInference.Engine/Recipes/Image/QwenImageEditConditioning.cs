using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Requests;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Builds the Qwen-Image-Edit(-Plus) reference conditioning: the per-reference rescales the recipe needs and
/// the <c>Picture N:</c> instruction template the Qwen2.5-VL tower is conditioned on. Mirrors ComfyUI's
/// <c>TextEncodeQwenImageEditPlus</c> (<c>comfy_extras/nodes_qwen.py</c>) — each reference is presented twice, once to
/// the vision tower at ~384² area and once to the VAE at ~1 MP area, and the prompt is prefixed with one
/// <c>Picture i: &lt;|vision_start|&gt;&lt;|image_pad|&gt;…&lt;|vision_end|&gt;</c> block per reference.</summary>
public static class QwenImageEditConditioning
{
    /// <summary>Reference slots the edit-plus template was trained with (ComfyUI exposes exactly image1..image3).</summary>
    public const int MaxReferences = 3;

    /// <summary>Pixel budget for the copy fed to the Qwen2.5-VL vision tower.</summary>
    public const int VisionTargetArea = 384 * 384;

    /// <summary>Pixel budget for the copy fed to the VAE as an in-context reference latent.</summary>
    public const int LatentTargetArea = 1024 * 1024;

    /// <summary>The edit-plus system block, verbatim from ComfyUI's <c>llama_template</c>; it replaces the plain
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
    /// present, then <paramref name="extraReferences"/> — capped at <see cref="MaxReferences"/>. Returns null when
    /// there is nothing to condition on. Caller disposes.</summary>
    public static References? Resolve(ImageData? initImage, IReadOnlyList<ImageData>? extraReferences)
    {
        List<ImageData> ordered = new List<ImageData>(MaxReferences);
        if (initImage is not null)
        {
            ordered.Add(initImage);
        }
        if (extraReferences is not null)
        {
            foreach (ImageData reference in extraReferences)
            {
                if (reference is not null && ordered.Count < MaxReferences)
                {
                    ordered.Add(reference);
                }
            }
        }
        if (ordered.Count == 0)
        {
            return null;
        }
        List<Tensor> latent = new List<Tensor>(ordered.Count);
        List<Tensor> vision = new List<Tensor>(ordered.Count);
        try
        {
            foreach (ImageData reference in ordered)
            {
                // 16, not ComfyUI's 8: packing the reference latent needs the patch size to divide it too.
                (int latentW, int latentH) = ScaleToArea(reference.Width, reference.Height, LatentTargetArea, multiple: 16);
                latent.Add(FeatureImaging.RgbToTensorMinusOneOne(
                    FeatureImaging.ResizeRgb24(reference, latentW, latentH), latentW, latentH));
                (int visionW, int visionH) = ScaleToArea(reference.Width, reference.Height, VisionTargetArea, multiple: 1);
                vision.Add(FeatureImaging.RgbToTensorZeroOne(
                    FeatureImaging.ResizeRgb24(reference, visionW, visionH), visionW, visionH));
            }
        }
        catch
        {
            foreach (Tensor tensor in latent)
            {
                tensor.Dispose();
            }
            foreach (Tensor tensor in vision)
            {
                tensor.Dispose();
            }
            throw;
        }
        return new References { Latent = latent, Vision = vision };
    }

    /// <summary>Builds the templated edit-instruction token ids plus the prefix-drop index (the count of leading
    /// system-block and user-header tokens whose hidden states the pipeline discards). One <c>Picture i:</c> block is
    /// emitted per entry of <paramref name="visionTokenCounts"/>, each carrying that many <c>&lt;|image_pad|&gt;</c>
    /// placeholders for the tower's merged tokens.</summary>
    public static (int[] Tokens, int DropIndex) BuildTokens(Qwen3Tokenizer tokenizer, string prompt,
        IReadOnlyList<int> visionTokenCounts)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(visionTokenCounts);
        List<int> ids = new List<int>(1024);
        ids.Add(Qwen3Tokenizer.ImStartId);
        ids.AddRange(tokenizer.EncodeRaw(SystemPrompt));
        ids.Add(Qwen3Tokenizer.ImEndId);
        ids.AddRange(tokenizer.EncodeRaw("\n"));
        ids.Add(Qwen3Tokenizer.ImStartId);
        ids.AddRange(tokenizer.EncodeRaw("user\n"));
        int dropIndex = ids.Count;
        for (int i = 0; i < visionTokenCounts.Count; i++)
        {
            int count = visionTokenCounts[i];
            if (count <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(visionTokenCounts),
                    $"Reference {i + 1} reported {count} merged vision tokens; the template cannot address an empty image.");
            }
            ids.AddRange(tokenizer.EncodeRaw($"Picture {i + 1}: "));
            ids.Add(Qwen25VlMultimodalEncoder.VisionStartId);
            for (int pad = 0; pad < count; pad++)
            {
                ids.Add(Qwen25VlMultimodalEncoder.ImageTokenId);
            }
            ids.Add(Qwen25VlMultimodalEncoder.VisionEndId);
        }
        ids.AddRange(tokenizer.EncodeRaw(prompt));
        ids.Add(Qwen3Tokenizer.ImEndId);
        ids.AddRange(tokenizer.EncodeRaw("\n"));
        ids.Add(Qwen3Tokenizer.ImStartId);
        ids.AddRange(tokenizer.EncodeRaw("assistant\n"));
        // Deliberately NOT truncated to the text-to-image path's 512 tokens: three references alone spend ~560
        // placeholder tokens, and ComfyUI's Qwen tokenizer imposes no limit on the edit template either.
        return (ids.ToArray(), dropIndex);
    }

    /// <summary>Aspect-preserving rescale to <paramref name="targetArea"/> pixels, each side snapped to a multiple of
    /// <paramref name="multiple"/> (ComfyUI's <c>sqrt(total / (w·h))</c> scale, same banker's rounding).</summary>
    private static (int Width, int Height) ScaleToArea(int width, int height, int targetArea, int multiple)
    {
        double scale = Math.Sqrt(targetArea / ((double)width * height));
        int scaledW = Math.Max(multiple, (int)Math.Round(width * scale / multiple) * multiple);
        int scaledH = Math.Max(multiple, (int)Math.Round(height * scale / multiple) * multiple);
        return (scaledW, scaledH);
    }
}
