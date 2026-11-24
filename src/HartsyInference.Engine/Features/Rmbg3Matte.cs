using System.Text.Json.Nodes;
using HartsyInference.Core.Imaging;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Features;

/// <summary>Bria RMBG 3.0 as a matte: the Fibo 2.0 edit path draws a greyscale picture of the subject, and this turns it into an
/// 8-bit alpha at the source size, as the model card does (<c>result.convert("L").resize(source.size, LANCZOS)</c>).</summary>
public static class Rmbg3Matte
{
    /// <summary>The one instruction the model was trained on; a changed or missing one gives a coloured picture (model card).</summary>
    public const string Instruction = "Generate a detailed grayscale alpha matte. Map the opaque foreground to white and the background "
        + "to black. Produce soft, anti-aliased grayscale gradients at the edges of the subject to represent fine details and transparency.";

    /// <summary>The card's seed, so a cutout is reproducible.</summary>
    public const int Seed = 0;

    /// <summary>Whether <paramref name="spec"/> is one of the RMBG 3.0 catalog entries.</summary>
    public static bool Handles(ModelSpec spec)
    {
        ArgumentNullException.ThrowIfNull(spec);
        return spec.Catalog?.Id is "rmbg3" or "rmbg3-int8";
    }

    /// <summary>Generates the matte for <paramref name="image"/> and returns it as straight 8-bit alpha, <c>width × height</c>.</summary>
    public static async Task<byte[]> AlphaAsync(InferenceEngine engine, ModelSpec spec, ImageData image, CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(engine);
        ArgumentNullException.ThrowIfNull(image);
        if (!Handles(spec))
            throw new ArgumentException($"'{spec.Requested}' is not an RMBG 3.0 model.", nameof(spec));
        (int width, int height) = Fibo2EditImages.TrainedSize(image.Width, image.Height);
        ImageRequest request = new()
        {
            Prompt = Instruction,
            Width = width,
            Height = height,
            Seed = Seed,
            Img2Img = new Img2Img { InitImage = image, Mode = Img2ImgMode.Reference },
        };
        ImageResult matte = await engine.Images.GenerateAsync(spec with { Modality = Modality.Image }, request, null, cancel)
            .ConfigureAwait(false);
        return AlphaFromPicture(matte.Rgb, matte.Width, matte.Height, image.Width, image.Height);
    }

    /// <summary>Grey level of each pixel of the generated RGB picture (PIL's integer ITU-R 601 <c>convert("L")</c>), resized to the
    /// source size with PIL's Lanczos.</summary>
    public static byte[] AlphaFromPicture(ReadOnlySpan<byte> rgb, int width, int height, int sourceWidth, int sourceHeight)
    {
        if (width <= 0 || height <= 0 || sourceWidth <= 0 || sourceHeight <= 0)
            throw new ArgumentException($"Sizes must be positive; got {width}x{height} and {sourceWidth}x{sourceHeight}.");
        if (rgb.Length < (long)width * height * 3)
            throw new ArgumentException($"RGB buffer holds {rgb.Length} bytes; a {width}x{height} picture needs {(long)width * height * 3}.", nameof(rgb));
        byte[] grey = new byte[width * height];
        for (int i = 0; i < grey.Length; i++)
            grey[i] = (byte)((rgb[i * 3] * 19595 + rgb[i * 3 + 1] * 38470 + rgb[i * 3 + 2] * 7471 + 0x8000) >> 16);
        if (width == sourceWidth && height == sourceHeight)
            return grey;
        byte[] alpha = new byte[sourceWidth * sourceHeight];
        Resample.LanczosPil8(grey, width, height, 1, alpha, sourceWidth, sourceHeight);
        return alpha;
    }

    /// <summary>Accepts a prompt that is blank, the fixed instruction, or a JSON object whose <c>edit_instruction</c> is it.</summary>
    /// <exception cref="NotSupportedException">The prompt asks for something else: the model only draws its one trained matte.</exception>
    public static void RequireInstruction(string? prompt)
    {
        if (string.IsNullOrWhiteSpace(prompt) || string.Equals(prompt.Trim(), Instruction, StringComparison.Ordinal))
            return;
        if (TryReadInstruction(prompt, out string? text) && string.Equals(text, Instruction, StringComparison.Ordinal))
            return;
        throw new NotSupportedException("RMBG 3.0 was trained on one fixed instruction and gives a coloured picture for any other; "
            + "leave the prompt empty.");
    }

    private static bool TryReadInstruction(string prompt, out string? text)
    {
        text = null;
        try
        {
            if (JsonNode.Parse(prompt) is JsonObject json && json["edit_instruction"] is JsonValue value)
                return value.TryGetValue(out text);
        }
        catch (System.Text.Json.JsonException)
        {
            // Not JSON: the caller refuses it as an unknown instruction.
        }
        return false;
    }
}
