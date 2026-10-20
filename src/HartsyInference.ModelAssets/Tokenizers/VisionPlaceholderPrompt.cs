using System.Globalization;
using System.Text;

namespace HartsyInference.ModelAssets.Tokenizers;

/// <summary>Qwen-VL style image placeholders in a text prompt: per image, an optional label then
/// <c>&lt;|vision_start|&gt;</c>, one <c>&lt;|image_pad|&gt;</c> per vision token and <c>&lt;|vision_end|&gt;</c>. Renders
/// text, so a family's chat template wraps it and encodes once with special tokens on. The ids come from the tokenizer,
/// never from constants.</summary>
public sealed class VisionPlaceholderPrompt
{
    /// <summary>Qwen-VL's literals.</summary>
    public const string DefaultVisionStart = "<|vision_start|>", DefaultImagePad = "<|image_pad|>", DefaultVisionEnd = "<|vision_end|>";

    /// <summary>The Qwen-Image-Edit-Plus / Ostris-edit label.</summary>
    public const string PictureLabel = "Picture {0}: ";

    private readonly string _visionStart;
    private readonly string _imagePad;
    private readonly string _visionEnd;

    /// <summary>Resolves the three literals through <paramref name="tokenizer"/>; throws naming any it lacks.</summary>
    public VisionPlaceholderPrompt(ILlmTokenizer tokenizer, string visionStart = DefaultVisionStart, string imagePad = DefaultImagePad,
        string visionEnd = DefaultVisionEnd)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        VisionStartId = Resolve(tokenizer, visionStart);
        ImagePadId = Resolve(tokenizer, imagePad);
        VisionEndId = Resolve(tokenizer, visionEnd);
        _visionStart = visionStart;
        _imagePad = imagePad;
        _visionEnd = visionEnd;
    }

    /// <summary>The default literals a tokenizer must know for this prompt.</summary>
    public static IReadOnlyList<string> RequiredLiterals { get; } = [DefaultVisionStart, DefaultImagePad, DefaultVisionEnd];

    /// <summary>Id of the vision-start token.</summary>
    public int VisionStartId { get; }

    /// <summary>Id of the per-vision-token placeholder.</summary>
    public int ImagePadId { get; }

    /// <summary>Id of the vision-end token.</summary>
    public int VisionEndId { get; }

    /// <summary>One block per image, in order, each holding <paramref name="padCounts"/>[i] placeholders and prefixed by
    /// <paramref name="labelFormat"/> with the 1-based image number (null for no label). No separators.</summary>
    public string Render(IReadOnlyList<int> padCounts, string? labelFormat = PictureLabel)
    {
        ArgumentNullException.ThrowIfNull(padCounts);
        StringBuilder text = new();
        for (int i = 0; i < padCounts.Count; i++)
        {
            if (padCounts[i] < 1)
                throw new ArgumentOutOfRangeException(nameof(padCounts), $"Image {i + 1} needs at least one vision token, got {padCounts[i]}.");
            if (labelFormat is not null)
                text.Append(string.Format(CultureInfo.InvariantCulture, labelFormat, i + 1));
            text.Append(_visionStart);
            for (int p = 0; p < padCounts[i]; p++) text.Append(_imagePad);
            text.Append(_visionEnd);
        }
        return text.ToString();
    }

    /// <summary>Placeholder tokens in <paramref name="ids"/>.</summary>
    public int CountPlaceholders(ReadOnlySpan<int> ids)
    {
        int count = 0;
        foreach (int id in ids)
        {
            if (id == ImagePadId) count++;
        }
        return count;
    }

    private static int Resolve(ILlmTokenizer tokenizer, string literal) =>
        tokenizer.SpecialId(literal) ?? throw new InvalidDataException($"The tokenizer has no '{literal}' token, which image input needs.");
}
