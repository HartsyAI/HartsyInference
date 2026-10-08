using HartsyInference.Core.Exceptions;

namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>The V4.1 image encoder on the host reference path: the vision tower, the aligner, and the three learned embeddings that mark an image span.</summary>
/// <remarks>Matches upstream <c>Transformer.encode_image</c>. Splicing the result into a prompt's hidden states is the caller's job and is not done here. The model owns and disposes the tower and the aligner.</remarks>
public sealed class DeepSeekV41VisionModel : IDisposable
{
    private readonly float[] _imageStart;
    private readonly float[] _imageEnd;
    private readonly float[] _imageNewline;

    /// <param name="tower">The vision tower.</param>
    /// <param name="aligner">The aligner reading the tower's features.</param>
    /// <param name="imageStart">Learned embedding of the span's opening position, <c>[aligner.OutputDim]</c>.</param>
    /// <param name="imageEnd">Learned embedding of the closing position.</param>
    /// <param name="imageNewline">Learned embedding that ends each token row.</param>
    /// <exception cref="HartsyInferenceException">The tower and aligner disagree on the feature width, or an embedding is not as wide as the aligner's output.</exception>
    public DeepSeekV41VisionModel(DeepSeekV41VisionTower tower, DeepSeekV41Aligner aligner, float[] imageStart, float[] imageEnd, float[] imageNewline)
    {
        ArgumentNullException.ThrowIfNull(tower);
        ArgumentNullException.ThrowIfNull(aligner);
        ArgumentNullException.ThrowIfNull(imageStart);
        ArgumentNullException.ThrowIfNull(imageEnd);
        ArgumentNullException.ThrowIfNull(imageNewline);
        foreach ((string name, float[] values) in new[] { ("image_start", imageStart), ("image_end", imageEnd), ("image_newline", imageNewline) })
        {
            if (values.Length != aligner.OutputDim)
                throw new HartsyInferenceException($"'{name}' holds {values.Length} values, the aligner emits {aligner.OutputDim}.");
        }
        Tower = tower;
        Aligner = aligner;
        _imageStart = imageStart;
        _imageEnd = imageEnd;
        _imageNewline = imageNewline;
    }

    /// <summary>The vision tower.</summary>
    public DeepSeekV41VisionTower Tower { get; }

    /// <summary>The aligner.</summary>
    public DeepSeekV41Aligner Aligner { get; }

    /// <summary>Width of every embedding this model produces: the language model's hidden width.</summary>
    public int OutputDim => Aligner.OutputDim;

    /// <summary>Learned embedding of an image span's <c>IMAGE_START</c> position.</summary>
    public ReadOnlySpan<float> ImageStart => _imageStart;

    /// <summary>Learned embedding of an image span's <c>IMAGE_END</c> position.</summary>
    public ReadOnlySpan<float> ImageEnd => _imageEnd;

    /// <summary>Learned embedding of the <c>IMAGE_NEW_LINE</c> position that ends each token row.</summary>
    public ReadOnlySpan<float> ImageNewline => _imageNewline;

    /// <summary>The language-side token grid an image of <paramref name="gridHeight"/> x <paramref name="gridWidth"/> patches occupies.</summary>
    public (int Height, int Width) TokenGrid(int gridHeight, int gridWidth) =>
        DeepSeekV41Aligner.TokenGrid(gridHeight, gridWidth, Tower.Config.DownsampleRatio);

    /// <summary>Encodes one image's patches into embeddings for its <c>IMAGE</c> positions, <c>[tokenHeight * tokenWidth, OutputDim]</c> in reading order.</summary>
    /// <param name="patches">Flattened patches in row-major grid order, <c>[gridHeight * gridWidth, 3 * patch * patch]</c>.</param>
    /// <param name="gridHeight">Patch rows.</param>
    /// <param name="gridWidth">Patch columns.</param>
    public float[] Encode(ReadOnlySpan<float> patches, int gridHeight, int gridWidth) =>
        Aligner.Forward(Tower.Forward(patches, gridHeight, gridWidth), gridHeight, gridWidth);

    /// <inheritdoc />
    public void Dispose()
    {
        Tower.Dispose();
        Aligner.Dispose();
    }
}
