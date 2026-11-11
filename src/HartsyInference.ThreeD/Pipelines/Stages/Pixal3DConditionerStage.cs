using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.Pixal3D;
using HartsyInference.ThreeD.Models.TrellisV2;

namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>
/// Pixal3D's image conditioning for one flow stage: DINOv3 tokens from the shared <see cref="Trellis2ImageConditioner"/> (the same checkpoint and
/// the same tokens as TRELLIS.2), split into the global tokens and the patch map, which <see cref="Pixal3DProjection"/> samples at each voxel's
/// projected position. The encoder's weights are resident on the backend only while it runs.
/// </summary>
public sealed class Pixal3DConditionerStage : IDisposable
{
    /// <summary>Pixels per DINOv3 patch edge.</summary>
    private const int PatchSize = 16;

    private readonly Trellis2ImageConditioner _conditioner;

    /// <param name="conditioner">This stage owns it.</param>
    public Pixal3DConditionerStage(Trellis2ImageConditioner conditioner)
        => _conditioner = conditioner ?? throw new ArgumentNullException(nameof(conditioner));

    /// <summary>The conditioning for <paramref name="image"/> encoded at <paramref name="imageSize"/> pixels: <paramref name="coords"/> are
    /// <c>(batch, x, y, z)</c> rows on a <paramref name="gridResolution"/> grid, or empty for the dense grid. The caller owns the result.</summary>
    public Pixal3DCondition Encode(
        IBackend backend, Trellis2PreparedImage image, int imageSize, int gridResolution, ReadOnlySpan<int> coords, Pixal3DView view)
    {
        ArgumentNullException.ThrowIfNull(backend);
        using Tensor tokens = EncodeTokens(backend, image, imageSize);
        return Split(tokens, imageSize, gridResolution, coords, view);
    }

    public void Dispose() => _conditioner.Dispose();

    /// <summary>Splits DINOv3 <paramref name="tokens"/> <c>[1, globals + patches², C]</c> into the global tokens and the cell-aligned samples
    /// of the <c>patches × patches</c> map; <paramref name="imageSize"/> is the encoded image's edge and fixes the patch count.</summary>
    internal static Pixal3DCondition Split(
        Tensor tokens, int imageSize, int gridResolution, ReadOnlySpan<int> coords, Pixal3DView view)
    {
        if (imageSize <= 0 || imageSize % PatchSize != 0)
            throw new ArgumentOutOfRangeException(nameof(imageSize), imageSize, $"The image size must be a positive multiple of {PatchSize}.");
        int patches = imageSize / PatchSize, patchCount = patches * patches;
        if (tokens.Shape.Rank != 3 || tokens.Shape[0] != 1 || tokens.Shape[1] <= patchCount)
            throw new ArgumentException($"{tokens.Shape} tokens leave no global tokens beside a {patches}x{patches} patch map.", nameof(tokens));
        int channels = (int)tokens.Shape[2], globalCount = (int)tokens.Shape[1] - patchCount;
        ReadOnlySpan<float> values = tokens.AsReadOnlySpan<float>();
        float[] uv = coords.IsEmpty
            ? Pixal3DProjection.DenseCoordinates(gridResolution, view, imageSize)
            : Pixal3DProjection.CellCoordinates(coords, gridResolution, view, imageSize);
        Tensor proj = Pixal3DProjection.Sample(values[(globalCount * channels)..], channels, patches, patches, uv);
        Tensor global = new(new TensorShape(1, globalCount, channels), DType.F32);
        values[..(globalCount * channels)].CopyTo(global.AsSpan<float>());
        return new Pixal3DCondition(global, proj);
    }

    private Tensor EncodeTokens(IBackend backend, Trellis2PreparedImage image, int imageSize)
    {
        backend.PreloadWeights(_conditioner.EnumerateWeights());
        try
        {
            return _conditioner.Encode(backend, image.Rgb, image.Width, image.Height, imageSize);
        }
        finally
        {
            backend.Sync();
            backend.FreeWeights(_conditioner.EnumerateWeights());
        }
    }
}
