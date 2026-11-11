using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.Pixal3D;
using HartsyInference.ThreeD.Models.TrellisV2;

namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>
/// Pixal3D's image conditioning for one flow stage: DINOv3 tokens from the shared <see cref="Trellis2ImageConditioner"/> (the same checkpoint and
/// the same tokens as TRELLIS.2), split into the global tokens and the patch map, which <see cref="Pixal3DProjection"/> samples at each voxel's
/// projected position. With a <see cref="Pixal3DNaf"/> the sample is <c>[patch map ‖ NAF-upsampled map]</c>, as the shape and texture flows expect.
/// The encoders' weights are resident on the backend only while they run.
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
    /// <c>(batch, x, y, z)</c> rows on a <paramref name="gridResolution"/> grid, or empty for the dense grid. With <paramref name="naf"/> the
    /// features are also upsampled to <paramref name="nafSize"/> pixels first (the structure flow has no NAF). The caller owns the result.</summary>
    public Pixal3DCondition Encode(
        IBackend backend,
        Trellis2PreparedImage image,
        int imageSize,
        int gridResolution,
        ReadOnlySpan<int> coords,
        Pixal3DView view,
        Pixal3DNaf? naf = null,
        int nafSize = 0)
    {
        ArgumentNullException.ThrowIfNull(backend);
        using Tensor tokens = EncodeTokens(backend, image, imageSize);
        using Tensor? query = naf is null ? null : NafQueryMap(backend, naf, image, imageSize, nafSize);
        return Split(tokens, imageSize, gridResolution, coords, view, query, nafSize);
    }

    public void Dispose() => _conditioner.Dispose();

    /// <summary>Splits DINOv3 <paramref name="tokens"/> <c>[1, globals + patches², C]</c> into the global tokens and the cell-aligned samples
    /// of the <c>patches × patches</c> map; <paramref name="imageSize"/> is the encoded image's edge and fixes the patch count. A
    /// <paramref name="nafQuery"/> map <c>[1, 256, nafSize, nafSize]</c> appends the NAF-upsampled samples of the same patch map as extra channels.</summary>
    internal static Pixal3DCondition Split(
        Tensor tokens,
        int imageSize,
        int gridResolution,
        ReadOnlySpan<int> coords,
        Pixal3DView view,
        Tensor? nafQuery = null,
        int nafSize = 0)
    {
        int patches = imageSize / PatchSize, patchCount = patches * patches;
        if (tokens.Shape.Rank != 3 || tokens.Shape[0] != 1 || tokens.Shape[1] <= patchCount)
            throw new ArgumentException($"{tokens.Shape} tokens leave no global tokens beside a {patches}x{patches} patch map.", nameof(tokens));
        int channels = (int)tokens.Shape[2], globalCount = (int)tokens.Shape[1] - patchCount;
        ReadOnlySpan<float> values = tokens.AsReadOnlySpan<float>();
        float[] uv = coords.IsEmpty
            ? Pixal3DProjection.DenseCoordinates(gridResolution, view, imageSize)
            : Pixal3DProjection.CellCoordinates(coords, gridResolution, view, imageSize);
        ReadOnlySpan<float> patchRows = values[(globalCount * channels)..];
        Tensor proj = Pixal3DProjection.Sample(patchRows, channels, patches, patches, uv);
        if (nafQuery is not null)
        {
            using Tensor lowResolution = proj;
            using Tensor upsampled = Pixal3DNaf.SampleUpsampled(
                nafQuery.AsReadOnlySpan<float>(), nafSize, nafSize, patchRows, patches, patches, channels, uv);
            proj = JoinChannels(lowResolution, upsampled);
        }

        Tensor global = new(new TensorShape(1, globalCount, channels), DType.F32);
        values[..(globalCount * channels)].CopyTo(global.AsSpan<float>());
        return new Pixal3DCondition(global, proj);
    }

    /// <summary>Per-row concatenation <c>[a ‖ b]</c> of two <c>[1, rows, channels]</c> tensors, upstream's <c>cat([lowres, naf], -1)</c>.</summary>
    private static Tensor JoinChannels(Tensor a, Tensor b)
    {
        long rows = a.Shape[1], ca = a.Shape[2], cb = b.Shape[2];
        Tensor joined = new(new TensorShape(1, rows, ca + cb), DType.F32);
        ReadOnlySpan<float> left = a.AsReadOnlySpan<float>(), right = b.AsReadOnlySpan<float>();
        Span<float> target = joined.AsSpan<float>();
        for (long r = 0; r < rows; r++)
        {
            left.Slice((int)(r * ca), (int)ca).CopyTo(target.Slice((int)(r * (ca + cb)), (int)ca));
            right.Slice((int)(r * cb), (int)cb).CopyTo(target.Slice((int)(r * (ca + cb) + ca), (int)cb));
        }

        return joined;
    }

    private static Tensor NafQueryMap(IBackend backend, Pixal3DNaf naf, Trellis2PreparedImage image, int imageSize, int nafSize)
    {
        // The guide is the same Lanczos-resized image the encoder sees, before normalization.
        using Tensor guide = Trellis2ImageConditioner.Pixels(image.Rgb, image.Width, image.Height, imageSize, normalize: false);
        backend.PreloadWeights(naf.EnumerateWeights());
        try
        {
            return naf.QueryMap(backend, guide, nafSize);
        }
        finally
        {
            backend.Sync();
            backend.FreeWeights(naf.EnumerateWeights());
        }
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
