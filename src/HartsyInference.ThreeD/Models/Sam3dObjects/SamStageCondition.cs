using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Dinov2;

namespace HartsyInference.ThreeD.Models.Sam3dObjects;

/// <summary>Which generator a condition feeds.</summary>
public enum SamStage
{
    /// <summary>The structure flow: two DINO towers plus the pointmap embedder, <c>forward_features</c> tokens.</summary>
    Structure,

    /// <summary>The SLAT flow: the two DINO towers on the <c>prenorm_features</c> path, no pointmap.</summary>
    Slat,
}

/// <summary>
/// The condition of one generator (upstream <c>EmbedderFuser</c> with its embedders): per keyword one embedder call, then the fuser's projection and index embedding,
/// concatenated in embedder order. Structure: <c>image</c> and <c>rgb_image</c> through the first DINO, <c>mask</c> and <c>rgb_image_mask</c> through the second, <c>pointmap</c> and
/// <c>rgb_pointmap</c> through the point-patch embedder; groups <c>cropped</c> and <c>full</c>. SLAT: the first two only, on the pre-norm path. All weights are the checkpoint's
/// <c>condition_embedder.*</c> (<see cref="SamCheckpointSection.ConditionEmbedder"/>), borrowed and owned by the caller.
/// </summary>
public sealed class SamStageCondition : IDisposable
{
    private readonly SamStage _stage;
    private readonly SamDinoEmbedder _image;
    private readonly SamDinoEmbedder _mask;
    private readonly SamPointPatchEmbed? _pointmap;
    private readonly SamPointPatchConfig _patch;
    private readonly int _dinoWidth;
    private readonly SamEmbedderFuser _fuser;

    /// <param name="stage">The generator this condition feeds.</param>
    /// <param name="section">The condition embedder's tensors by name inside it: <c>module_list.{0,1}.backbone.*</c>, <c>module_list.2.*</c> (structure only), <c>idx_emb</c>, <c>projection_nets.*</c>.</param>
    public SamStageCondition(SamStage stage, IReadOnlyDictionary<string, Tensor> section)
        : this(stage, section, Dinov2Preset.LargeReg, SamPointPatchConfig.Released)
    {
    }

    /// <summary>The same condition at other sizes, for tests on a tiny network; the released one is <see cref="Dinov2Preset.LargeReg"/> with <see cref="SamPointPatchConfig.Released"/>.</summary>
    internal SamStageCondition(SamStage stage, IReadOnlyDictionary<string, Tensor> section, Dinov2Preset dino, SamPointPatchConfig patch)
    {
        ArgumentNullException.ThrowIfNull(section);
        ArgumentNullException.ThrowIfNull(dino);
        ArgumentNullException.ThrowIfNull(patch);
        _stage = stage;
        _patch = patch;
        _dinoWidth = dino.HiddenSize;
        SamDinoEmbedder? image = null, mask = null;
        try
        {
            image = new SamDinoEmbedder(dino, section, "module_list.0.backbone.");
            mask = new SamDinoEmbedder(dino, section, "module_list.1.backbone.");
            List<SamFuserEmbedder> embedders =
            [
                new SamFuserEmbedder(_dinoWidth, [("image", "cropped"), ("rgb_image", "full")]),
                new SamFuserEmbedder(_dinoWidth, [("mask", "cropped"), ("rgb_image_mask", "full")]),
            ];
            if (stage == SamStage.Structure)
            {
                Dictionary<string, Tensor> pointmap = new(StringComparer.Ordinal);
                const string prefix = "module_list.2.";
                foreach (KeyValuePair<string, Tensor> entry in section)
                {
                    if (entry.Key.StartsWith(prefix, StringComparison.Ordinal))
                        pointmap[entry.Key[prefix.Length..]] = entry.Value;
                }

                _pointmap = new SamPointPatchEmbed(patch, pointmap);
                embedders.Add(new SamFuserEmbedder(patch.EmbedDim, [("pointmap", "cropped"), ("rgb_pointmap", "full")]));
            }

            _fuser = new SamEmbedderFuser(embedders, _dinoWidth, section);
            _image = image;
            _mask = mask;
        }
        catch
        {
            image?.Dispose();
            mask?.Dispose();
            throw;
        }
    }

    /// <summary>The fused condition tokens, F32 <c>[1, n, 1024]</c> as the flows take them, owned by the caller: 4 x 1370 plus 2 x 1024 rows for the structure stage, 4 x 1374 for SLAT.</summary>
    /// <param name="inputs">The preprocessed images, masks and (structure stage) pointmaps.</param>
    /// <param name="forceDrop">Keywords whose tokens are zeroed after the index embedding (the structure model's dropout of <c>pointmap</c> and <c>rgb_pointmap</c> in object-only mode); null drops nothing.</param>
    public Tensor Build(IBackend backend, SamConditionInputs inputs, IReadOnlySet<string>? forceDrop = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(inputs);
        Dictionary<string, Tensor> tokens = new(StringComparer.Ordinal);
        List<Tensor> owners = [];
        try
        {
            if (_stage == SamStage.Structure)
            {
                tokens["image"] = Rows(owners, _image.StructureTokens(backend, inputs.CroppedImage));
                tokens["rgb_image"] = Rows(owners, _image.StructureTokens(backend, inputs.FullImage));
                tokens["mask"] = Rows(owners, _mask.StructureTokensOfMask(backend, inputs.CroppedMask));
                tokens["rgb_image_mask"] = Rows(owners, _mask.StructureTokensOfMask(backend, inputs.FullMask));
                tokens["pointmap"] = PointmapTokens(backend, "pointmap", inputs.CroppedPointmap, forceDrop);
                tokens["rgb_pointmap"] = PointmapTokens(backend, "rgb_pointmap", inputs.FullPointmap, forceDrop);
            }
            else
            {
                tokens["image"] = Rows(owners, _image.SlatTokens(backend, inputs.CroppedImage));
                tokens["rgb_image"] = Rows(owners, _image.SlatTokens(backend, inputs.FullImage));
                tokens["mask"] = Rows(owners, _mask.SlatTokensOfMask(backend, inputs.CroppedMask));
                tokens["rgb_image_mask"] = Rows(owners, _mask.SlatTokensOfMask(backend, inputs.FullMask));
            }

            using Tensor fused = _fuser.Fuse(backend, tokens, forceDrop);
            Tensor batched = new(new TensorShape(1, fused.Shape[0], fused.Shape[1]), DType.F32);
            fused.AsReadOnlySpan<float>().CopyTo(batched.AsSpan<float>());
            return batched;
        }
        finally
        {
            foreach (Tensor token in tokens.Values)
                token.Dispose();
            foreach (Tensor owner in owners)
                owner.Dispose();
        }
    }

    /// <summary>The point-patch tokens of one pointmap; a force-dropped keyword gets zeros without the pointmap being read (the fuser zeroes them after projecting anyway), so object-only runs need no pointmap.</summary>
    private Tensor PointmapTokens(IBackend backend, string keyword, SamPointmapInput? pointmap, IReadOnlySet<string>? forceDrop)
    {
        if (forceDrop is not null && forceDrop.Contains(keyword))
            return new Tensor(new TensorShape(_patch.Windows, _patch.EmbedDim), DType.F32);
        SamPointmapInput input = pointmap ?? throw new ArgumentException($"The structure stage needs the pointmap for '{keyword}' unless it is dropped.", nameof(pointmap));
        return _pointmap!.Embed(backend, input.Planar, input.Height, input.Width);
    }

    /// <summary>A <c>[1, n, d]</c> tensor as a <c>[n, d]</c> view (the fuser takes rows); the input is added to <paramref name="owners"/>, which the caller disposes after the views.</summary>
    private static Tensor Rows(List<Tensor> owners, Tensor batched)
    {
        owners.Add(batched);
        return batched.Reshape(new TensorShape(batched.Shape[1], batched.Shape[2]));
    }

    public void Dispose()
    {
        _image.Dispose();
        _mask.Dispose();
    }
}
