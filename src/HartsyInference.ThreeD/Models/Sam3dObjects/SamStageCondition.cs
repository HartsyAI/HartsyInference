using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

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
    private readonly SamEmbedderFuser _fuser;

    /// <param name="stage">The generator this condition feeds.</param>
    /// <param name="section">The condition embedder's tensors by name inside it: <c>module_list.{0,1}.backbone.*</c>, <c>module_list.2.*</c> (structure only), <c>idx_emb</c>, <c>projection_nets.*</c>.</param>
    public SamStageCondition(SamStage stage, IReadOnlyDictionary<string, Tensor> section)
    {
        ArgumentNullException.ThrowIfNull(section);
        _stage = stage;
        SamDinoEmbedder? image = null, mask = null;
        try
        {
            image = new SamDinoEmbedder(section, "module_list.0.backbone.");
            mask = new SamDinoEmbedder(section, "module_list.1.backbone.");
            List<SamFuserEmbedder> embedders =
            [
                new SamFuserEmbedder(SamDinoEmbedder.Width, [("image", "cropped"), ("rgb_image", "full")]),
                new SamFuserEmbedder(SamDinoEmbedder.Width, [("mask", "cropped"), ("rgb_image_mask", "full")]),
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

                _pointmap = new SamPointPatchEmbed(SamPointPatchConfig.Released, pointmap);
                embedders.Add(new SamFuserEmbedder(SamPointPatchConfig.Released.EmbedDim, [("pointmap", "cropped"), ("rgb_pointmap", "full")]));
            }

            _fuser = new SamEmbedderFuser(embedders, SamDinoEmbedder.Width, section);
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
        try
        {
            if (_stage == SamStage.Structure)
            {
                SamPointmapInput cropped = inputs.CroppedPointmap ?? throw new ArgumentException("The structure stage needs the cropped pointmap.", nameof(inputs));
                SamPointmapInput full = inputs.FullPointmap ?? throw new ArgumentException("The structure stage needs the full pointmap.", nameof(inputs));
                tokens["image"] = Rows(_image.StructureTokens(backend, inputs.CroppedImage));
                tokens["rgb_image"] = Rows(_image.StructureTokens(backend, inputs.FullImage));
                tokens["mask"] = Rows(_mask.StructureTokensOfMask(backend, inputs.CroppedMask));
                tokens["rgb_image_mask"] = Rows(_mask.StructureTokensOfMask(backend, inputs.FullMask));
                tokens["pointmap"] = _pointmap!.Embed(backend, cropped.Planar, cropped.Height, cropped.Width);
                tokens["rgb_pointmap"] = _pointmap.Embed(backend, full.Planar, full.Height, full.Width);
            }
            else
            {
                tokens["image"] = Rows(_image.SlatTokens(backend, inputs.CroppedImage));
                tokens["rgb_image"] = Rows(_image.SlatTokens(backend, inputs.FullImage));
                tokens["mask"] = Rows(_mask.SlatTokensOfMask(backend, inputs.CroppedMask));
                tokens["rgb_image_mask"] = Rows(_mask.SlatTokensOfMask(backend, inputs.FullMask));
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
        }
    }

    /// <summary>A <c>[1, n, d]</c> tensor as an independent <c>[n, d]</c> one (the fuser takes rows); the input is disposed.</summary>
    private static Tensor Rows(Tensor batched)
    {
        Tensor rows = new(new TensorShape(batched.Shape[1], batched.Shape[2]), DType.F32);
        batched.AsReadOnlySpan<float>().CopyTo(rows.AsSpan<float>());
        batched.Dispose();
        return rows;
    }

    public void Dispose()
    {
        _image.Dispose();
        _mask.Dispose();
    }
}
