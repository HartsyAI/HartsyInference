using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Segmentation.Sam3;

/// <summary>SAM 3 vision backbone: RoPE ViT trunk followed by the FPN neck, for the text-prompted image path.</summary>
/// <remarks>Input is a normalized image; preprocessing belongs to the pipeline. Runs in F32.</remarks>
public sealed class Sam3VisionBackbone : IDisposable
{
    private readonly Sam3VitTrunk _trunk;
    private readonly Sam3FpnNeck _neck;
    private readonly Sam3Rope _rope;

    /// <summary>Validates the configuration and loads trunk and neck weights; throws naming any missing key.</summary>
    /// <param name="config">SAM 3 hyperparameters.</param>
    /// <param name="trunkWeights">Keys relative to <c>detector.backbone.vision_backbone.trunk.</c>.</param>
    /// <param name="neckWeights">Keys relative to <c>detector.backbone.vision_backbone.</c>.</param>
    public Sam3VisionBackbone(
        Sam3Config config,
        IReadOnlyDictionary<string, Tensor> trunkWeights,
        IReadOnlyDictionary<string, Tensor> neckWeights)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(trunkWeights);
        ArgumentNullException.ThrowIfNull(neckWeights);
        ValidateConfig(config);

        _trunk = new Sam3VitTrunk(config, trunkWeights);
        _neck = new Sam3FpnNeck(config, neckWeights);
        _rope = new Sam3Rope(config.PatchGrid, config.VitWindowSize, config.VitEmbedDim / config.VitNumHeads, config.RopeTheta);
    }

    /// <summary>Runs the trunk and neck on a normalized image.</summary>
    /// <param name="backend">Backend that runs the ops.</param>
    /// <param name="image">F32 <c>[1, 3, ImageSize, ImageSize]</c> in [-1, 1].</param>
    /// <returns>Owned features and positions; the caller disposes them.</returns>
    public Sam3VisionFeatures Forward(IBackend backend, Tensor image)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(image);

        using Tensor trunkFeatures = _trunk.Forward(backend, image, _rope);
        return _neck.Forward(backend, trunkFeatures);
    }

    /// <summary>Releases the RoPE tables. Weights are owned by the caller and are not released.</summary>
    public void Dispose()
    {
        _rope.Dispose();
    }

    private static void ValidateConfig(Sam3Config config)
    {
        if (config.PatchSize <= 0 || config.ImageSize <= 0 || config.ImageSize % config.PatchSize != 0)
            throw new ArgumentException($"ImageSize {config.ImageSize} must be a positive multiple of PatchSize {config.PatchSize}.", nameof(config));
        if (config.VitWindowSize <= 0 || config.PatchGrid % config.VitWindowSize != 0)
            throw new ArgumentException($"Patch grid {config.PatchGrid} must be a multiple of the window {config.VitWindowSize}; padded windows are not implemented.", nameof(config));
        if (config.VitNumHeads <= 0 || config.VitEmbedDim % config.VitNumHeads != 0
            || (config.VitEmbedDim / config.VitNumHeads) % 4 != 0)
        {
            throw new ArgumentException($"ViT head dim must be a positive multiple of 4 (dim {config.VitEmbedDim}, heads {config.VitNumHeads}).", nameof(config));
        }
        if (config.VitDepth <= 0 || config.VitMlpHidden <= 0 || config.VitPretrainGrid <= 0)
            throw new ArgumentException("ViT depth, MLP width and pretrain grid must be positive.", nameof(config));
        foreach (int block in config.VitGlobalAttentionBlocks)
        {
            if (block < 0 || block >= config.VitDepth)
                throw new ArgumentException($"Global attention block {block} is outside [0, {config.VitDepth}).", nameof(config));
        }
        if (config.DModel <= 0 || config.DModel % 4 != 0)
            throw new ArgumentException($"DModel {config.DModel} must be a positive multiple of 4 for the sine encoding.", nameof(config));
        if (config.NeckScaleFactors.Length == 0 || config.ScalpLevels < 0 || config.ScalpLevels >= config.NeckScaleFactors.Length)
            throw new ArgumentException("ScalpLevels must leave at least one neck scale.", nameof(config));
    }
}
