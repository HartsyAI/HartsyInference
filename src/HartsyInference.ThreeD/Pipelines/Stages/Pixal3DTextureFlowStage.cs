using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Sparse;
using HartsyInference.ThreeD.Models.Pixal3D;
using HartsyInference.ThreeD.Models.TrellisV2;

namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>
/// Pixal3D texture sampling on the real checkpoint: the shape latent is normalized with the shape statistics and concatenated to the noise at every
/// step, the projection-mode 1024 texture flow samples the 32-channel texture latent from the global tokens and the per-voxel projected features, and
/// that is denormalized with the texture statistics (both from <c>pipeline.json</c>, the same as TRELLIS.2's). The 1.3B flow is read from disk for the
/// call and released afterwards. Pixal3D has no 512 texture flow.
/// </summary>
/// <remarks>Before it starts, the live conditioning and the shape latent are offloaded to the host and <see cref="Trellis2DeviceMemory.ReleaseAll"/>
/// sweeps the device, as the TRELLIS.2 texture flow does.</remarks>
public sealed class Pixal3DTextureFlowStage : IPixal3DTextureFlowStage
{
    private readonly string _flowPath;
    private readonly float[] _shapeMean, _shapeStd, _textureMean, _textureStd;

    /// <param name="flowPath">The 1024 projection texture flow.</param>
    /// <param name="config">A <c>pipeline.json</c> with the texture sampler and normalization.</param>
    public Pixal3DTextureFlowStage(string flowPath, Trellis2PipelineConfig config)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(flowPath);
        ArgumentNullException.ThrowIfNull(config);
        if (!config.HasTexture)
            throw new HartsyInferenceException("pipeline.json has no tex_slat_sampler and tex_slat_normalization, so the texture path cannot run.");
        _flowPath = flowPath;
        _shapeMean = config.ShapeLatentMean;
        _shapeStd = config.ShapeLatentStd;
        _textureMean = config.TextureLatentMean!;
        _textureStd = config.TextureLatentStd!;
    }

    /// <inheritdoc/>
    public SparseTensor Sample(
        IBackend backend,
        SparseTensor shape,
        Pixal3DCondition cond,
        Pixal3DCondition negCond,
        int seed,
        Trellis2SamplerParams parameters,
        Action<int>? onStep)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(shape);
        ArgumentNullException.ThrowIfNull(cond);
        ArgumentNullException.ThrowIfNull(negCond);
        return Trellis2TextureFlowStage.SampleFlow(
            backend,
            _flowPath,
            shape,
            seed,
            parameters,
            onStep,
            _shapeMean,
            _shapeStd,
            _textureMean,
            _textureStd,
            requireProjection: true,
            [cond.Global, cond.Proj, negCond.Global, negCond.Proj],
            (flow, x, shapeNormalized, tModel, conditional) =>
            {
                Pixal3DCondition c = conditional ? cond : negCond;
                return flow.Forward(backend, x, shapeNormalized, shape.Coords, tModel, c.Global, c.Proj);
            });
    }
}
