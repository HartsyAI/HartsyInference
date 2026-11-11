using HartsyInference.Core.Backends;
using HartsyInference.Core.Sparse;
using HartsyInference.ThreeD.Models.Pixal3D;
using HartsyInference.ThreeD.Models.TrellisV2;

namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>
/// Pixal3D stage 2 on the real checkpoints: the projection-mode 512 or 1024 shape flow samples a 32-channel latent per voxel from the global tokens
/// and the per-voxel projected features, which is then denormalized with the <c>pipeline.json</c> statistics (the same ones as TRELLIS.2). Each
/// 1.3B flow is read from disk for the call and released afterwards, so only one is in memory at a time.
/// </summary>
public sealed class Pixal3DShapeFlowStage : IPixal3DShapeFlowStage
{
    private readonly string _flow512Path, _flow1024Path;
    private readonly float[] _mean, _std;

    public Pixal3DShapeFlowStage(string flow512Path, string flow1024Path, Trellis2PipelineConfig config)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(flow512Path);
        ArgumentException.ThrowIfNullOrWhiteSpace(flow1024Path);
        ArgumentNullException.ThrowIfNull(config);
        _flow512Path = flow512Path;
        _flow1024Path = flow1024Path;
        _mean = config.ShapeLatentMean;
        _std = config.ShapeLatentStd;
    }

    /// <inheritdoc/>
    public SparseTensor Sample(
        IBackend backend,
        Trellis2ShapeFlowKind kind,
        int[] coords,
        int gridResolution,
        Pixal3DCondition cond,
        Pixal3DCondition negCond,
        int seed,
        Trellis2SamplerParams parameters,
        Action<int>? onStep)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(coords);
        ArgumentNullException.ThrowIfNull(cond);
        ArgumentNullException.ThrowIfNull(negCond);
        Pixal3DDeviceMemory.ReleaseAll(backend, cond, negCond);
        return Trellis2ShapeFlowStage.SampleFlow(
            backend,
            kind == Trellis2ShapeFlowKind.Res512 ? _flow512Path : _flow1024Path,
            coords,
            gridResolution,
            seed,
            parameters,
            onStep,
            _mean,
            _std,
            requireProjection: true,
            (flow, x, tModel, conditional) =>
            {
                Pixal3DCondition c = conditional ? cond : negCond;
                return flow.Forward(backend, x, coords, tModel, c.Global, c.Proj);
            });
    }
}
