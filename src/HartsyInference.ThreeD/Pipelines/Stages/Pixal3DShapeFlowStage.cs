using HartsyInference.Core.Backends;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Utilities;
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
        string path = kind == Trellis2ShapeFlowKind.Res512 ? _flow512Path : _flow1024Path;
        using Trellis2FlowWeights weights = Trellis2FlowWeights.Load(path);
        if (!weights.Config.HasProjection)
            throw new InvalidOperationException($"'{path}' is not a projection flow; Pixal3D's shape flows have image_attn_mode \"proj\".");
        using Trellis2SlatFlow flow = new(weights);
        int n = coords.Length / 4;
        using Tensor noise = SeedGenerator.CreateNoise(new TensorShape(1, n, flow.Config.InChannels), seed);
        Tensor latent;
        backend.PreloadWeights(flow.EnumerateWeights());
        try
        {
            latent = Trellis2FlowSampler.Sample(
                backend,
                noise,
                parameters,
                (x, tModel, conditional) =>
                {
                    Pixal3DCondition c = conditional ? cond : negCond;
                    return flow.Forward(backend, x, coords, tModel, c.Global, c.Proj);
                },
                onStep);
        }
        finally
        {
            backend.Sync();
            backend.FreeWeights(flow.EnumerateWeights());
        }

        Trellis2ShapeFlowStage.Denormalize(latent, _mean, _std);
        return new SparseTensor(latent, coords, gridResolution);
    }
}
