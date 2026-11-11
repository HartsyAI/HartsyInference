using HartsyInference.Core.Backends;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ThreeD.Models.TrellisV2;

namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>
/// TRELLIS.2 stage 2 on the real checkpoints: the 512 or 1024 shape flow samples a 32-channel latent per voxel, which is then
/// denormalized with the <c>pipeline.json</c> statistics. Each 1.3B flow is read from disk for the call and released afterwards,
/// so only one is in memory at a time.
/// </summary>
public sealed class Trellis2ShapeFlowStage : ITrellis2ShapeFlowStage
{
    private readonly string _flow512Path, _flow1024Path;
    private readonly float[] _mean, _std;

    public Trellis2ShapeFlowStage(string flow512Path, string flow1024Path, Trellis2PipelineConfig config)
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
        Tensor cond,
        Tensor negCond,
        int seed,
        Trellis2SamplerParams parameters,
        Action<int>? onStep)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(coords);
        return SampleFlow(
            backend,
            kind == Trellis2ShapeFlowKind.Res512 ? _flow512Path : _flow1024Path,
            coords,
            gridResolution,
            seed,
            parameters,
            onStep,
            _mean,
            _std,
            requireProjection: false,
            (flow, x, tModel, conditional) => flow.Forward(backend, x, coords, tModel, conditional ? cond : negCond));
    }

    /// <summary>The sequence both TRELLIS.2 and Pixal3D shape stages run: load the flow, sample a latent over <paramref name="coords"/> with
    /// <paramref name="forward"/> as the model, release the flow, denormalize. <paramref name="requireProjection"/> says which kind of flow the stage
    /// expects, so a checkpoint of the other model fails before any step runs. The latent is released if a step throws.</summary>
    internal static SparseTensor SampleFlow(
        IBackend backend,
        string flowPath,
        int[] coords,
        int gridResolution,
        int seed,
        Trellis2SamplerParams parameters,
        Action<int>? onStep,
        ReadOnlySpan<float> mean,
        ReadOnlySpan<float> std,
        bool requireProjection,
        Func<Trellis2SlatFlow, Tensor, float, bool, Tensor> forward)
    {
        Trellis2DeviceMemory.ReleaseBetweenStages(backend);
        using Trellis2FlowWeights weights = Trellis2FlowWeights.Load(flowPath);
        if (weights.Config.HasProjection != requireProjection)
        {
            throw new InvalidOperationException(
                $"'{flowPath}' {(weights.Config.HasProjection ? "is" : "is not")} a projection flow; this stage needs a "
                + $"{(requireProjection ? "Pixal3D projection" : "TRELLIS.2 plain")} flow.");
        }

        using Trellis2SlatFlow flow = new(weights);
        int n = coords.Length / 4;
        using Tensor noise = SeedGenerator.CreateNoise(new TensorShape(1, n, flow.Config.InChannels), seed);
        Tensor latent;
        backend.PreloadWeights(flow.EnumerateWeights());
        try
        {
            latent = Trellis2FlowSampler.Sample(backend, noise, parameters, (x, tModel, conditional) => forward(flow, x, tModel, conditional), onStep);
        }
        finally
        {
            backend.Sync();
            backend.FreeWeights(flow.EnumerateWeights());
        }

        try
        {
            Trellis2LatentNormalization.Denormalize(latent, mean, std);
            return new SparseTensor(latent, coords, gridResolution);
        }
        catch
        {
            latent.Dispose();
            throw;
        }
    }
}
