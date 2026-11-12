using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ThreeD.Models.TrellisV2;

namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>
/// TRELLIS.2 texture sampling on the real checkpoints: the shape latent is normalized with the shape statistics and concatenated to the
/// noise at every step, the 512 or 1024 texture flow samples the 32-channel texture latent, and that is denormalized with the texture
/// statistics. Each 1.3B flow is read from disk for the call and released afterwards.
/// </summary>
/// <remarks>Before it starts, a call drops every cached device weight on the backend (<see cref="Trellis2DeviceMemory.ReleaseAll"/>), the shape
/// decoder's included: left in place they slowed this flow 20 times on a 12 GB card. Weights of an outside caller on the same backend upload
/// again on their next use.</remarks>
public sealed class Trellis2TextureFlowStage : ITrellis2TextureFlowStage
{
    private readonly string? _flow512Path, _flow1024Path;
    private readonly float[] _shapeMean, _shapeStd, _textureMean, _textureStd;

    /// <param name="flow512Path">The 512 texture flow, or null when it is not installed.</param>
    /// <param name="flow1024Path">The 1024 texture flow, or null when it is not installed.</param>
    /// <param name="config">A <c>pipeline.json</c> with the texture sampler and normalization.</param>
    public Trellis2TextureFlowStage(string? flow512Path, string? flow1024Path, Trellis2PipelineConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (!config.HasTexture)
            throw new HartsyInferenceException("pipeline.json has no tex_slat_sampler and tex_slat_normalization, so the texture path cannot run.");
        _flow512Path = flow512Path;
        _flow1024Path = flow1024Path;
        _shapeMean = config.ShapeLatentMean;
        _shapeStd = config.ShapeLatentStd;
        _textureMean = config.TextureLatentMean!;
        _textureStd = config.TextureLatentStd!;
    }

    /// <inheritdoc/>
    public bool Supports(Trellis2ShapeFlowKind kind) => (kind == Trellis2ShapeFlowKind.Res512 ? _flow512Path : _flow1024Path) is not null;

    /// <inheritdoc/>
    public SparseTensor Sample(
        IBackend backend,
        Trellis2ShapeFlowKind kind,
        SparseTensor shape,
        Tensor cond,
        Tensor negCond,
        int seed,
        Trellis2SamplerParams parameters,
        Action<int>? onStep)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(shape);
        string path = (kind == Trellis2ShapeFlowKind.Res512 ? _flow512Path : _flow1024Path)
            ?? throw new HartsyInferenceException(
                $"The {(kind == Trellis2ShapeFlowKind.Res512 ? "512" : "1024")} texture flow checkpoint is not installed.");
        return SampleFlow(
            backend,
            path,
            shape,
            seed,
            parameters,
            onStep,
            _shapeMean,
            _shapeStd,
            _textureMean,
            _textureStd,
            requireProjection: false,
            [cond, negCond],
            (flow, x, shapeNormalized, tModel, conditional) => flow.Forward(backend, x, shapeNormalized, shape.Coords, tModel, conditional ? cond : negCond));
    }

    /// <summary>The sequence both TRELLIS.2 and Pixal3D texture stages run: sweep the device (keeping <paramref name="live"/> and the shape latent),
    /// load the flow, sample over the voxels of <paramref name="shape"/> with the normalized shape latent concatenated at every step, release the
    /// flow, denormalize. <paramref name="requireProjection"/> says which kind of flow the stage expects, so a checkpoint of the other model fails
    /// before any step runs.</summary>
    internal static SparseTensor SampleFlow(
        IBackend backend,
        string path,
        SparseTensor shape,
        int seed,
        Trellis2SamplerParams parameters,
        Action<int>? onStep,
        ReadOnlySpan<float> shapeMean,
        ReadOnlySpan<float> shapeStd,
        ReadOnlySpan<float> textureMean,
        ReadOnlySpan<float> textureStd,
        bool requireProjection,
        Tensor[] live,
        Func<Trellis2TextureFlow, Tensor, Tensor, float, bool, Tensor> forward)
    {
        int n = shape.Count;
        if (shape.Feats.Shape.ElementCount != (long)n * shapeMean.Length)
            throw new ArgumentException($"The shape latent must hold {shapeMean.Length} channels per voxel; got {shape.Feats.Shape}.", nameof(shape));

        Trellis2DeviceMemory.ReleaseAll(backend, [.. live, shape.Feats]);
        using Trellis2FlowWeights weights = Trellis2FlowWeights.Load(path);
        if (weights.Config.HasProjection != requireProjection)
        {
            throw new HartsyInferenceException(
                $"'{path}' {(weights.Config.HasProjection ? "is" : "is not")} a projection flow; this stage needs a "
                + $"{(requireProjection ? "Pixal3D projection" : "TRELLIS.2 plain")} texture flow.");
        }

        using Trellis2TextureFlow flow = new(weights);
        if (flow.ConcatChannels != shapeMean.Length || flow.LatentChannels != textureMean.Length)
        {
            throw new HartsyInferenceException(
                $"'{path}' takes {flow.ConcatChannels} shape and {flow.LatentChannels} texture channels; expected "
                + $"{shapeMean.Length} and {textureMean.Length}.");
        }

        using Tensor shapeNormalized = new(new TensorShape(1, n, shapeMean.Length), DType.F32);
        shape.Feats.AsReadOnlySpan<float>().CopyTo(shapeNormalized.AsSpan<float>());
        Trellis2LatentNormalization.Normalize(shapeNormalized, shapeMean, shapeStd);

        using Tensor noise = SeedGenerator.CreateNoise(new TensorShape(1, n, flow.LatentChannels), seed);
        Tensor latent;
        backend.PreloadWeights(flow.EnumerateWeights());
        try
        {
            latent = Trellis2FlowSampler.Sample(
                backend,
                noise,
                parameters,
                (x, tModel, conditional) => forward(flow, x, shapeNormalized, tModel, conditional),
                onStep);
        }
        finally
        {
            backend.Sync();
            backend.FreeWeights(flow.EnumerateWeights());
        }

        Trellis2LatentNormalization.Denormalize(latent, textureMean, textureStd);
        return new SparseTensor(latent, shape.Coords, shape.Resolution);
    }
}
