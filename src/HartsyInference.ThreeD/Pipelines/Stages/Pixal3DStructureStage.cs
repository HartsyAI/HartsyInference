using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ThreeD.Models.Pixal3D;
using HartsyInference.ThreeD.Models.TrellisV2;

namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>
/// Pixal3D stage 1: the projection-mode structure flow samples an <c>[1, 8, 16, 16, 16]</c> latent from the global tokens and the per-cell
/// projected DINOv3 features, then <see cref="Trellis2StructureDecoder"/> (the same decoder checkpoint as TRELLIS.2) turns it into occupied
/// voxels on the shape flow's grid. The 1.3B flow is read from disk for each request and released afterwards.
/// </summary>
public sealed class Pixal3DStructureStage : IPixal3DStructureStage, IDisposable
{
    private readonly string _flowPath;
    private readonly Trellis2StructureDecoder _decoder;

    private Pixal3DStructureStage(string flowPath, Trellis2StructureDecoder decoder)
    {
        _flowPath = flowPath;
        _decoder = decoder;
    }

    /// <summary>Opens the decoder checkpoint now and records the flow checkpoint, which is loaded per request.</summary>
    public static Pixal3DStructureStage Load(string flowPath, string decoderPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(flowPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(decoderPath);
        if (!File.Exists(flowPath))
            throw new FileNotFoundException($"Pixal3D structure flow not found: '{flowPath}'.", flowPath);
        return new Pixal3DStructureStage(flowPath, Trellis2StructureDecoder.Load(decoderPath));
    }

    /// <inheritdoc/>
    public int[] Sample(
        IBackend backend,
        Pixal3DCondition cond,
        Pixal3DCondition negCond,
        int seed,
        Trellis2SamplerParams parameters,
        int outputResolution,
        Action<int>? onStep)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(cond);
        ArgumentNullException.ThrowIfNull(negCond);
        Pixal3DDeviceMemory.ReleaseAll(backend, cond, negCond);
        using Trellis2FlowWeights weights = Trellis2FlowWeights.Load(_flowPath);
        if (!weights.Config.HasProjection)
            throw new InvalidOperationException($"'{_flowPath}' is not a projection flow; Pixal3D's structure flow has image_attn_mode \"proj\".");
        using Trellis2SparseStructureFlow flow = new(weights);
        int r = flow.Config.Resolution;
        using Tensor noise = SeedGenerator.CreateNoise(new TensorShape([1L, flow.Config.InChannels, r, r, r]), seed);
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
                    return flow.Forward(backend, x, tModel, c.Global, c.Proj);
                },
                onStep);
        }
        finally
        {
            backend.Sync();
            backend.FreeWeights(flow.EnumerateWeights());
        }

        using (latent)
            return _decoder.Decode(backend, latent, outputResolution);
    }

    public void Dispose() => _decoder.Dispose();
}
