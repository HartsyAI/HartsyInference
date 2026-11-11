using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ThreeD.Models.TrellisV2;

namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>
/// TRELLIS.2 stage 1: the dense flow samples an <c>[1, 8, 16, 16, 16]</c> latent, <see cref="Trellis2StructureDecoder"/> turns it into
/// occupied voxels on the shape flow's grid. The 1.3B flow is read from disk for each request and released afterwards.
/// </summary>
public sealed class Trellis2StructureStage : ITrellis2StructureStage, IDisposable
{
    private readonly string _flowPath;
    private readonly Trellis2StructureDecoder _decoder;

    private Trellis2StructureStage(string flowPath, Trellis2StructureDecoder decoder)
    {
        _flowPath = flowPath;
        _decoder = decoder;
    }

    /// <summary>Opens the decoder checkpoint now and records the flow checkpoint, which is loaded per request.</summary>
    public static Trellis2StructureStage Load(string flowPath, string decoderPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(flowPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(decoderPath);
        if (!File.Exists(flowPath))
            throw new FileNotFoundException($"TRELLIS.2 structure flow not found: '{flowPath}'.", flowPath);
        return new Trellis2StructureStage(flowPath, Trellis2StructureDecoder.Load(decoderPath));
    }

    /// <inheritdoc/>
    public int[] Sample(
        IBackend backend, Tensor cond, Tensor negCond, int seed, Trellis2SamplerParams parameters, int outputResolution, Action<int>? onStep)
    {
        ArgumentNullException.ThrowIfNull(backend);
        using Trellis2FlowWeights weights = Trellis2FlowWeights.Load(_flowPath);
        using Trellis2SparseStructureFlow flow = new(weights);
        int r = flow.Config.Resolution;
        using Tensor noise = SeedGenerator.CreateNoise(new TensorShape([1L, flow.Config.InChannels, r, r, r]), seed);
        Tensor latent;
        backend.PreloadWeights(flow.EnumerateWeights());
        try
        {
            latent = Trellis2FlowSampler.Sample(
                backend, noise, parameters, (x, tModel, conditional) => flow.Forward(backend, x, tModel, conditional ? cond : negCond), onStep);
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
