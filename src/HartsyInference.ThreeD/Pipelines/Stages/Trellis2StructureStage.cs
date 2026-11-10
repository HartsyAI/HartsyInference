using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Models.Trellis;
using HartsyInference.ThreeD.Models.TrellisV2;

namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>
/// TRELLIS.2 stage 1: the dense flow samples an <c>[1, 8, 16, 16, 16]</c> latent, the v1 <see cref="SparseStructureDecoder"/> (the
/// same <c>ss_dec_conv3d_16l8</c> checkpoint) turns it into a 64³ occupancy grid, and the occupied cells are pooled to the shape
/// flow's grid. The 1.3B flow is read from disk for each request and released afterwards; the small decoder stays resident.
/// </summary>
public sealed class Trellis2StructureStage : ITrellis2StructureStage, IDisposable
{
    private const int DecodedResolution = 64;

    private readonly string _flowPath;
    private readonly SparseStructureDecoder _decoder;
    private readonly SafeTensorsLoader _decoderLoader;

    private Trellis2StructureStage(string flowPath, SparseStructureDecoder decoder, SafeTensorsLoader decoderLoader)
    {
        _flowPath = flowPath;
        _decoder = decoder;
        _decoderLoader = decoderLoader;
    }

    /// <summary>Opens the decoder checkpoint now and records the flow checkpoint, which is loaded per request.</summary>
    public static Trellis2StructureStage Load(string flowPath, string decoderPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(flowPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(decoderPath);
        if (!File.Exists(flowPath))
            throw new FileNotFoundException($"TRELLIS.2 structure flow not found: '{flowPath}'.", flowPath);
        SafeTensorsLoader loader = new();
        try
        {
            loader.Load(decoderPath);
            SparseStructureDecoder decoder = new();
            decoder.LoadWeights(loader.GetAllTensors());
            return new Trellis2StructureStage(flowPath, decoder, loader);
        }
        catch
        {
            loader.Dispose();
            throw;
        }
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
        {
            backend.PreloadWeights(_decoder.EnumerateWeights());
            Tensor occupancy;
            try
            {
                occupancy = _decoder.Decode(backend, latent);
            }
            finally
            {
                backend.Sync();
                backend.FreeWeights(_decoder.EnumerateWeights());
            }

            using (occupancy)
            {
                if (occupancy.Shape.ElementCount != (long)DecodedResolution * DecodedResolution * DecodedResolution)
                    throw new HartsyInferenceException($"The structure decoder returned {occupancy.Shape}; expected a {DecodedResolution}^3 grid.");
                int[] coords = Trellis2Cascade.OccupiedVoxels(occupancy.AsReadOnlySpan<float>(), DecodedResolution, outputResolution);
                if (coords.Length == 0)
                    throw new HartsyInferenceException("The sampled structure is empty: no voxel is occupied. Try another seed.");
                return coords;
            }
        }
    }

    public void Dispose() => _decoderLoader.Dispose();
}
