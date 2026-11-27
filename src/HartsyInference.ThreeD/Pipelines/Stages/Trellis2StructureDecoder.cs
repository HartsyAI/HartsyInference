using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Models.Trellis;
using HartsyInference.ThreeD.Models.TrellisV2;

namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>The structure latent's way to voxels, shared by the TRELLIS.2 and Pixal3D stage 1: the v1 <see cref="SparseStructureDecoder"/>
/// (the <c>ss_dec_conv3d_16l8</c> checkpoint) turns an <c>[1, 8, 16, 16, 16]</c> latent into a 64³ occupancy grid, which is pooled to the
/// shape flow's grid. The small decoder stays resident.</summary>
internal sealed class Trellis2StructureDecoder : IDisposable
{
    private const int DecodedResolution = 64;

    private readonly SparseStructureDecoder _decoder;
    private readonly SafeTensorsLoader _loader;

    private Trellis2StructureDecoder(SparseStructureDecoder decoder, SafeTensorsLoader loader)
    {
        _decoder = decoder;
        _loader = loader;
    }

    /// <summary>Opens the decoder checkpoint.</summary>
    internal static Trellis2StructureDecoder Load(string decoderPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(decoderPath);
        SafeTensorsLoader loader = new();
        try
        {
            loader.Load(decoderPath);
            SparseStructureDecoder decoder = new();
            decoder.LoadWeights(loader.GetAllTensors());
            return new Trellis2StructureDecoder(decoder, loader);
        }
        catch
        {
            loader.Dispose();
            throw;
        }
    }

    /// <summary>Voxels <c>(0, x, y, z)</c> on an <paramref name="outputResolution"/> grid for <paramref name="latent"/>; the caller keeps the latent.</summary>
    /// <exception cref="HartsyInferenceException">The decoder returns no 64³ grid, or no voxel is occupied.</exception>
    internal int[] Decode(IBackend backend, Tensor latent, int outputResolution)
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

    public void Dispose() => _loader.Dispose();
}
