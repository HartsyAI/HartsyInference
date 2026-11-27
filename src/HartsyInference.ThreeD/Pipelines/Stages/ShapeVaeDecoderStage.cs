using HartsyInference.Core.Backends;
using HartsyInference.Core.Sparse;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Models.TrellisV2;

namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>Shape-VAE latent to dual-grid field: runs <see cref="ShapeVaeDecoder"/> and splits its head. It also serves TRELLIS.2's
/// cascade, which asks the decoder only where its first four levels put voxels.</summary>
public sealed class ShapeVaeDecoderStage : ISparseShapeSubdivisionDecoder, ISparseShapeUpsampler, IDisposable
{
    private readonly ShapeVaeDecoder _decoder;
    private bool _disposed;

    /// <param name="decoder">Built from the shape VAE decoder weights; this stage owns it.</param>
    public ShapeVaeDecoderStage(ShapeVaeDecoder decoder) => _decoder = decoder ?? throw new ArgumentNullException(nameof(decoder));

    /// <summary>Opens the TRELLIS.2 / Pixal3D shape VAE decoder checkpoint (<c>shape_dec_next_dc_f16c32_fp16</c>). The returned stage owns the decoder;
    /// <paramref name="loaders"/> receives the safetensors mapping its weights borrow from, which must outlive the stage.</summary>
    public static ShapeVaeDecoderStage LoadNextDcF16C32(string path, ICollection<IDisposable> loaders)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(loaders);
        SafeTensorsLoader loader = new();
        loaders.Add(loader);
        loader.Load(path);
        return new ShapeVaeDecoderStage(new ShapeVaeDecoder(ShapeVaeConfig.NextDcF16C32.Decoder, loader.GetAllTensors()));
    }

    /// <inheritdoc/>
    public DualGridField Decode(IBackend backend, SparseTensor latent) => DecodeCore(backend, latent, null);

    /// <inheritdoc/>
    public DualGridField Decode(IBackend backend, SparseTensor latent, ShapeVaeSubdivisions record)
    {
        ArgumentNullException.ThrowIfNull(record);
        return DecodeCore(backend, latent, record);
    }

    private DualGridField DecodeCore(IBackend backend, SparseTensor latent, ShapeVaeSubdivisions? record)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(latent);
        SparseTensor head = _decoder.Decode(backend, latent, record);
        try
        {
            return ShapeVaeHead.Split(head);
        }
        finally
        {
            head.Feats.Dispose();
        }
    }

    /// <inheritdoc/>
    public int[] Upsample(IBackend backend, SparseTensor latent, int times) => _decoder.Upsample(backend, latent, times);

    /// <summary>Releases the decoder and its weights.</summary>
    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        _decoder.Dispose();
    }
}
