using SharpInference.Core.Backends;
using SharpInference.Core.Pipelines;
using SharpInference.Core.Tensors;
using SharpInference.Vision.Clip;

namespace SharpInference.Vision.Embeddings;

/// <summary>High-level façade for CLIP image embedding. Combines a
/// <see cref="ClipImagePreprocessor"/> with a <see cref="ClipImageEncoder"/> so callers can hand
/// in raw RGB pixels and receive a ready-to-score <see cref="ImageEmbedding"/>.
/// <para>Implements <see cref="IVisionPipeline"/> for registration in the Server / SwarmUI layers.
/// Does not implement the byte-stream <see cref="IEmbeddingPipeline"/> yet — the engine has no
/// image codec (PNG/JPEG decoder), so the public surface here is the raw-pixel one. A future
/// codec wrapper can adapt this pipeline to <see cref="IEmbeddingPipeline"/>.</para></summary>
public sealed class ImageEmbeddingPipeline : IVisionPipeline
{
    private readonly IBackend _backend;
    private readonly ClipImagePreprocessor _preprocessor;
    private readonly ClipImageEncoder _encoder;
    private int _disposed;

    /// <summary>Preset name (e.g. <c>"openai/clip-vit-large-patch14"</c>).</summary>
    public string ModelName { get; }

    /// <summary>Output embedding dimensionality.</summary>
    public int EmbeddingDim => _encoder.EmbeddingDim;

    /// <summary>Constructs the pipeline. The preprocessor and encoder are not owned — the caller (typically <see cref="ClipModelLoader"/>) keeps them alive.</summary>
    public ImageEmbeddingPipeline(IBackend backend, ClipImagePreprocessor preprocessor, ClipImageEncoder encoder, string modelName)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(preprocessor);
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentNullException.ThrowIfNull(modelName);
        _backend = backend;
        _preprocessor = preprocessor;
        _encoder = encoder;
        ModelName = modelName;
    }

    /// <summary>Preprocesses and encodes a single image given HWC-packed RGB u8 pixels.</summary>
    /// <param name="rgbPixels">Source pixels in HWC packed RGB u8 format (the standard output of any image decoder).</param>
    /// <param name="width">Source image width in pixels.</param>
    /// <param name="height">Source image height in pixels.</param>
    /// <param name="sourceId">Optional identifier (file path, hash) carried through onto the embedding for indexing convenience.</param>
    public ImageEmbedding Embed(ReadOnlySpan<byte> rgbPixels, int width, int height, string? sourceId = null)
    {
        ThrowIfDisposed();
        using Tensor pixelValues = _preprocessor.Preprocess(rgbPixels, width, height);
        return _encoder.Encode(_backend, pixelValues, sourceId);
    }

    /// <summary>Preprocesses and encodes a single image already in CHW tensor form. Skips the preprocessor — caller is responsible for resize + normalize. Useful for testing and for pipelines that decode + preprocess upstream (e.g. a server batching layer that wants to share resize work).</summary>
    public ImageEmbedding EmbedTensor(Tensor pixelValues, string? sourceId = null)
    {
        ThrowIfDisposed();
        return _encoder.Encode(_backend, pixelValues, sourceId);
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
    }

    /// <summary>Releases pipeline-internal state. The preprocessor and encoder are owned by the loader, not this pipeline, so they are not disposed here.</summary>
    public void Dispose()
    {
        Interlocked.Exchange(ref _disposed, 1);
    }
}
