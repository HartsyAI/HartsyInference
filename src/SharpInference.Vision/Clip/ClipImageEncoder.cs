using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.TextEncoders;
using SharpInference.Vision.Embeddings;

namespace SharpInference.Vision.Clip;

/// <summary>Thin wrapper over the diffusion package's <see cref="ClipVisionEncoder"/>. Adds standalone-CLIP behavior: L2-normalizes the projected CLS embedding so the result is a unit vector ready for cosine similarity. The transformer math itself is unchanged — see <see cref="ClipVisionEncoder"/>.</summary>
/// <remarks>The wrapped encoder is not owned by this class; <see cref="ClipModelLoader"/> manages its lifetime alongside the safetensors mmap.</remarks>
public sealed unsafe class ClipImageEncoder
{
    private readonly ClipVisionEncoder _encoder;

    /// <summary>Vision tower configuration in use.</summary>
    public ClipVisionEncoderConfig Config => _encoder.Config;

    /// <summary>Output embedding dimensionality (equal to <see cref="ClipVisionEncoderConfig.ProjectionDim"/>).</summary>
    public int EmbeddingDim => _encoder.Config.ProjectionDim;

    /// <summary>Wraps a pre-loaded vision encoder. <paramref name="encoder"/> must have its weights loaded via <see cref="ClipVisionEncoder.LoadWeights"/>, including <c>visual_projection</c>.</summary>
    public ClipImageEncoder(ClipVisionEncoder encoder)
    {
        ArgumentNullException.ThrowIfNull(encoder);
        if (encoder.Config.ProjectionDim <= 0)
            throw new ArgumentException("Vision encoder must have ProjectionDim > 0 for standalone CLIP scoring — a checkpoint without visual_projection cannot produce a comparable embedding.", nameof(encoder));
        _encoder = encoder;
    }

    /// <summary>Encodes a preprocessed pixel tensor (<c>[B, 3, H, W]</c>, CLIP-normalized F32) into batched embeddings (<c>[B, embeddingDim]</c>) without L2 normalization. Use this when you want raw projections (e.g. for downstream linear-probe heads).</summary>
    public Tensor EncodeRaw(IBackend backend, Tensor pixelValues)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(pixelValues);
        return _encoder.EncodeImageEmbeds(backend, pixelValues);
    }

    /// <summary>Encodes a single preprocessed image into an L2-normalized <see cref="ImageEmbedding"/>. <paramref name="pixelValues"/> must be <c>[1, 3, H, W]</c> F32 — exactly what <see cref="ClipImagePreprocessor.Preprocess"/> produces.</summary>
    public ImageEmbedding Encode(IBackend backend, Tensor pixelValues, string? sourceId = null)
    {
        if (pixelValues.Shape.Rank != 4 || pixelValues.Shape[0] != 1)
            throw new ArgumentException($"Single-image Encode requires a [1, 3, H, W] tensor; got {pixelValues.Shape}. For batches use {nameof(EncodeBatch)}.", nameof(pixelValues));

        Tensor raw = EncodeRaw(backend, pixelValues);
        L2NormalizeInPlace(raw);
        return new ImageEmbedding(raw, sourceId);
    }

    /// <summary>Encodes a batch of preprocessed images. The input is a single <c>[B, 3, H, W]</c> tensor; the output is a list of <see cref="ImageEmbedding"/> values, one per batch element. Each embedding owns a freshly-allocated <c>[1, embeddingDim]</c> tensor sliced from the batch result.</summary>
    public IReadOnlyList<ImageEmbedding> EncodeBatch(IBackend backend, Tensor pixelValues, IReadOnlyList<string?>? sourceIds = null)
    {
        if (pixelValues.Shape.Rank != 4)
            throw new ArgumentException($"Batch Encode requires a [B, 3, H, W] tensor; got {pixelValues.Shape}.", nameof(pixelValues));

        int batch = (int)pixelValues.Shape[0];
        if (sourceIds is not null && sourceIds.Count != batch)
            throw new ArgumentException($"sourceIds length {sourceIds.Count} != batch {batch}.", nameof(sourceIds));

        using Tensor batchOut = EncodeRaw(backend, pixelValues);
        int dim = EmbeddingDim;
        ImageEmbedding[] results = new ImageEmbedding[batch];

        float* batchPtr = (float*)batchOut.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            Tensor slice = new Tensor(new TensorShape(1, dim), DType.F32);
            float* slicePtr = (float*)slice.DataPointer;
            ReadOnlySpan<float> src = new ReadOnlySpan<float>(batchPtr + b * dim, dim);
            Span<float> dst = new Span<float>(slicePtr, dim);
            src.CopyTo(dst);
            L2NormalizeInPlace(slice);
            results[b] = new ImageEmbedding(slice, sourceIds?[b]);
        }

        return results;
    }

    /// <summary>L2-normalize a <c>[N, dim]</c> tensor in place. Each row becomes a unit vector. Tiny rows (norm &lt; 1e-12) are passed through to avoid div-by-zero — they should never occur with real embeddings.</summary>
    internal static void L2NormalizeInPlace(Tensor t)
    {
        if (t.DType != DType.F32)
            throw new ArgumentException($"L2NormalizeInPlace requires F32; got {t.DType}.", nameof(t));
        long n = t.Shape[0];
        long dim = t.Shape.Rank >= 2 ? t.Shape[1] : t.ElementCount / n;

        float* ptr = (float*)t.DataPointer;
        for (long row = 0; row < n; row++)
        {
            float* rowPtr = ptr + row * dim;
            double sumSq = 0.0;
            for (long i = 0; i < dim; i++)
            {
                float v = rowPtr[i];
                sumSq += v * v;
            }
            float invNorm = sumSq < 1e-24 ? 1f : (float)(1.0 / Math.Sqrt(sumSq));
            for (long i = 0; i < dim; i++)
            {
                rowPtr[i] *= invNorm;
            }
        }
    }
}
