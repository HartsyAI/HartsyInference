using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Per-head RMSNorm with learnable scale for query-key normalization. Prevents attention logit explosion by normalizing each head's Q/K vector independently before the dot product.</summary>
public sealed unsafe class QkNorm
{
    private readonly int _headDim;
    private readonly float _eps;
    private Tensor? _weight;

    /// <summary>Creates a QK-norm layer for the given head dimension.</summary>
    /// <param name="headDim">Per-head dimension (64 for SD3).</param>
    /// <param name="eps">RMSNorm epsilon. Default: 1e-6.</param>
    public QkNorm(int headDim, float eps = 1e-6f)
    {
        _headDim = headDim;
        _eps = eps;
    }

    /// <summary>Loads the learnable scale weight. Shape: [headDim]. Auto-casts to F32 if needed (Forward uses float* directly).</summary>
    public void LoadWeights(Tensor weight)
    {
        _weight = weight.DType != DType.F32 ? weight.CastTo(DType.F32) : weight;
    }

    /// <summary>Enumerates all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_weight is not null) yield return _weight;
    }

    /// <summary>Applies per-head RMSNorm in-place. Data is treated as numVectors contiguous vectors of length headDim.</summary>
    /// <param name="output">Output tensor, same size as input.</param>
    /// <param name="input">Input tensor laid out as [numVectors * headDim] contiguous floats.</param>
    /// <param name="numVectors">Total number of headDim-length vectors (= batch * seqLen * numHeads).</param>
    public void Forward(Tensor output, Tensor input, int numVectors)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        float* wPtr = (float*)_weight!.DataPointer;

        for (int v = 0; v < numVectors; v++)
        {
            int offset = v * _headDim;

            // RMS = sqrt(mean(x^2) + eps)
            float sumSq = 0f;
            for (int d = 0; d < _headDim; d++)
            {
                float val = inPtr[offset + d];
                sumSq += val * val;
            }
            float invRms = 1.0f / MathF.Sqrt(sumSq / _headDim + _eps);

            // Normalize and apply learnable scale
            for (int d = 0; d < _headDim; d++)
            {
                outPtr[offset + d] = inPtr[offset + d] * invRms * wPtr[d];
            }
        }
    }
}
