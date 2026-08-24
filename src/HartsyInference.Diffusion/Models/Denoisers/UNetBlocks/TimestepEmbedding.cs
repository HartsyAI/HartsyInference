using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.UNetBlocks;

/// <summary>Sinusoidal timestep embedding followed by a two-layer MLP. Converts scalar timestep to a dense vector for conditioning UNet residual blocks.</summary>
/// <param name="embeddingDim">Dimension of the sinusoidal embedding (typically model_channels, e.g., 320).</param>
/// <param name="timeDim">Output dimension of the MLP (typically 4 * model_channels, e.g., 1280).</param>
public sealed unsafe class TimestepEmbedding(int embeddingDim, int timeDim)
{
    private readonly int _embeddingDim = embeddingDim;
    private readonly int _timeDim = timeDim;

    // MLP: Linear(embeddingDim, timeDim) → SiLU → Linear(timeDim, timeDim)
    private Tensor? _linear1Weight;
    private Tensor? _linear1Bias;
    private Tensor? _linear2Weight;
    private Tensor? _linear2Bias;

    /// <summary>Loads weights from named tensors using the given prefix (e.g., "time_embedding").</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _linear1Weight = weights[$"{prefix}.linear_1.weight"];
        _linear1Bias = weights[$"{prefix}.linear_1.bias"];
        _linear2Weight = weights[$"{prefix}.linear_2.weight"];
        _linear2Bias = weights[$"{prefix}.linear_2.bias"];
    }

    /// <summary>Enumerates all weight tensors held by this module.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_linear1Weight is not null) yield return _linear1Weight;
        if (_linear1Bias is not null) yield return _linear1Bias;
        if (_linear2Weight is not null) yield return _linear2Weight;
        if (_linear2Bias is not null) yield return _linear2Bias;
    }

    /// <summary>Computes timestep embeddings for a batch of timesteps. Returns [B, timeDim].</summary>
    public Tensor Forward(IBackend backend, ReadOnlySpan<float> timesteps, int batch)
    {
        Tensor sinEmb = ComputeSinusoidalEmbedding(timesteps, batch);
        Tensor output = ForwardEmbedding(backend, sinEmb);
        sinEmb.Dispose();
        return output;
    }

    /// <summary>Runs only the MLP half on a caller-built embedding vector [B, embeddingDim] → [B, timeDim]. diffusers' <c>TimestepEmbedding</c> is used this way for non-timestep inputs too (e.g. the union ControlNet's sinusoid-encoded control-type vector).</summary>
    public Tensor ForwardEmbedding(IBackend backend, Tensor embedding)
    {
        int batch = (int)embedding.Shape[0];

        TensorShape outShape = new TensorShape(batch, _timeDim);
        Tensor linear1Out = new Tensor(outShape, DType.F32);
        backend.Linear(linear1Out, embedding, _linear1Weight!, _linear1Bias!);

        Tensor siluOut = new Tensor(outShape, DType.F32);
        backend.Silu(siluOut, linear1Out);
        linear1Out.Dispose();

        Tensor linear2Out = new Tensor(outShape, DType.F32);
        backend.Linear(linear2Out, siluOut, _linear2Weight!, _linear2Bias!);
        siluOut.Dispose();

        return linear2Out;
    }

    /// <summary>Computes sinusoidal positional embedding for timesteps. Returns [B, embeddingDim].</summary>
    private Tensor ComputeSinusoidalEmbedding(ReadOnlySpan<float> timesteps, int batch)
    {
        int halfDim = _embeddingDim / 2;
        TensorShape shape = new TensorShape(batch, _embeddingDim);
        Tensor embedding = new Tensor(shape, DType.F32);
        float* embPtr = (float*)embedding.DataPointer;

        // freq[i] = exp(-ln(10000) * i / halfDim)
        // SD1.5 config: downscale_freq_shift=0, flip_sin_to_cos=True
        // diffusers: exponent = -log(10000) * arange(half_dim) / (half_dim - freq_shift)
        //   with freq_shift=0 → divisor = half_dim
        // flip_sin_to_cos=True → cos first, sin second
        float logBase = -MathF.Log(10000.0f) / halfDim;

        for (int b = 0; b < batch; b++)
        {
            float t = timesteps[b];
            for (int i = 0; i < halfDim; i++)
            {
                float freq = MathF.Exp(logBase * i);
                float angle = t * freq;
                embPtr[b * _embeddingDim + i] = MathF.Cos(angle);
                embPtr[b * _embeddingDim + halfDim + i] = MathF.Sin(angle);
            }
        }

        return embedding;
    }

}
