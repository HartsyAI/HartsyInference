using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Denoisers.UNetBlocks;

/// <summary>Sinusoidal timestep embedding followed by a two-layer MLP. Converts scalar timestep to a dense vector for conditioning UNet residual blocks.</summary>
public sealed unsafe class TimestepEmbedding
{
    private readonly int _embeddingDim;
    private readonly int _timeDim;

    // MLP: Linear(embeddingDim, timeDim) → SiLU → Linear(timeDim, timeDim)
    private Tensor? _linear1Weight;
    private Tensor? _linear1Bias;
    private Tensor? _linear2Weight;
    private Tensor? _linear2Bias;

    /// <summary>Creates a timestep embedding module.</summary>
    /// <param name="embeddingDim">Dimension of the sinusoidal embedding (typically model_channels, e.g., 320).</param>
    /// <param name="timeDim">Output dimension of the MLP (typically 4 * model_channels, e.g., 1280).</param>
    public TimestepEmbedding(int embeddingDim, int timeDim)
    {
        _embeddingDim = embeddingDim;
        _timeDim = timeDim;
    }

    /// <summary>Loads weights from named tensors using the given prefix (e.g., "time_embedding").</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _linear1Weight = weights[$"{prefix}.linear_1.weight"];
        _linear1Bias = weights[$"{prefix}.linear_1.bias"];
        _linear2Weight = weights[$"{prefix}.linear_2.weight"];
        _linear2Bias = weights[$"{prefix}.linear_2.bias"];
    }

    /// <summary>Computes timestep embeddings for a batch of timesteps. Returns [B, timeDim].</summary>
    public Tensor Forward(IBackend backend, ReadOnlySpan<float> timesteps, int batch)
    {
        // 1. Sinusoidal positional embedding: [B, embeddingDim]
        Tensor sinEmb = ComputeSinusoidalEmbedding(timesteps, batch);

        // 2. Linear1: [B, embeddingDim] → [B, timeDim]
        TensorShape outShape = new TensorShape(batch, _timeDim);
        Tensor linear1Out = new Tensor(outShape, DType.F32);
        LinearForward(linear1Out, sinEmb, _linear1Weight!, _linear1Bias!, batch, _embeddingDim, _timeDim);
        sinEmb.Dispose();

        // 3. SiLU activation
        Tensor siluOut = new Tensor(outShape, DType.F32);
        backend.Silu(siluOut, linear1Out);
        linear1Out.Dispose();

        // 4. Linear2: [B, timeDim] → [B, timeDim]
        Tensor linear2Out = new Tensor(outShape, DType.F32);
        LinearForward(linear2Out, siluOut, _linear2Weight!, _linear2Bias!, batch, _timeDim, _timeDim);
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

    /// <summary>Dense linear layer: output = input @ weight^T + bias.</summary>
    private static void LinearForward(Tensor output, Tensor input, Tensor weight, Tensor bias, int batch, int inDim, int outDim)
    {
        float* inPtr = (float*)input.DataPointer;
        float* wPtr = (float*)weight.DataPointer;
        float* bPtr = (float*)bias.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        // weight is [outDim, inDim], so output[b,o] = sum_i(input[b,i] * weight[o,i]) + bias[o]
        for (int b = 0; b < batch; b++)
        {
            for (int o = 0; o < outDim; o++)
            {
                float sum = bPtr[o];
                int wOffset = o * inDim;
                int inOffset = b * inDim;
                for (int i = 0; i < inDim; i++)
                {
                    sum += inPtr[inOffset + i] * wPtr[wOffset + i];
                }
                outPtr[b * outDim + o] = sum;
            }
        }
    }
}
