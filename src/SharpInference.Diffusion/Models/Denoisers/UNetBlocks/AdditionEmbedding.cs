using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Denoisers.UNetBlocks;

/// <summary>SDXL ADM micro-conditioning embedding. Embeds size/crop/target scalars via sinusoidal encoding, concatenates with pooled text embedding, and projects to the timestep embedding dimension.</summary>
public sealed unsafe class AdditionEmbedding
{
    private readonly int _admInChannels;
    private readonly int _timeDim;
    private readonly int _additionTimeEmbedDim;

    // Linear projection: admInChannels → timeDim → timeDim
    private Tensor? _linear1Weight;
    private Tensor? _linear1Bias;
    private Tensor? _linear2Weight;
    private Tensor? _linear2Bias;

    /// <summary>Creates an ADM addition embedding module.</summary>
    /// <param name="admInChannels">Total input dimension (2816 for SDXL base: 1280 pooled + 6*256 scalars).</param>
    /// <param name="timeDim">Output dimension matching timestep embedding (1280 for SDXL base).</param>
    /// <param name="additionTimeEmbedDim">Sinusoidal embedding dimension per scalar (256 for SDXL).</param>
    public AdditionEmbedding(int admInChannels, int timeDim, int additionTimeEmbedDim)
    {
        _admInChannels = admInChannels;
        _timeDim = timeDim;
        _additionTimeEmbedDim = additionTimeEmbedDim;
    }

    /// <summary>Loads weights from named tensors using the given prefix (e.g., "add_embedding").</summary>
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

    /// <summary>Computes the ADM conditioning vector. pooledTextEmb [B, pooledDim], sizeCondition contains scalar values to embed sinusoidally. Returns [B, timeDim] to be added to timestep embedding.</summary>
    /// <param name="pooledTextEmb">Pooled text embedding from CLIP-G [B, 1280].</param>
    /// <param name="sizeCondition">Flattened scalar conditioning values. For SDXL base: [origH, origW, cropTop, cropLeft, targetH, targetW]. For refiner: [origH, origW, cropTop, cropLeft, aestheticScore].</param>
    /// <param name="batch">Batch size.</param>
    public Tensor Forward(IBackend backend, Tensor pooledTextEmb, ReadOnlySpan<float> sizeCondition, int batch)
    {
        int pooledDim = (int)pooledTextEmb.Shape[1];
        int numScalars = sizeCondition.Length;

        // 1. Embed each scalar with sinusoidal encoding [additionTimeEmbedDim]
        int totalEmbDim = numScalars * _additionTimeEmbedDim;

        // 2. Concatenate: [pooledDim] + [numScalars * additionTimeEmbedDim] = [admInChannels]
        TensorShape admShape = new TensorShape(batch, _admInChannels);
        Tensor admVector = new Tensor(admShape, DType.F32);
        float* admPtr = (float*)admVector.DataPointer;
        float* pooledPtr = (float*)pooledTextEmb.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            int outOffset = b * _admInChannels;

            // Copy pooled text embedding
            for (int d = 0; d < pooledDim; d++)
            {
                admPtr[outOffset + d] = pooledPtr[b * pooledDim + d];
            }

            // Embed each scalar and concatenate
            int embOffset = outOffset + pooledDim;
            for (int s = 0; s < numScalars; s++)
            {
                float value = sizeCondition[s];
                EmbedScalar(admPtr, embOffset + s * _additionTimeEmbedDim, value, _additionTimeEmbedDim);
            }
        }

        // 3. Linear1: [B, admInChannels] → [B, timeDim]
        TensorShape outShape = new TensorShape(batch, _timeDim);
        Tensor linear1Out = new Tensor(outShape, DType.F32);
        backend.Linear(linear1Out, admVector, _linear1Weight!, _linear1Bias!);
        admVector.Dispose();

        // 4. SiLU activation
        Tensor siluOut = new Tensor(outShape, DType.F32);
        backend.Silu(siluOut, linear1Out);
        linear1Out.Dispose();

        // 5. Linear2: [B, timeDim] → [B, timeDim]
        Tensor output = new Tensor(outShape, DType.F32);
        backend.Linear(output, siluOut, _linear2Weight!, _linear2Bias!);
        siluOut.Dispose();

        return output;
    }

    /// <summary>Embeds a single scalar value using sinusoidal positional encoding. Same function as timestep embedding.</summary>
    private static void EmbedScalar(float* output, int offset, float value, int dim)
    {
        int halfDim = dim / 2;
        float logBase = -MathF.Log(10000.0f) / halfDim;

        for (int i = 0; i < halfDim; i++)
        {
            float freq = MathF.Exp(logBase * i);
            float angle = value * freq;
            output[offset + i] = MathF.Cos(angle);
            output[offset + halfDim + i] = MathF.Sin(angle);
        }
    }

}
