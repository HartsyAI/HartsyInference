using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Chroma's <c>ChromaCombinedTimestepTextProjEmbeddings</c>: builds the 64-dim per-row input vector
/// fed to <see cref="ChromaApproximator"/>. For each modulation slot in <c>[0, mod_index_length)</c>, emits
/// <c>concat([Timesteps_16(t), Timesteps_16(0), get_timestep_embedding_32(slot * 1000)])</c>.
///
/// The first half (32 values, depends on the timestep) is identical across all rows of a batch element;
/// the second half (32 values, the row's slot index encoded via sinusoidal embedding) is constant across
/// all batch elements but varies per row. The result is `[B, mod_index_length, 64]`.
///
/// Reference: <c>diffusers/models/transformers/transformer_chroma.py:152-181</c>.</summary>
public sealed unsafe class ChromaCombinedTimestepEmbeddings
{
    private readonly int _numChannels;
    private readonly int _modIndexLength;
    private readonly float[] _modProj;

    /// <summary>Creates the timestep+slot embedding builder.</summary>
    /// <param name="numChannels">Per-Timesteps width (<c>approximator_num_channels / 4 = 16</c> for v1). The output dim equals <c>4 * numChannels</c>.</param>
    /// <param name="modIndexLength">Number of modulation rows: <c>3 * num_singles + 12 * num_doubles + 2</c> (= 344 for Chroma v1).</param>
    public ChromaCombinedTimestepEmbeddings(int numChannels, int modIndexLength)
    {
        _numChannels = numChannels;
        _modIndexLength = modIndexLength;

        int modProjDim = 2 * numChannels;
        _modProj = new float[modIndexLength * modProjDim];
        BuildModProj(_modProj, modIndexLength, modProjDim);
    }

    /// <summary>Per-row width (<c>4 * numChannels</c> = 64 for v1).</summary>
    public int OutDim => 4 * _numChannels;

    /// <summary>Number of modulation rows produced.</summary>
    public int ModIndexLength => _modIndexLength;

    /// <summary>Computes the input vector for the approximator. Output shape: <c>[batch, modIndexLength, 4 * numChannels]</c>.</summary>
    /// <param name="timestep">Timestep value already scaled by 1000 (the transformer multiplies by 1000 before calling).</param>
    /// <param name="batch">Batch size. All batch elements share the same timestep (transformer broadcasts per-batch but the value is the same).</param>
    public Tensor Forward(float timestep, int batch)
    {
        int outDim = OutDim;
        int halfRow = 2 * _numChannels;

        TensorShape shape = new TensorShape(batch, _modIndexLength, outDim);
        Tensor output = new Tensor(shape, DType.F32);

        Span<float> tEmb = stackalloc float[_numChannels];
        Span<float> gEmb = stackalloc float[_numChannels];
        TimestepEmbedding(tEmb, timestep, _numChannels);
        TimestepEmbedding(gEmb, 0.0f, _numChannels);

        float* outPtr = (float*)output.DataPointer;
        fixed (float* modProjPtr = _modProj)
        {
            for (int b = 0; b < batch; b++)
            {
                for (int r = 0; r < _modIndexLength; r++)
                {
                    long rowOffset = ((long)b * _modIndexLength + r) * outDim;
                    for (int i = 0; i < _numChannels; i++)
                        outPtr[rowOffset + i] = tEmb[i];
                    for (int i = 0; i < _numChannels; i++)
                        outPtr[rowOffset + _numChannels + i] = gEmb[i];

                    int modProjRowOffset = r * halfRow;
                    for (int i = 0; i < halfRow; i++)
                        outPtr[rowOffset + halfRow + i] = modProjPtr[modProjRowOffset + i];
                }
            }
        }

        return output;
    }

    /// <summary>Sinusoidal timestep embedding with <c>flip_sin_to_cos=True, downscale_freq_shift=0</c>.
    /// Output layout: <c>[cos_0..cos_{half-1}, sin_0..sin_{half-1}]</c>. Matches <c>get_timestep_embedding</c>
    /// with the matching kwargs (the cos half originates from the second half of <c>[sin, cos]</c> after the flip).</summary>
    private static void TimestepEmbedding(Span<float> output, float timestep, int numChannels)
    {
        int halfDim = numChannels / 2;
        float maxPeriod = 10000.0f;
        for (int i = 0; i < halfDim; i++)
        {
            float exponent = -MathF.Log(maxPeriod) * i / halfDim;
            float freq = MathF.Exp(exponent);
            float angle = timestep * freq;
            output[i] = MathF.Cos(angle);
            output[halfDim + i] = MathF.Sin(angle);
        }
    }

    /// <summary>Builds the static slot-index sinusoidal table. Each row <c>r</c> = <c>get_timestep_embedding(r * 1000, 2 * num_channels, flip_sin_to_cos=True, downscale_freq_shift=0)</c>.</summary>
    private static void BuildModProj(float[] modProj, int modIndexLength, int embeddingDim)
    {
        int halfDim = embeddingDim / 2;
        float maxPeriod = 10000.0f;
        for (int r = 0; r < modIndexLength; r++)
        {
            float t = r * 1000.0f;
            int rowBase = r * embeddingDim;
            for (int i = 0; i < halfDim; i++)
            {
                float exponent = -MathF.Log(maxPeriod) * i / halfDim;
                float freq = MathF.Exp(exponent);
                float angle = t * freq;
                modProj[rowBase + i] = MathF.Cos(angle);
                modProj[rowBase + halfDim + i] = MathF.Sin(angle);
            }
        }
    }
}
