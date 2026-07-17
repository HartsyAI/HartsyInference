using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Segmentation.Sam2;

/// <summary>SAM prompt encoder: turns point/box prompts into sparse token embeddings the mask decoder
/// consumes. Each prompt coordinate is positionally encoded (<see cref="SamPositionalEncoding"/>) and added
/// to a learned per-type embedding:
/// <list type="bullet">
///   <item>index 0 — background point</item>
///   <item>index 1 — foreground point</item>
///   <item>index 2 — box top-left corner</item>
///   <item>index 3 — box bottom-right corner</item>
/// </list>
/// A "not a point" embedding pads point-only prompts (SAM appends one padding token when no box is given).</summary>
public sealed unsafe class SamPromptEncoder
{
    private readonly SamPositionalEncoding _pe;
    private readonly float[][] _pointEmbeddings; // [4][embedDim]
    private readonly float[] _notAPoint;         // [embedDim]
    private readonly int _embedDim;

    /// <summary>Embedding dimensionality.</summary>
    public int EmbedDim => _embedDim;

    /// <summary>Creates the prompt encoder from loaded weights.</summary>
    /// <param name="pe">Positional encoder built from the Gaussian matrix.</param>
    /// <param name="pointEmbeddings">Four learned <c>[embedDim]</c> embeddings (bg, fg, box-tl, box-br).</param>
    /// <param name="notAPointEmbedding">The padding embedding <c>[embedDim]</c>.</param>
    public SamPromptEncoder(SamPositionalEncoding pe, IReadOnlyList<Tensor> pointEmbeddings, Tensor notAPointEmbedding)
    {
        if (pointEmbeddings.Count != 4)
            throw new ArgumentException($"SAM expects 4 point embeddings (bg, fg, box-tl, box-br); got {pointEmbeddings.Count}.", nameof(pointEmbeddings));

        _pe = pe;
        _embedDim = pe.EmbedDim;
        _pointEmbeddings = new float[4][];
        for (int i = 0; i < 4; i++)
        {
            _pointEmbeddings[i] = ToF32Array(pointEmbeddings[i], _embedDim);
        }
        _notAPoint = ToF32Array(notAPointEmbedding, _embedDim);
    }

    /// <summary>Builds the dense positional-encoding grid <c>[1, embedDim, height, width]</c> the mask decoder
    /// adds to the image tokens — SAM's <c>prompt_encoder.get_dense_pe()</c>. Each grid cell is encoded with the
    /// same Gaussian PE as the sparse prompts (cell-centered, normalized by the grid size), so prompt and image
    /// positions share one coordinate frame.</summary>
    public Tensor BuildDensePositionalEncoding(int height, int width)
    {
        if (height <= 0 || width <= 0)
            throw new ArgumentException($"Dense PE grid must be positive; got {height}x{width}.");

        Tensor outT = new Tensor(new TensorShape(1, _embedDim, height, width), DType.F32);
        float* o = (float*)outT.DataPointer;
        Span<float> scratch = stackalloc float[_embedDim];
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                _pe.Encode(j, i, width, height, scratch);
                for (int d = 0; d < _embedDim; d++)
                {
                    o[((long)d * height + i) * width + j] = scratch[d];
                }
            }
        }
        return outT;
    }

    /// <summary>Encodes a prompt into sparse token embeddings <c>[1, numTokens, embedDim]</c>. Token order:
    /// each point in order, then box corners (if any), then one padding token when no box is present.</summary>
    public Tensor EncodeSparse(SamPrompt prompt, int imageWidth, int imageHeight)
    {
        int numPoints = prompt.Points.Count;
        bool hasBox = prompt.Box.HasValue;
        int numTokens = numPoints + (hasBox ? 2 : 0) + (hasBox ? 0 : 1); // pad one token when no box
        if (numTokens == 0) numTokens = 1;

        Tensor outT = new Tensor(new TensorShape(1, numTokens, _embedDim), DType.F32);
        float* o = (float*)outT.DataPointer;
        Span<float> scratch = stackalloc float[_embedDim];

        int t = 0;
        foreach (SamPointPrompt p in prompt.Points)
        {
            _pe.Encode(p.X, p.Y, imageWidth, imageHeight, scratch);
            float[] typeEmbed = _pointEmbeddings[p.Foreground ? 1 : 0];
            WriteToken(o, t++, scratch, typeEmbed);
        }

        if (hasBox)
        {
            SamBoxPrompt b = prompt.Box!.Value;
            _pe.Encode(b.X0, b.Y0, imageWidth, imageHeight, scratch);
            WriteToken(o, t++, scratch, _pointEmbeddings[2]);
            _pe.Encode(b.X1, b.Y1, imageWidth, imageHeight, scratch);
            WriteToken(o, t++, scratch, _pointEmbeddings[3]);
        }
        else
        {
            // Padding token: not-a-point embedding (no positional component).
            for (int d = 0; d < _embedDim; d++) o[t * _embedDim + d] = _notAPoint[d];
            t++;
        }

        return outT;
    }

    private void WriteToken(float* o, int tokenIdx, ReadOnlySpan<float> posEnc, float[] typeEmbed)
    {
        int baseOff = tokenIdx * _embedDim;
        for (int d = 0; d < _embedDim; d++)
        {
            o[baseOff + d] = posEnc[d] + typeEmbed[d];
        }
    }

    private static float[] ToF32Array(Tensor t, int expectedLen)
    {
        Tensor f32 = t.DType == DType.F32 ? t : t.CastTo(DType.F32);
        if (f32.Shape.ElementCount < expectedLen)
            throw new ArgumentException($"Embedding has {f32.Shape.ElementCount} elements; need {expectedLen}.");
        float[] arr = new float[expectedLen];
        f32.AsReadOnlySpan<float>()[..expectedLen].CopyTo(arr);
        if (!ReferenceEquals(f32, t)) f32.Dispose();
        return arr;
    }
}
