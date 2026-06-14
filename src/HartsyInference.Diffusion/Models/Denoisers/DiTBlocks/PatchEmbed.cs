using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Patch embedding for SD3 / SD3.5 MMDiT. Converts a latent [B, C, H, W] into a sequence of patch token embeddings [B, numPatches, embedDim] via a strided Conv2D, then adds learned 2D positional embeddings (center-cropped from the stored max-size grid for variable resolutions).</summary>
public sealed unsafe class PatchEmbed
{
    private readonly int _patchSize;
    private readonly int _inChannels;
    private readonly int _embedDim;

    private Tensor? _projWeight;
    private Tensor? _projBias;
    private Tensor? _posEmbed;
    private int _posEmbedMaxSize;

    /// <summary>Creates a patch embedding layer.</summary>
    /// <param name="patchSize">Patch size (2 for SD3).</param>
    /// <param name="inChannels">Input latent channels (16 for SD3).</param>
    /// <param name="embedDim">Output embedding dimension (= hidden_size).</param>
    public PatchEmbed(int patchSize, int inChannels, int embedDim)
    {
        _patchSize = patchSize;
        _inChannels = inChannels;
        _embedDim = embedDim;
    }

    /// <summary>Loads the Conv2D projection weights and optional precomputed positional embeddings. The pos_embed is shape [1, maxSize*maxSize, embedDim] flattened from a [maxSize, maxSize, embedDim] grid. The pos_embed is auto-cast to F32 because <see cref="AddPositionalEmbedding"/> reads it via <c>float*</c> on the CPU side and would otherwise bit-reinterpret F16/BF16 bytes as F32 (cf. PHASE_3_DEVIATIONS #30 for the same class of bug in Z-Image RmsNorm).</summary>
    public void LoadWeights(Tensor projWeight, Tensor projBias, Tensor? posEmbed = null)
    {
        _projWeight = projWeight;
        _projBias = projBias;
        _posEmbed = posEmbed is not null && posEmbed.DType != DType.F32
            ? posEmbed.CastTo(DType.F32)
            : posEmbed;

        if (_posEmbed is not null)
        {
            long numStored = _posEmbed.Shape[_posEmbed.Shape.Rank - 2];
            int maxSize = (int)MathF.Round(MathF.Sqrt(numStored));
            if (maxSize * maxSize != numStored)
                throw new InvalidOperationException(
                    $"pos_embed must be a square grid; got {numStored} entries (sqrt={maxSize})");
            _posEmbedMaxSize = maxSize;
        }
    }

    /// <summary>Yields all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_projWeight is not null) yield return _projWeight;
        if (_projBias is not null) yield return _projBias;
        if (_posEmbed is not null) yield return _posEmbed;
    }

    /// <summary>Patch-embeds a latent. Input: [B, C, H, W] → Output: [B, numPatches, embedDim] with positional embeddings added.</summary>
    public Tensor Forward(IBackend backend, Tensor input)
    {
        int batch = (int)input.Shape[0];
        int height = (int)input.Shape[2];
        int width = (int)input.Shape[3];
        int gridH = height / _patchSize;
        int gridW = width / _patchSize;
        int numPatches = gridH * gridW;

        TensorShape convShape = new TensorShape(batch, _embedDim, gridH, gridW);
        Tensor convOut = new Tensor(convShape, DType.F32);
        backend.Conv2D(convOut, input, _projWeight!, _projBias!, _patchSize, _patchSize, 0, 0);

        TensorShape seqShape = new TensorShape(batch, numPatches, _embedDim);
        Tensor tokens = new Tensor(seqShape, DType.F32);

        float* convPtr = (float*)convOut.DataPointer;
        float* tokPtr = (float*)tokens.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < _embedDim; c++)
            {
                for (int p = 0; p < numPatches; p++)
                {
                    tokPtr[(b * numPatches + p) * _embedDim + c] = convPtr[(b * _embedDim + c) * numPatches + p];
                }
            }
        }

        convOut.Dispose();

        if (_posEmbed is not null)
        {
            AddPositionalEmbedding(tokens, _posEmbed, batch, gridH, gridW, _embedDim, _posEmbedMaxSize);
        }

        return tokens;
    }

    /// <summary>Returns the patch grid dimensions for a given spatial resolution.</summary>
    public (int gridH, int gridW) GetGridSize(int height, int width) => (height / _patchSize, width / _patchSize);

    /// <summary>Adds a learned 2D positional embedding, center-cropped from the stored max-size grid. Stored layout: [1, maxSize*maxSize, embedDim] in row-major (row r, col c → index r*maxSize + c).</summary>
    private static void AddPositionalEmbedding(Tensor tokens, Tensor posEmbed, int batch, int gridH, int gridW, int embedDim, int maxSize)
    {
        float* tokPtr = (float*)tokens.DataPointer;
        float* posPtr = (float*)posEmbed.DataPointer;

        int startH = (maxSize - gridH) / 2;
        int startW = (maxSize - gridW) / 2;
        int numPatches = gridH * gridW;

        for (int b = 0; b < batch; b++)
        {
            for (int r = 0; r < gridH; r++)
            {
                for (int c = 0; c < gridW; c++)
                {
                    int patchIdx = r * gridW + c;
                    int posIdx = (startH + r) * maxSize + (startW + c);
                    int tokOffset = (b * numPatches + patchIdx) * embedDim;
                    int posOffset = posIdx * embedDim;
                    for (int d = 0; d < embedDim; d++)
                        tokPtr[tokOffset + d] += posPtr[posOffset + d];
                }
            }
        }
    }
}
