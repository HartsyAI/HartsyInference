using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Patch embedding for SD3 MMDiT. Converts a latent [B, C, H, W] into a sequence of patch token embeddings [B, numPatches, embedDim] via a strided Conv2D.</summary>
public sealed unsafe class PatchEmbed
{
    private readonly int _patchSize;
    private readonly int _inChannels;
    private readonly int _embedDim;

    private Tensor? _projWeight;
    private Tensor? _projBias;
    private Tensor? _posEmbed;

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

    /// <summary>Loads the Conv2D projection weights and optional precomputed positional embeddings.</summary>
    public void LoadWeights(Tensor projWeight, Tensor projBias, Tensor? posEmbed = null)
    {
        _projWeight = projWeight;
        _projBias = projBias;
        _posEmbed = posEmbed;
    }

    /// <summary>Patch-embeds a latent. Input: [B, C, H, W] → Output: [B, numPatches, embedDim] with positional embeddings added.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="input">Latent tensor [B, inChannels, H, W].</param>
    /// <returns>Patch tokens [B, gridH * gridW, embedDim] with positional embeddings.</returns>
    public Tensor Forward(IBackend backend, Tensor input)
    {
        int batch = (int)input.Shape[0];
        int height = (int)input.Shape[2];
        int width = (int)input.Shape[3];
        int gridH = height / _patchSize;
        int gridW = width / _patchSize;
        int numPatches = gridH * gridW;

        // Conv2D with stride = patchSize: [B, C, H, W] → [B, embedDim, gridH, gridW]
        TensorShape convShape = new TensorShape(batch, _embedDim, gridH, gridW);
        Tensor convOut = new Tensor(convShape, DType.F32);
        backend.Conv2D(convOut, input, _projWeight!, _projBias!, _patchSize, _patchSize, 0, 0);

        // Flatten spatial to sequence: [B, embedDim, gridH, gridW] → [B, gridH*gridW, embedDim]
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

        // Add positional embedding (cropped to actual grid size)
        if (_posEmbed is not null)
        {
            AddPositionalEmbedding(tokens, _posEmbed, batch, numPatches, _embedDim);
        }

        return tokens;
    }

    /// <summary>Returns the patch grid dimensions for a given spatial resolution.</summary>
    public (int gridH, int gridW) GetGridSize(int height, int width) => (height / _patchSize, width / _patchSize);

    private static void AddPositionalEmbedding(Tensor tokens, Tensor posEmbed, int batch, int numPatches, int embedDim)
    {
        float* tokPtr = (float*)tokens.DataPointer;
        float* posPtr = (float*)posEmbed.DataPointer;

        // posEmbed is [1, maxPatches, embedDim] — crop to numPatches
        for (int b = 0; b < batch; b++)
        {
            for (int p = 0; p < numPatches; p++)
            {
                int tokOffset = (b * numPatches + p) * embedDim;
                int posOffset = p * embedDim;
                for (int d = 0; d < embedDim; d++)
                {
                    tokPtr[tokOffset + d] += posPtr[posOffset + d];
                }
            }
        }
    }
}
