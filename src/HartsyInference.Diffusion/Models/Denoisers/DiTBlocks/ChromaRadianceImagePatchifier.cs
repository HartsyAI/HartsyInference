using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Chroma Radiance input patchifier: <c>img_in_patch = Conv2d(3 → hidden, kernel P, stride P)</c> over raw
/// RGB in [-1, 1], flattened to <c>[B, (H/P)·(W/P), hidden]</c> transformer tokens. Replaces both the VAE encode and
/// classic Chroma's <c>img_in</c> linear. Keys: <c>img_in_patch.{weight [hidden,3,P,P], bias [hidden]}</c>.</summary>
public sealed unsafe class ChromaRadianceImagePatchifier : IDisposable
{
    private Tensor? _weight;
    private Tensor? _bias;
    private int _disposed;

    /// <summary>Patch size read from the conv weight on load (16 for the published release).</summary>
    public int PatchSize { get; private set; } = 16;

    /// <summary>Transformer hidden size read from the conv weight on load (3072).</summary>
    public int HiddenSize { get; private set; } = 3072;

    /// <summary>Loads <c>img_in_patch.*</c> from a converted Radiance weight dict.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        _weight = weights["img_in_patch.weight"];
        weights.TryGetValue("img_in_patch.bias", out _bias);

        if (_weight.Shape.Rank != 4)
            throw new ArgumentException($"img_in_patch.weight must be 4D [hidden, 3, P, P]; got {_weight.Shape}.");
        HiddenSize = (int)_weight.Shape[0];
        PatchSize = (int)_weight.Shape[2];
    }

    /// <summary>Enumerates weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_weight is not null) yield return _weight;
        if (_bias is not null) yield return _bias;
    }

    /// <summary>Patchifies <paramref name="rgb"/> [B, 3, H, W] into transformer tokens [B, (H/P)·(W/P), hidden].</summary>
    public Tensor Forward(IBackend backend, Tensor rgb)
    {
        if (_weight is null)
            throw new InvalidOperationException("ChromaRadianceImagePatchifier weights not loaded.");

        int batch = (int)rgb.Shape[0];
        int height = (int)rgb.Shape[2];
        int width = (int)rgb.Shape[3];
        int patch = PatchSize;
        if (height % patch != 0 || width % patch != 0)
            throw new ArgumentException($"Image H/W ({height}x{width}) must be divisible by patch size {patch}.");

        int hPacked = height / patch;
        int wPacked = width / patch;
        int seqLen = hPacked * wPacked;

        TensorShape convShape = new TensorShape(batch, HiddenSize, hPacked, wPacked);
        Tensor conv = new Tensor(convShape, DType.F32);
        backend.Conv2D(conv, rgb, _weight, _bias, patch, patch, 0, 0);

        // [B, hidden, N] → [B, N, hidden] layout flip per batch element via the shared transpose op.
        TensorShape tokShape = new TensorShape(batch, seqLen, HiddenSize);
        Tensor tokens = new Tensor(tokShape, DType.F32);
        if (batch == 1)
        {
            backend.Transpose2D(tokens, conv, HiddenSize, seqLen);
        }
        else
        {
            float* src = (float*)conv.DataPointer;
            float* dst = (float*)tokens.DataPointer;
            for (int b = 0; b < batch; b++)
            {
                long inBase = (long)b * HiddenSize * seqLen;
                long outBase = (long)b * seqLen * HiddenSize;
                for (int c = 0; c < HiddenSize; c++)
                    for (int s = 0; s < seqLen; s++)
                        dst[outBase + (long)s * HiddenSize + c] = src[inBase + (long)c * seqLen + s];
            }
        }
        conv.Dispose();
        return tokens;
    }

    /// <summary>Releases weight references.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _weight = null;
            _bias = null;
        }
        GC.SuppressFinalize(this);
    }
}
