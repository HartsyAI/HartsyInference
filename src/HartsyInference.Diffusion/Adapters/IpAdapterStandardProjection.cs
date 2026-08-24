using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Adapters;

/// <summary>The image-prompt projection used by the standard (non-Plus) IP-Adapter: a single Linear from the CLIP visual_projection output (typically 1024 dims) to <c>num_tokens × cross_attention_dim</c>, reshaped into <c>[B, num_tokens, cross_attention_dim]</c> and finished with a LayerNorm over the per-token cross-attention dim. Math is CPU-side here on purpose — even at SDXL's largest case (1024 → 8192 linear, batch=1) it's a couple of MB of floats, dwarfed by the UNet's per-step cost. Avoiding a backend round-trip keeps the projection as a self-contained unit and lets us skip allocating intermediate device tensors that would flow into <c>cnn</c>'s weight cache. The Plus variant's resampler uses backend ops instead — its matmul shapes (273×1024 K/V over 8 layers) are ~6 GFLOPs and benefit from GPU. Weight key layout matches diffusers / tencent-ailab IP-Adapter: <c>image_proj.proj.weight</c>, <c>image_proj.proj.bias</c>, <c>image_proj.norm.weight</c>, <c>image_proj.norm.bias</c>.</summary>
public sealed unsafe class IpAdapterStandardProjection(int crossAttnDim, int numTokens) : IIpAdapterImageProjection
{
    private readonly int _crossAttnDim = crossAttnDim;
    private readonly int _numTokens = numTokens;

    private Tensor? _projWeight; // [numTokens * crossAttnDim, projectionDim]
    private Tensor? _projBias;   // [numTokens * crossAttnDim]
    private Tensor? _normWeight; // [crossAttnDim]
    private Tensor? _normBias;   // [crossAttnDim]

    public int NumTokens => _numTokens;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "image_proj")
    {
        _projWeight = TensorCasts.EnsureF32(weights[$"{prefix}.proj.weight"]);
        _projBias = TensorCasts.EnsureF32(weights[$"{prefix}.proj.bias"]);
        _normWeight = TensorCasts.EnsureF32(weights[$"{prefix}.norm.weight"]);
        _normBias = TensorCasts.EnsureF32(weights[$"{prefix}.norm.bias"]);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_projWeight is not null) yield return _projWeight;
        if (_projBias is not null) yield return _projBias;
        if (_normWeight is not null) yield return _normWeight;
        if (_normBias is not null) yield return _normBias;
    }

    /// <summary>Projects <paramref name="imageEmbeds"/> (CLS embedding from CLIP-Vision's <c>visual_projection</c>, shape <c>[B, projectionDim]</c>) into image-prompt tokens of shape <c>[B, numTokens, crossAttnDim]</c>. The pipeline then feeds these tokens to each cross-attention layer's IPA K/V branch.</summary>
    public Tensor Forward(IBackend backend, Tensor imageEmbeds)
    {
        if (imageEmbeds.Shape.Rank != 2)
        {
            throw new ArgumentException(
                $"imageEmbeds must be [B, projectionDim]; got {imageEmbeds.Shape}.", nameof(imageEmbeds));
        }
        int batch = (int)imageEmbeds.Shape[0];
        int inDim = (int)imageEmbeds.Shape[1];
        int outDim = _numTokens * _crossAttnDim;

        // Allocate output directly in [B, numTokens, crossAttnDim] form. The Linear
        // fills the underlying memory in [B, outDim] row-major order, which is exactly
        // the same byte layout as [B, numTokens, crossAttnDim] — no separate reshape step.
        Tensor output = new Tensor(new TensorShape(batch, _numTokens, _crossAttnDim), DType.F32);
        float* xp = (float*)imageEmbeds.DataPointer;
        float* wp = (float*)_projWeight!.DataPointer;
        float* biasPtr = (float*)_projBias!.DataPointer;
        float* op = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            int xOff = b * inDim;
            int oOff = b * outDim;
            for (int o = 0; o < outDim; o++)
            {
                float sum = biasPtr[o];
                int wRow = o * inDim;
                for (int i = 0; i < inDim; i++)
                {
                    sum += xp[xOff + i] * wp[wRow + i];
                }
                op[oOff + o] = sum;
            }
        }

        // LayerNorm over the per-token crossAttnDim. Treat the tensor as
        // (batch * numTokens) rows × crossAttnDim — same layout as [B, T, D] in row-major.
        IpAdapterMath.LayerNormInPlace(output, _normWeight!, _normBias!,
            rows: batch * _numTokens, dim: _crossAttnDim, eps: 1e-5f);
        return output;
    }
}

/// <summary>Common interface implemented by both the standard MLP projection and the Plus Resampler. Pipelines stay agnostic to which one is loaded. Tensors are owned by the safetensors loader (mmap-backed) so projections themselves don't need disposal.</summary>
public interface IIpAdapterImageProjection
{
    /// <summary>Number of image-prompt tokens this projection emits per generation.</summary>
    int NumTokens { get; }

    /// <summary>Loads the image projection's weights from the IP-Adapter safetensors dict.</summary>
    void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "image_proj");

    /// <summary>Yields all tensors held by this projection for GPU preloading / freeing.</summary>
    IEnumerable<Tensor> EnumerateWeights();

    /// <summary>Project the CLIP-Vision output into <c>[B, numTokens, crossAttnDim]</c> image-prompt tokens. Standard takes the visual_projection CLS embed (<c>[B, projDim]</c>); Plus takes the penultimate hidden states (<c>[B, seqLen, hiddenSize]</c>).</summary>
    Tensor Forward(IBackend backend, Tensor visionInput);
}

/// <summary>Tiny shared math helpers used by both IPA projection variants. CPU-side on purpose for the simple ops; the Plus resampler still uses backend kernels for its larger matmuls.</summary>
internal static unsafe class IpAdapterMath
{
    /// <summary>In-place LayerNorm over the last dimension. Treats the input tensor as <c>rows × dim</c> in row-major order regardless of declared rank, so the same routine works for both <c>[B, D]</c> and <c>[B, T, D]</c> inputs (caller passes <c>rows = B</c> or <c>rows = B*T</c>).</summary>
    public static void LayerNormInPlace(Tensor x, Tensor weight, Tensor bias, int rows, int dim, float eps)
    {
        float* xp = (float*)x.DataPointer;
        float* wp = (float*)weight.DataPointer;
        float* bp = (float*)bias.DataPointer;
        for (int r = 0; r < rows; r++)
        {
            int off = r * dim;
            float mean = 0f;
            for (int d = 0; d < dim; d++) mean += xp[off + d];
            mean /= dim;
            float var = 0f;
            for (int d = 0; d < dim; d++)
            {
                float diff = xp[off + d] - mean;
                var += diff * diff;
            }
            var /= dim;
            float invStd = 1.0f / MathF.Sqrt(var + eps);
            for (int d = 0; d < dim; d++)
            {
                float normalized = (xp[off + d] - mean) * invStd;
                xp[off + d] = normalized * wp[d] + bp[d];
            }
        }
    }
}
