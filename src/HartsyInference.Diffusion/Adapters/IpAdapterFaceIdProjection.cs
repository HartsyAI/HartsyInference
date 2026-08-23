using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Adapters;

/// <summary>The image-prompt projection used by plain IP-Adapter FaceID (h94's <c>MLPProjModel</c>): a 2-layer MLP from the L2-normalized 512-d InsightFace ArcFace identity embedding to <c>num_tokens × cross_attention_dim</c> — <c>Linear(512 → 1024) → GELU → Linear(1024 → tokens·crossDim)</c>, reshaped to <c>[B, numTokens, crossAttnDim]</c> and finished with a LayerNorm over the per-token dim. Checkpoint keys (h94/IP-Adapter-FaceID <c>.bin</c>, flattened): <c>image_proj.proj.0.weight/bias</c>, <c>image_proj.proj.2.weight/bias</c>, <c>image_proj.norm.weight/bias</c>. The FaceID-Plus variants (<c>ProjPlusModel</c>, which additionally mix CLIP-Vision features through a perceiver resampler) are a different module and are NOT handled here. Math is CPU-side like <see cref="IpAdapterStandardProjection"/> — the largest case (1024 → 8192, batch 1) is a few MFLOPs, dwarfed by everything around it. GELU uses the exact erf form matching <c>torch.nn.GELU</c>'s default.</summary>
public sealed unsafe class IpAdapterFaceIdProjection : IIpAdapterImageProjection
{
    private readonly int _crossAttnDim;
    private readonly int _numTokens;

    private Tensor? _proj0Weight; // [hidden, idDim]
    private Tensor? _proj0Bias;   // [hidden]
    private Tensor? _proj2Weight; // [numTokens * crossAttnDim, hidden]
    private Tensor? _proj2Bias;   // [numTokens * crossAttnDim]
    private Tensor? _normWeight;  // [crossAttnDim]
    private Tensor? _normBias;    // [crossAttnDim]

    public int NumTokens => _numTokens;

    public IpAdapterFaceIdProjection(int crossAttnDim, int numTokens)
    {
        _crossAttnDim = crossAttnDim;
        _numTokens = numTokens;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "image_proj")
    {
        _proj0Weight = TensorCasts.EnsureF32(weights[$"{prefix}.proj.0.weight"]);
        _proj0Bias = TensorCasts.EnsureF32(weights[$"{prefix}.proj.0.bias"]);
        _proj2Weight = TensorCasts.EnsureF32(weights[$"{prefix}.proj.2.weight"]);
        _proj2Bias = TensorCasts.EnsureF32(weights[$"{prefix}.proj.2.bias"]);
        _normWeight = TensorCasts.EnsureF32(weights[$"{prefix}.norm.weight"]);
        _normBias = TensorCasts.EnsureF32(weights[$"{prefix}.norm.bias"]);
        long expected = (long)_numTokens * _crossAttnDim;
        if (_proj2Weight.Shape[0] != expected)
        {
            throw new InvalidOperationException(
                $"FaceID projection shape mismatch: proj.2.weight rows = {_proj2Weight.Shape[0]}, expected numTokens×crossAttnDim = {_numTokens}×{_crossAttnDim} = {expected}.");
        }
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_proj0Weight is not null) yield return _proj0Weight;
        if (_proj0Bias is not null) yield return _proj0Bias;
        if (_proj2Weight is not null) yield return _proj2Weight;
        if (_proj2Bias is not null) yield return _proj2Bias;
        if (_normWeight is not null) yield return _normWeight;
        if (_normBias is not null) yield return _normBias;
    }

    /// <summary>Projects the L2-normalized ArcFace identity embedding (<c>[B, 512]</c>) into <c>[B, numTokens, crossAttnDim]</c> image-prompt tokens.</summary>
    public Tensor Forward(IBackend backend, Tensor faceEmbeds)
    {
        if (_proj0Weight is null)
            throw new InvalidOperationException("FaceID projection weights not loaded.");
        if (faceEmbeds.Shape.Rank != 2)
            throw new ArgumentException($"faceEmbeds must be [B, idDim]; got {faceEmbeds.Shape}.", nameof(faceEmbeds));
        int batch = (int)faceEmbeds.Shape[0];
        int idDim = (int)faceEmbeds.Shape[1];
        int hidden = (int)_proj0Weight.Shape[0];
        int outDim = _numTokens * _crossAttnDim;
        if (idDim != (int)_proj0Weight.Shape[1])
        {
            throw new ArgumentException(
                $"faceEmbeds dim {idDim} does not match the checkpoint's id-embedding dim {_proj0Weight.Shape[1]}.", nameof(faceEmbeds));
        }

        Tensor hiddenT = new(new TensorShape(batch, hidden), DType.F32);
        LinearHost(faceEmbeds, _proj0Weight, _proj0Bias!, hiddenT, batch, idDim, hidden);
        backend.GeluErf(hiddenT, hiddenT);

        // Row-major [B, outDim] shares the byte layout of [B, numTokens, crossAttnDim].
        Tensor output = new(new TensorShape(batch, _numTokens, _crossAttnDim), DType.F32);
        LinearHost(hiddenT, _proj2Weight!, _proj2Bias!, output, batch, hidden, outDim);
        hiddenT.Dispose();

        IpAdapterMath.LayerNormInPlace(output, _normWeight!, _normBias!,
            rows: batch * _numTokens, dim: _crossAttnDim, eps: 1e-5f);
        return output;
    }

    private static void LinearHost(Tensor input, Tensor weight, Tensor bias, Tensor output, int batch, int inDim, int outDim)
    {
        float* xp = (float*)input.DataPointer;
        float* wp = (float*)weight.DataPointer;
        float* bp = (float*)bias.DataPointer;
        float* op = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int o = 0; o < outDim; o++)
            {
                float sum = bp[o];
                int wRow = o * inDim;
                int xOff = b * inDim;
                for (int i = 0; i < inDim; i++)
                {
                    sum += xp[xOff + i] * wp[wRow + i];
                }
                op[b * outDim + o] = sum;
            }
        }
    }
}
