using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Vae;

/// <summary>Self-attention layer for the VAE mid-block. Single-head attention (heads=1, head_dim=channels) with GroupNorm and residual connection.</summary>
public sealed class VaeAttention
{
    private readonly int _channels;
    private readonly int _normGroups;
    private readonly float _normEps;

    // GroupNorm before attention projections
    private Tensor? _groupNormWeight;
    private Tensor? _groupNormBias;

    // Q, K, V projections (linear, implemented as 1x1 conv or matmul)
    private Tensor? _toQWeight;
    private Tensor? _toQBias;
    private Tensor? _toKWeight;
    private Tensor? _toKBias;
    private Tensor? _toVWeight;
    private Tensor? _toVBias;

    // Output projection
    private Tensor? _toOutWeight;
    private Tensor? _toOutBias;

    /// <summary>Creates a VAE attention layer with the specified channel count.</summary>
    public VaeAttention(int channels, int normGroups = 32, float normEps = 1e-6f)
    {
        _channels = channels;
        _normGroups = normGroups;
        _normEps = normEps;
    }

    /// <summary>Loads weights from named tensors using the given prefix (e.g., "decoder.mid_block.attentions.0").</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _groupNormWeight = weights[$"{prefix}.group_norm.weight"];
        _groupNormBias = weights[$"{prefix}.group_norm.bias"];
        _toQWeight = weights[$"{prefix}.to_q.weight"];
        _toQBias = weights[$"{prefix}.to_q.bias"];
        _toKWeight = weights[$"{prefix}.to_k.weight"];
        _toKBias = weights[$"{prefix}.to_k.bias"];
        _toVWeight = weights[$"{prefix}.to_v.weight"];
        _toVBias = weights[$"{prefix}.to_v.bias"];
        _toOutWeight = weights[$"{prefix}.to_out.0.weight"];
        _toOutBias = weights[$"{prefix}.to_out.0.bias"];
    }

    /// <summary>Enumerates all weight tensors held by this module.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_groupNormWeight is not null) yield return _groupNormWeight;
        if (_groupNormBias is not null) yield return _groupNormBias;
        if (_toQWeight is not null) yield return _toQWeight;
        if (_toQBias is not null) yield return _toQBias;
        if (_toKWeight is not null) yield return _toKWeight;
        if (_toKBias is not null) yield return _toKBias;
        if (_toVWeight is not null) yield return _toVWeight;
        if (_toVBias is not null) yield return _toVBias;
        if (_toOutWeight is not null) yield return _toOutWeight;
        if (_toOutBias is not null) yield return _toOutBias;
    }

    /// <summary>Forward pass: input [B, C, H, W] → output [B, C, H, W] with residual connection.</summary>
    public Tensor Forward(IBackend backend, Tensor input)
    {
        int batch = (int)input.Shape[0];
        int channels = (int)input.Shape[1];
        int height = (int)input.Shape[2];
        int width = (int)input.Shape[3];
        int seqLen = height * width;

        // GroupNorm
        TensorShape spatialShape = new TensorShape(batch, channels, height, width);
        Tensor normed = new Tensor(spatialShape, DType.F32);
        backend.GroupNorm(normed, input, _groupNormWeight!, _groupNormBias!, _normGroups, _normEps);

        // Reshape to [B, C, seqLen] then transpose to [B, seqLen, C] for attention
        // Q, K, V projections: weight is [C, C], we do matmul [B, seqLen, C] @ [C, C]^T
        TensorShape seqShape = new TensorShape(batch, seqLen, channels);
        Tensor normedSeq = normed.Reshape(new TensorShape(batch, channels, seqLen));

        // Transpose [B, C, seqLen] → [B, seqLen, C] by copying
        Tensor normedTransposed = new Tensor(seqShape, DType.F32);
        TransposeBCtoBC(backend, normedTransposed, normedSeq, batch, channels, seqLen);
        normed.Dispose();

        // Project Q, K, V via batched matmul: [B, seqLen, C] @ [C, C]^T = [B, seqLen, C]
        Tensor query = ProjectLinear(backend, normedTransposed, _toQWeight!, _toQBias!, batch, seqLen, channels);
        Tensor key = ProjectLinear(backend, normedTransposed, _toKWeight!, _toKBias!, batch, seqLen, channels);
        Tensor value = ProjectLinear(backend, normedTransposed, _toVWeight!, _toVBias!, batch, seqLen, channels);
        normedTransposed.Dispose();

        // Reshape to 4D for single-head attention: [B, seqLen, C] → [B, 1, seqLen, C]
        TensorShape attn4DShape = new TensorShape(batch, 1, seqLen, channels);
        Tensor query4D = query.Reshape(attn4DShape);
        Tensor key4D = key.Reshape(attn4DShape);
        Tensor value4D = value.Reshape(attn4DShape);

        float scale = 1.0f / MathF.Sqrt(channels);
        Tensor attnOut4D = new Tensor(attn4DShape, DType.F32);
        backend.ScaledDotProductAttention(attnOut4D, query4D, key4D, value4D, null, scale);
        query.Dispose();
        key.Dispose();
        value.Dispose();

        // Reshape back to 3D: [B, 1, seqLen, C] → [B, seqLen, C]
        Tensor attnOut = attnOut4D.Reshape(seqShape);

        // Output projection: [B, seqLen, C] @ [C, C]^T = [B, seqLen, C]
        Tensor projected = ProjectLinear(backend, attnOut, _toOutWeight!, _toOutBias!, batch, seqLen, channels);
        attnOut.Dispose();

        // Transpose back [B, seqLen, C] → [B, C, seqLen] → reshape to [B, C, H, W]
        Tensor projectedChannelFirst = new Tensor(new TensorShape(batch, channels, seqLen), DType.F32);
        TransposeBCtoBC(backend, projectedChannelFirst, projected, batch, seqLen, channels);
        projected.Dispose();

        Tensor projectedSpatial = projectedChannelFirst.Reshape(spatialShape);

        // Residual connection
        Tensor output = new Tensor(spatialShape, DType.F32);
        backend.Add(output, input, projectedSpatial);
        projectedChannelFirst.Dispose();

        return output;
    }

    /// <summary>Linear projection: output[b] = input[b] @ weight^T + bias for each batch.</summary>
    private static Tensor ProjectLinear(IBackend backend, Tensor input, Tensor weight, Tensor bias, int batch, int seqLen, int channels)
    {
        TensorShape outShape = new TensorShape(batch, seqLen, channels);
        Tensor output = new Tensor(outShape, DType.F32);

        // backend.Linear computes output = input @ weight^T + bias on GPU
        // Weight transpose and bias addition are handled by cuBLAS SGEMM (OP_T) + GPU kernel
        backend.Linear(output, input, weight, bias);

        return output;
    }

    /// <summary>Transposes [B, dim1, dim2] → [B, dim2, dim1] via element copy.</summary>
    private static unsafe void TransposeBCtoBC(IBackend backend, Tensor output, Tensor input, int batch, int dim1, int dim2)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            float* batchIn = inPtr + b * dim1 * dim2;
            float* batchOut = outPtr + b * dim2 * dim1;
            for (int i = 0; i < dim1; i++)
            {
                for (int j = 0; j < dim2; j++)
                {
                    batchOut[j * dim1 + i] = batchIn[i * dim2 + j];
                }
            }
        }
    }

}
