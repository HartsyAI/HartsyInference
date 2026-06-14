using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

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
        int seqLen = (int)(input.Shape[2] * input.Shape[3]);
        return ForwardCore(backend, input, batch, channels, seqLen, null);
    }

    /// <summary>Forward pass over a video tensor <c>[B, C, T, H, W]</c>: attention over all <c>T·H·W</c>
    /// tokens with an optional frame-causal additive mask (tokens attend only to frames ≤ their own —
    /// HunyuanVideo VAE mid-block, <c>prepare_causal_attention_mask</c>). GroupNorm statistics span the
    /// full clip, matching diffusers' channel-dim GroupNorm over the flattened sequence.</summary>
    public Tensor Forward3D(IBackend backend, Tensor input, bool frameCausal = true)
    {
        int batch = (int)input.Shape[0];
        int channels = (int)input.Shape[1];
        int frames = (int)input.Shape[2];
        int spatial = (int)(input.Shape[3] * input.Shape[4]);

        Tensor? mask = frameCausal && frames > 1 ? BuildFrameCausalMask(frames, spatial) : null;
        Tensor result = ForwardCore(backend, input, batch, channels, frames * spatial, mask);
        mask?.Dispose();
        return result;
    }

    /// <summary>Additive attention mask <c>[1, 1, S, S]</c> with 0 where key-frame ≤ query-frame, −inf elsewhere.</summary>
    private static unsafe Tensor BuildFrameCausalMask(int frames, int spatial)
    {
        long s = (long)frames * spatial;
        Tensor mask = new Tensor(new TensorShape([1L, 1, s, s]), DType.F32);
        float* p = (float*)mask.DataPointer;
        for (long qi = 0; qi < s; qi++)
        {
            long qFrame = qi / spatial;
            long row = qi * s;
            long allowed = (qFrame + 1) * spatial;
            for (long ki = 0; ki < allowed; ki++) p[row + ki] = 0f;
            for (long ki = allowed; ki < s; ki++) p[row + ki] = float.NegativeInfinity;
        }
        return mask;
    }

    /// <summary>Shared core: treats the input as <c>[B, C, seqLen]</c> (any trailing spatial dims) and returns a tensor with the input's shape.</summary>
    private Tensor ForwardCore(IBackend backend, Tensor input, int batch, int channels, int seqLen, Tensor? mask)
    {
        DType dtype = input.DType;

        // GroupNorm — viewed as [B, C, seqLen, 1] so group statistics span the whole sequence.
        TensorShape spatialShape = new TensorShape(batch, channels, seqLen, 1);
        Tensor input4d = input.Reshape(spatialShape);
        Tensor normed = new Tensor(spatialShape, dtype);
        backend.GroupNorm(normed, input4d, _groupNormWeight!, _groupNormBias!, _normGroups, _normEps);

        // Reshape to [B, C, seqLen] then transpose to [B, seqLen, C] for attention
        // Q, K, V projections: weight is [C, C], we do matmul [B, seqLen, C] @ [C, C]^T
        TensorShape seqShape = new TensorShape(batch, seqLen, channels);
        Tensor normedSeq = normed.Reshape(new TensorShape(batch, channels, seqLen));

        // Transpose [B, C, seqLen] → [B, seqLen, C] via backend
        Tensor normedTransposed = new Tensor(seqShape, dtype);
        backend.Transpose2D(normedTransposed, normedSeq, channels, seqLen);
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
        Tensor attnOut4D = new Tensor(attn4DShape, dtype);
        backend.ScaledDotProductAttention(attnOut4D, query4D, key4D, value4D, mask, scale);
        query.Dispose();
        key.Dispose();
        value.Dispose();

        // Reshape back to 3D: [B, 1, seqLen, C] → [B, seqLen, C]
        Tensor attnOut = attnOut4D.Reshape(seqShape);

        // Output projection: [B, seqLen, C] @ [C, C]^T = [B, seqLen, C]
        Tensor projected = ProjectLinear(backend, attnOut, _toOutWeight!, _toOutBias!, batch, seqLen, channels);
        attnOut.Dispose();

        // Transpose back [B, seqLen, C] → [B, C, seqLen] → reshape to the input's layout
        Tensor projectedChannelFirst = new Tensor(new TensorShape(batch, channels, seqLen), dtype);
        backend.Transpose2D(projectedChannelFirst, projected, seqLen, channels);
        projected.Dispose();

        Tensor projectedSpatial = projectedChannelFirst.Reshape(input.Shape);

        // Residual connection
        Tensor output = new Tensor(input.Shape, dtype);
        backend.Add(output, input, projectedSpatial);
        projectedChannelFirst.Dispose();

        return output;
    }

    /// <summary>Linear projection: output[b] = input[b] @ weight^T + bias for each batch.</summary>
    private static Tensor ProjectLinear(IBackend backend, Tensor input, Tensor weight, Tensor bias, int batch, int seqLen, int channels)
    {
        TensorShape outShape = new TensorShape(batch, seqLen, channels);
        Tensor output = new Tensor(outShape, input.DType);

        // backend.Linear computes output = input @ weight^T + bias on GPU
        // Weight transpose and bias addition are handled by cuBLAS SGEMM (OP_T) + GPU kernel
        backend.Linear(output, input, weight, bias);

        return output;
    }

}
