using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>Self-attention layer for the VAE mid-block. Single-head attention (heads=1, head_dim=channels) with GroupNorm and residual connection. Creates a VAE attention layer with the specified channel count.</summary>
public sealed class VaeAttention(int channels, int normGroups = 32, float normEps = 1e-6f)
{
    private readonly int _channels = channels;
    private readonly int _normGroups = normGroups;
    private readonly float _normEps = normEps;

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

    /// <summary>Shared core: treats the input as <c>[B, C, seqLen]</c> (any trailing spatial dims) and returns a tensor with the input's shape. No Reshape views anywhere — on CUDA a view of a GPU-produced activation forces a device→host sync + re-upload (~16 MB per view at SDXL VAE scale); every op here either reads the buffer flat (Transpose2D, Linear) or derives its layout from explicit dims, so outputs are allocated directly in the shape the next op needs.</summary>
    private Tensor ForwardCore(IBackend backend, Tensor input, int batch, int channels, int seqLen, Tensor? mask)
    {
        DType dtype = input.DType;

        // GroupNorm statistics span all trailing dims (spatial = seqLen), so the native layout is fine.
        Tensor normed = new Tensor(input.Shape, dtype);
        backend.GroupNorm(normed, input, _groupNormWeight!, _groupNormBias!, _normGroups, _normEps);

        // [B, C, seqLen] → [B, seqLen, C]
        TensorShape seqShape = new TensorShape(batch, seqLen, channels);
        Tensor normedTransposed = new Tensor(seqShape, dtype);
        backend.Transpose2D(normedTransposed, normed, channels, seqLen);
        normed.Dispose();

        // Q/K/V projections, allocated directly in the single-head 4D attention layout (Linear derives
        // its row count from element count / weight inDim, so the unit head dim is free).
        TensorShape attn4DShape = new TensorShape(batch, 1, seqLen, channels);
        Tensor query = new Tensor(attn4DShape, dtype);
        backend.Linear(query, normedTransposed, _toQWeight!, _toQBias!);
        Tensor key = new Tensor(attn4DShape, dtype);
        backend.Linear(key, normedTransposed, _toKWeight!, _toKBias!);
        Tensor value = new Tensor(attn4DShape, dtype);
        backend.Linear(value, normedTransposed, _toVWeight!, _toVBias!);
        normedTransposed.Dispose();

        float scale = 1.0f / MathF.Sqrt(channels);
        Tensor attnOut = new Tensor(attn4DShape, dtype);
        backend.ScaledDotProductAttention(attnOut, query, key, value, mask, scale);
        query.Dispose();
        key.Dispose();
        value.Dispose();

        // Output projection back to [B, seqLen, C], then channel-first in the input's native shape.
        Tensor projected = new Tensor(seqShape, dtype);
        backend.Linear(projected, attnOut, _toOutWeight!, _toOutBias!);
        attnOut.Dispose();

        Tensor projectedChannelFirst = new Tensor(input.Shape, dtype);
        backend.Transpose2D(projectedChannelFirst, projected, seqLen, channels);
        projected.Dispose();

        // Residual connection
        Tensor output = new Tensor(input.Shape, dtype);
        backend.Add(output, input, projectedChannelFirst);
        projectedChannelFirst.Dispose();

        return output;
    }

}
