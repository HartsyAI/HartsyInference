using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Adapters;

/// <summary>IP-Adapter (Image Prompt Adapter). Projects CLIP image embeddings into additional cross-attention key/value pairs that are added to the text-conditioned attention in each UNet/DiT block. Enables image-guided generation while preserving text control.</summary>
public sealed class IpAdapter : IDisposable
{
    private readonly IpAdapterConfig _config;
    private int _disposed;

    // Image projection: projects CLIP image embedding to cross-attention token space
    // Standard: Linear(image_dim, cross_attn_dim * num_tokens)
    // Plus: Perceiver Resampler with learnable queries
    private Tensor? _projWeight, _projBias;

    // Per-layer K/V projection weights for IP-Adapter cross-attention
    // One pair per cross-attention layer in the base model
    private Tensor?[] _toKIpWeights = [];
    private Tensor?[] _toVIpWeights = [];

    /// <summary>Creates an IP-Adapter from configuration.</summary>
    public IpAdapter(IpAdapterConfig config)
    {
        _config = config;
    }

    /// <summary>Loads IP-Adapter weights from named tensors.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        // TODO: Implement weight loading
        // IP-Adapter weights are stored separately from the base model
        // Key structure:
        // - image_proj.weight/bias (or image_proj.proj.weight for Plus)
        // - ip_adapter.{layer_idx}.to_k_ip.weight
        // - ip_adapter.{layer_idx}.to_v_ip.weight
        throw new NotImplementedException("IpAdapter.LoadWeights requires per-layer cross-attention weight mapping");
    }

    /// <summary>Projects a CLIP image embedding into cross-attention tokens.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="imageEmbedding">CLIP image embedding [B, imageDim] (CLS token or patch features).</param>
    /// <returns>Projected image tokens [B, numTokens, crossAttentionDim] for cross-attention injection.</returns>
    public Tensor ProjectImageEmbedding(IBackend backend, Tensor imageEmbedding)
    {
        // TODO: Implement projection:
        // Standard: Linear → reshape to [B, numTokens, crossAttnDim]
        // Plus: Perceiver resampler (cross-attention with learnable queries)
        throw new NotImplementedException("IpAdapter.ProjectImageEmbedding requires backend linear ops");
    }

    /// <summary>Computes additional K/V for a specific cross-attention layer from projected image tokens.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="imageTokens">Projected image tokens [B, numTokens, crossAttnDim].</param>
    /// <param name="layerIndex">Index of the cross-attention layer in the base model.</param>
    /// <returns>Additional key and value tensors to concatenate with text K/V.</returns>
    public (Tensor key, Tensor value) ComputeIpKV(IBackend backend, Tensor imageTokens, int layerIndex)
    {
        // TODO: Implement per-layer K/V projection
        // key = imageTokens @ to_k_ip[layerIndex]
        // value = imageTokens @ to_v_ip[layerIndex]
        throw new NotImplementedException("IpAdapter.ComputeIpKV requires backend linear ops");
    }

    /// <summary>Yields all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_projWeight is not null) yield return _projWeight;
        if (_projBias is not null) yield return _projBias;

        foreach (Tensor? w in _toKIpWeights)
            if (w is not null) yield return w;
        foreach (Tensor? v in _toVIpWeights)
            if (v is not null) yield return v;
    }

    /// <summary>Releases all tensor references.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _projWeight = null;
            _projBias = null;
            Array.Clear(_toKIpWeights);
            Array.Clear(_toVIpWeights);
        }
        GC.SuppressFinalize(this);
    }
}
