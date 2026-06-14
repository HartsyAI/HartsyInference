using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Adapters;

/// <summary>IP-Adapter (Image Prompt Adapter): bolts a parallel image-conditioning path onto a base UNet's cross-attention layers. The CLIP-Vision encoding of a reference image is projected into N image-prompt tokens (4 for standard, 16 for Plus), and at every cross-attention block in the UNet a separate <c>K_ip</c> / <c>V_ip</c> linear projection produces image-conditioned key/value pairs that, together with the standard text K/V, drive an additional attention computation. The two attention outputs (text-attn + scale × image-attn) are summed before the cross-attention's output projection.
///
/// <para>The adapter holds two pieces of state: an <see cref="IIpAdapterImageProjection"/> that maps CLIP-Vision output to image-prompt tokens (different module for standard vs Plus / Plus-Face / Full-Face), and a flat list of per-cross-attention-layer <c>(to_k_ip, to_v_ip)</c> weight pairs keyed by integer index (<c>0</c>, <c>1</c>, …, matching the UNet's cross-attention enumeration order — diffusers' <c>attn_processors</c> dict iteration order, which our <see cref="HartsyInference.Diffusion.Models.Denoisers.UNet"/> matches).</para>
///
/// <para><b>Currently supported:</b> SDXL standard + Plus + Plus-Face (same architecture as Plus, just different training).</para>
///
/// <para><b>Not yet implemented:</b> SD 1.5 IPA (mechanical clone — same architecture, different cross-attn dim 768 and base model wiring not yet plumbed); FaceID variants (use InsightFace ArcFace embeddings instead of CLIP-Vision; different image encoder + projection); Flux IPA (DiT cross-attention has a different shape).</para></summary>
public sealed unsafe class IpAdapter : IDisposable
{
    private readonly IpAdapterConfig _config;
    private readonly IIpAdapterImageProjection _imageProjection;
    private int _disposed;

    /// <summary>Per-cross-attention-layer <c>to_k_ip</c> projection weights. Shape <c>[crossAttnDim, projOutputDim]</c> — <c>projOutputDim</c> equals the image-prompt token dim, which the projection sets to <c>cross_attention_dim</c> for standard and <c>output_dim</c> for Plus's <c>proj_out</c>. Index in the array matches the UNet's natural cross-attention enumeration order.</summary>
    private Tensor[] _toKIpWeights = [];
    private Tensor[] _toVIpWeights = [];

    /// <summary>The number of cross-attention layers this adapter expects in the base UNet, derived from the count of <c>ip_adapter.{i}.*</c> keys in the checkpoint.</summary>
    public int CrossAttentionLayerCount => _toKIpWeights.Length;

    /// <summary>How many image-prompt tokens the projection emits (4 standard, 16 Plus).</summary>
    public int NumImageTokens => _imageProjection.NumTokens;

    public IpAdapterConfig Config => _config;

    /// <summary>The image projection (standard MLP or Plus Resampler) — exposed for callers that want to drive it directly. Pipelines normally use <see cref="ProjectImage"/> instead.</summary>
    public IIpAdapterImageProjection ImageProjection => _imageProjection;

    /// <summary>Creates an IP-Adapter from configuration. Picks the standard or Plus projection module based on <see cref="IpAdapterConfig.IsPlus"/>; the per-cross-attn-layer K/V tensors are populated by <see cref="LoadWeights"/>.</summary>
    public IpAdapter(IpAdapterConfig config)
    {
        if (config.BaseModel == IpAdapterBaseModel.Flux)
        {
            throw new NotSupportedException(
                "Flux IP-Adapter uses a DiT-based cross-attention layout and isn't handled by this adapter. " +
                "A separate Flux-specific IP-Adapter class is needed.");
        }
        if (config.IsFaceId)
        {
            throw new NotSupportedException(
                "IP-Adapter FaceID uses InsightFace ArcFace embeddings instead of CLIP-Vision and ships a " +
                "different image encoder. Not yet supported by this implementation.");
        }
        _config = config;
        _imageProjection = config.IsPlus
            ? BuildPlusResampler(config)
            : new IpAdapterStandardProjection(config.CrossAttentionDim, config.NumImageTokens);
    }

    /// <summary>Builds the Plus Resampler with the canonical hyperparameters for the detected base model. SDXL Plus uses <c>depth=4, hidden=1024, heads=12, head_dim=64</c> (so inner=768) — these are the values the released SDXL Plus checkpoints were trained with.</summary>
    private static IpAdapterPlusResampler BuildPlusResampler(IpAdapterConfig config)
    {
        // For SDXL Plus: 4 layers, hidden=1024, heads=12, head_dim=64 → inner_dim=768.
        // Embedding dim = 1280 (CLIP-Vision-H hidden), output_dim = 2048 (SDXL cross_attn_dim).
        return new IpAdapterPlusResampler(
            embeddingDim: 1280,           // CLIP-Vision-H/14 hidden_size
            hiddenDim: 1024,              // Resampler working dim
            numHeads: 12,                 // PerceiverAttention heads
            headDim: 64,                  // Per-head dim → inner_dim = 12 * 64 = 768
            numTokens: config.NumImageTokens,
            outputDim: config.CrossAttentionDim,
            depth: 4,                     // 4 alternating attention + FFN layers
            ffMultiplier: 4);
    }

    /// <summary>Loads the IP-Adapter checkpoint. Drives the image projection's load and counts <c>ip_adapter.{i}.to_k_ip.weight</c> entries to discover the cross-attention layer count, then loads each per-layer K/V pair. Throws if the count doesn't match what the standard SDXL UNet has (70 cross-attn layers).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        // 1. Image projection
        _imageProjection.LoadWeights(weights);

        // 2. Discover layer count by counting ip_adapter.{i}.to_k_ip.weight keys.
        //    Keys are integer indices but stored as strings — sort numerically to preserve order.
        const string ipAdapterPrefix = "ip_adapter.";
        const string toKSuffix = ".to_k_ip.weight";
        SortedDictionary<int, Tensor> kByIdx = new();
        SortedDictionary<int, Tensor> vByIdx = new();
        foreach ((string key, Tensor tensor) in weights)
        {
            if (!key.StartsWith(ipAdapterPrefix, StringComparison.Ordinal)) continue;
            string rest = key[ipAdapterPrefix.Length..];
            int dotIdx = rest.IndexOf('.');
            if (dotIdx < 0) continue;
            string idxStr = rest[..dotIdx];
            if (!int.TryParse(idxStr, out int layerIdx)) continue;
            string suffix = rest[dotIdx..];
            if (suffix == toKSuffix)
            {
                kByIdx[layerIdx] = EnsureF32(tensor);
            }
            else if (suffix == ".to_v_ip.weight")
            {
                vByIdx[layerIdx] = EnsureF32(tensor);
            }
        }

        if (kByIdx.Count == 0)
        {
            throw new InvalidOperationException(
                "IP-Adapter checkpoint has no ip_adapter.{i}.to_k_ip.weight entries — invalid file or unsupported variant.");
        }
        if (kByIdx.Count != vByIdx.Count)
        {
            throw new InvalidOperationException(
                $"IP-Adapter checkpoint has mismatched K/V counts: {kByIdx.Count} K, {vByIdx.Count} V.");
        }

        int numLayers = kByIdx.Count;
        _toKIpWeights = new Tensor[numLayers];
        _toVIpWeights = new Tensor[numLayers];
        int i = 0;
        foreach ((int idx, Tensor k) in kByIdx)
        {
            _toKIpWeights[i] = k;
            if (!vByIdx.TryGetValue(idx, out Tensor? v))
            {
                throw new InvalidOperationException($"IP-Adapter has K but no V for layer index {idx}.");
            }
            _toVIpWeights[i] = v;
            i++;
        }
    }

    /// <summary>Project a CLIP-Vision output tensor into image-prompt tokens. For standard the input is the visual_projection CLS embed (<c>[B, projDim]</c>); for Plus the input is the penultimate layer's full hidden states (<c>[B, seqLen, hiddenSize]</c>). Returns <c>[B, numTokens, crossAttnDim]</c>.</summary>
    public Tensor ProjectImage(IBackend backend, Tensor visionInput)
    {
        ThrowIfDisposed();
        return _imageProjection.Forward(backend, visionInput);
    }

    /// <summary>The K_ip projection weight for cross-attention layer <paramref name="layerIdx"/> (0-based). Shape <c>[crossAttnDim, crossAttnDim]</c> — projects image-prompt tokens (per-token dim = crossAttnDim) into the same K space the UNet's text K projection produces.</summary>
    public Tensor GetToKIpWeight(int layerIdx) => _toKIpWeights[layerIdx];

    /// <summary>The V_ip projection weight for cross-attention layer <paramref name="layerIdx"/>.</summary>
    public Tensor GetToVIpWeight(int layerIdx) => _toVIpWeights[layerIdx];

    /// <summary>Yields all weight tensors for GPU preloading / freeing.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor w in _imageProjection.EnumerateWeights()) yield return w;
        foreach (Tensor w in _toKIpWeights) if (w is not null) yield return w;
        foreach (Tensor w in _toVIpWeights) if (w is not null) yield return w;
    }

    private static Tensor EnsureF32(Tensor t) => t.DType != DType.F32 ? t.CastTo(DType.F32) : t;

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
    }

    /// <summary>Releases tensor references. Tensors are owned by the safetensors loader (mmap-backed) so disposal is best-effort metadata only.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            Array.Clear(_toKIpWeights);
            Array.Clear(_toVIpWeights);
        }
        GC.SuppressFinalize(this);
    }
}
