using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Adapters;

/// <summary>IP-Adapter (Image Prompt Adapter): bolts a parallel image-conditioning path onto a base UNet's cross-attention layers. The CLIP-Vision encoding of a reference image is projected into N image-prompt tokens (4 for standard, 16 for Plus), and at every cross-attention block in the UNet a separate <c>K_ip</c> / <c>V_ip</c> linear projection produces image-conditioned key/value pairs that, together with the standard text K/V, drive an additional attention computation. The two attention outputs (text-attn + scale × image-attn) are summed before the cross-attention's output projection. The adapter holds two pieces of state: an <see cref="IIpAdapterImageProjection"/> that maps CLIP-Vision output to image-prompt tokens (different module for standard vs Plus / Plus-Face / Full-Face), and a flat list of per-cross-attention-layer <c>(to_k_ip, to_v_ip)</c> weight pairs keyed by integer index. The list order is diffusers' <c>attn_processors</c> dict iteration order, which is down → up → MID LAST (diffusers registers the empty <c>up_blocks</c> ModuleList before <c>mid_block</c>), NOT the forward-traversal down → mid → up order. <see cref="HartsyInference.Diffusion.Models.Denoisers.UNet"/>.Forward maps its traversal onto this checkpoint order internally. <b>Currently supported:</b> SDXL and SD 1.5, standard + Plus + Plus-Face (Plus-Face is the same architecture as Plus, just different training) + plain FaceID (ArcFace 512-d identity embedding → <see cref="IpAdapterFaceIdProjection"/> MLP; the K/V mechanism is identical, keyed at the cross-attn indices of the checkpoint's combined attn enumeration) + FaceID-Plus / Plus-v2 (<see cref="IpAdapterFaceIdPlusProjection"/>: the FaceID MLP tokens refined by a perceiver resampler over CLIP-Vision hidden states of the aligned face crop — project via the two-input <see cref="ProjectImage(IBackend, Tensor, Tensor, float)"/> overload). SD 1.5 is the same mechanism at cross-attn dim 768 over the SD1.5 UNet's 16 cross-attention sub-layers. NOTE: all FaceID checkpoints also ship a rank-128 LoRA over ALL UNet attention layers — that half is applied through the normal LoRA infrastructure (the released kohya <c>*_lora.safetensors</c> companion), not by this class. <b>Not yet implemented:</b> Flux IPA (DiT cross-attention has a different shape).</summary>
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
        _config = config;
        _imageProjection = config.IsFaceId
            ? config.IsPlus
                ? new IpAdapterFaceIdPlusProjection(config.CrossAttentionDim, config.NumImageTokens, config.ClipEmbeddingDim, useShortcut: config.IsFaceIdV2)
                : new IpAdapterFaceIdProjection(config.CrossAttentionDim, config.NumImageTokens)
            : config.IsPlus
                ? BuildPlusResampler(config)
                : new IpAdapterStandardProjection(config.CrossAttentionDim, config.NumImageTokens);
    }

    /// <summary>Builds the Plus Resampler with the canonical hyperparameters for the detected base model, matching the released tencent-ailab checkpoints: SD1.5 Plus uses <c>dim=768, heads=12</c> (proj_in <c>[768, 1280]</c>, to_q <c>[768, 768]</c> in ip-adapter-plus_sd15); SDXL Plus uses <c>dim=1280, heads=20</c> (proj_in <c>[1280, 1280]</c>, to_q <c>[1280, 1280]</c> in ip-adapter-plus_sdxl_vit-h). Both are <c>depth=4, head_dim=64, ff_mult=4</c> over CLIP-Vision-H penultimate hidden states (1280-dim), projected out to the base UNet's cross-attention dim.</summary>
    private static IpAdapterPlusResampler BuildPlusResampler(IpAdapterConfig config)
    {
        bool isSd15 = config.BaseModel == IpAdapterBaseModel.Sd15;
        int hiddenDim = isSd15 ? 768 : 1280;
        int numHeads = isSd15 ? 12 : 20;
        return new IpAdapterPlusResampler(
            embeddingDim: 1280,           // CLIP-Vision-H/14 hidden_size (penultimate hidden states)
            hiddenDim: hiddenDim,         // Resampler working dim (= inner_dim: to_q is square in the checkpoints)
            numHeads: numHeads,
            headDim: 64,
            numTokens: config.NumImageTokens,
            outputDim: config.CrossAttentionDim,
            depth: 4,                     // 4 alternating attention + FFN layers
            ffMultiplier: 4);
    }

    /// <summary>Loads the IP-Adapter checkpoint. Drives the image projection's load and counts <c>ip_adapter.{i}.to_k_ip.weight</c> entries to discover the cross-attention layer count (16 for SD1.5, 70 for SDXL), then loads each per-layer K/V pair. The base UNet validates the count against its own enumeration in <c>UNet.Forward</c>.</summary>
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
                kByIdx[layerIdx] = TensorCasts.EnsureF32(tensor);
            }
            else if (suffix == ".to_v_ip.weight")
            {
                vByIdx[layerIdx] = TensorCasts.EnsureF32(tensor);
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

    /// <summary>Project an image-encoder output tensor into image-prompt tokens. For standard the input is the CLIP visual_projection CLS embed (<c>[B, projDim]</c>); for Plus the penultimate layer's full hidden states (<c>[B, seqLen, hiddenSize]</c>); for FaceID the L2-normalized ArcFace identity embedding (<c>[B, 512]</c>). FaceID-Plus needs two inputs — use <see cref="ProjectImage(IBackend, Tensor, Tensor, float)"/> (this overload throws). Returns <c>[B, numTokens, crossAttnDim]</c>.</summary>
    public Tensor ProjectImage(IBackend backend, Tensor visionInput)
    {
        ThrowIfDisposed();
        return _imageProjection.Forward(backend, visionInput);
    }

    /// <summary>FaceID-Plus / Plus-v2 projection: mixes the L2-normalized ArcFace identity embedding (<c>[B, 512]</c>) with the CLIP-Vision penultimate hidden states of the aligned 224×224 face crop (<c>[B, seqLen, 1280]</c>). <paramref name="shortcutScale"/> is the v2 "FaceID V2 weight" (official default 1.0; ignored for v1 checkpoints). Returns <c>[B, numTokens, crossAttnDim]</c>. Throws for non-FaceID-Plus variants.</summary>
    public Tensor ProjectImage(IBackend backend, Tensor faceEmbeds, Tensor clipEmbeds, float shortcutScale = 1.0f)
    {
        ThrowIfDisposed();
        if (_imageProjection is not IpAdapterFaceIdPlusProjection plusProjection)
        {
            throw new InvalidOperationException(
                $"The two-input (faceEmbeds, clipEmbeds) projection is only valid for FaceID-Plus adapters; this adapter uses {_imageProjection.GetType().Name}.");
        }
        return plusProjection.Forward(backend, faceEmbeds, clipEmbeds, shortcutScale);
    }

    /// <summary>The K_ip projection weight for cross-attention layer <paramref name="layerIdx"/> (0-based, in the checkpoint's down → up → mid order). Shape <c>[layerInnerDim, crossAttnDim]</c> — projects image-prompt tokens (per-token dim = crossAttnDim) into the same K space the layer's text K projection produces (inner dim varies per layer: 640/1280 on SDXL, 320/640/1280 on SD1.5).</summary>
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
