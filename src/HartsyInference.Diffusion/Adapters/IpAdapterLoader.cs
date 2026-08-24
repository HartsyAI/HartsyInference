using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Diffusion.Adapters;

/// <summary>Opens an IP-Adapter checkpoint (safetensors, or the torch-pickle <c>.bin</c> the FaceID releases ship), detects the variant (Standard / Plus / FaceID) and base model from key signatures and tensor shapes, and returns the parsed weight dictionary plus a derived config. Pickle checkpoints are flattened from their nested <c>{image_proj: {…}, ip_adapter: {…}}</c> layout into the same dotted keys safetensors files use.</summary>
public static class IpAdapterLoader
{
    /// <summary>Loads an IP-Adapter checkpoint with auto-detection. Supports <c>.safetensors</c> plus torch <c>.bin</c>/<c>.pt</c>/<c>.pth</c> pickle containers (the h94 FaceID release format).</summary>
    public static IpAdapterFile Load(string filePath)
    {
        bool isPickle = filePath.EndsWith(".bin", StringComparison.OrdinalIgnoreCase)
            || filePath.EndsWith(".pt", StringComparison.OrdinalIgnoreCase)
            || filePath.EndsWith(".pth", StringComparison.OrdinalIgnoreCase);
        IDisposable loader;
        Dictionary<string, Tensor> weights;
        if (isPickle)
        {
            PytorchPickleLoader pickle = new();
            loader = pickle;
            try
            {
                pickle.Load(filePath, recursiveFlatten: true);
                weights = new Dictionary<string, Tensor>(pickle.GetAllTensors());
            }
            catch
            {
                pickle.Dispose();
                throw;
            }
        }
        else
        {
            SafeTensorsLoader st = new();
            loader = st;
            try
            {
                st.Load(filePath);
                weights = st.GetAllTensors();
            }
            catch
            {
                st.Dispose();
                throw;
            }
        }

        try
        {
            (IpAdapterBaseModel baseModel, bool isPlus, bool isFaceId, bool isFaceIdV2) = Detect(filePath, weights);
            IpAdapterConfig config = BuildConfig(baseModel, isPlus, isFaceId, weights, isFaceIdV2);

            Logs.Info($"Loaded IP-Adapter '{Path.GetFileName(filePath)}' (base={baseModel}, plus={isPlus}, faceId={isFaceId}, faceIdV2={isFaceIdV2}, tensors={weights.Count}).");

            IpAdapterFile file = new()
            {
                FilePath = filePath,
                BaseModel = baseModel,
                Config = config,
                Weights = weights,
            };
            file.AttachLoader(loader);
            return file;
        }
        catch
        {
            loader.Dispose();
            throw;
        }
    }

    /// <summary>Detects (baseModel, isPlus, isFaceId, isFaceIdV2) from the filename and weight-key signatures. Public so the detection logic is unit-testable against synthetic key sets.</summary>
    public static (IpAdapterBaseModel baseModel, bool isPlus, bool isFaceId, bool isFaceIdV2) Detect(string filePath, IReadOnlyDictionary<string, Tensor> weights)
    {
        string lowered = Path.GetFileNameWithoutExtension(filePath).ToLowerInvariant();
        // FaceID signature: the checkpoint carries the embedded rank-128 LoRA half (ip_adapter.{i}.to_q_lora.*)
        // alongside the K/V projections, and/or the FaceID MLP proj (image_proj.proj.0.weight).
        bool isFaceId = lowered.Contains("faceid") || lowered.Contains("face_id")
            || weights.Keys.Any(k => k.Contains("id_proj", StringComparison.Ordinal)
            || k.Contains("_lora.down.weight", StringComparison.Ordinal));
        // Plus signature = a resampler: proj_in / latents keys (standard Plus) or the FaceID-Plus perceiver.
        // Standard checkpoints ALSO carry image_proj.norm.* (the post-projection LayerNorm), so norm keys must
        // NOT imply Plus; for FaceID files "plus" in the filename is authoritative (ProjPlusModel).
        bool hasPerceiver = weights.Keys.Any(k => k.StartsWith("image_proj.perceiver_resampler.", StringComparison.Ordinal));
        bool isPlus = lowered.Contains("plus") || weights.ContainsKey("image_proj.proj_in.weight")
            || weights.ContainsKey("image_proj.latents") || hasPerceiver;
        // FaceID-Plus v2 is structurally identical to v1 (same ProjPlusModel keys), differing only in
        // training (shortcut path enabled) — the filename is the only signal (h94 ships "plusv2").
        bool isFaceIdV2 = isFaceId && isPlus
            && (lowered.Contains("plusv2") || lowered.Contains("plus-v2") || lowered.Contains("plus_v2"));

        IpAdapterBaseModel baseModel = DetectBaseModel(filePath, weights);
        return (baseModel, isPlus, isFaceId, isFaceIdV2);
    }

    private static IpAdapterBaseModel DetectBaseModel(string filePath, IReadOnlyDictionary<string, Tensor> weights)
    {
        string lowered = Path.GetFileNameWithoutExtension(filePath).ToLowerInvariant();
        if (lowered.Contains("sdxl") || lowered.Contains("xl_")) return IpAdapterBaseModel.Sdxl;
        if (lowered.Contains("flux")) return IpAdapterBaseModel.Flux;
        if (lowered.Contains("sd15") || lowered.Contains("sd-1") || lowered.Contains("sd_1")) return IpAdapterBaseModel.Sd15;

        Tensor? probe = null;
        foreach ((string key, Tensor tensor) in weights)
        {
            if (key.EndsWith(".to_k_ip.weight", StringComparison.Ordinal) ||
                key.EndsWith(".to_v_ip.weight", StringComparison.Ordinal))
            {
                probe = tensor;
                break;
            }
        }

        if (probe is not null && probe.Shape.Rank == 2)
        {
            long crossDim = probe.Shape[1];
            if (crossDim >= 4096) return IpAdapterBaseModel.Flux;
            if (crossDim >= 1024) return IpAdapterBaseModel.Sdxl;
            return IpAdapterBaseModel.Sd15;
        }

        IEnumerable<string> sample = weights.Keys.Take(8);
        throw new HartsyInferenceException($"Could not detect IP-Adapter base model from key signatures. Sample keys: {string.Join(", ", sample)}");
    }

    /// <summary>Derives the variant config from the detected flags and projection tensor shapes. Public so the shape-derivation logic is unit-testable against synthetic weight dicts.</summary>
    public static IpAdapterConfig BuildConfig(IpAdapterBaseModel baseModel, bool isPlus, bool isFaceId, IReadOnlyDictionary<string, Tensor> weights, bool isFaceIdV2 = false)
    {
        int crossDim = baseModel switch
        {
            IpAdapterBaseModel.Sd15 => 768,
            IpAdapterBaseModel.Sdxl => 2048,
            IpAdapterBaseModel.Flux => 4096,
            _ => 768,
        };
        int numTokens = isPlus ? 16 : 4;
        int imageDim = 1024;
        int clipEmbeddingDim = 1280;
        if (isFaceId)
        {
            // FaceID MLP: proj.0 [hidden, idDim] gives the ArcFace embed dim (512); proj.2
            // [numTokens*crossDim, hidden] gives the token count (4 plain + Plus, 16 portrait).
            imageDim = 512;
            numTokens = 4;
            if (weights.TryGetValue("image_proj.proj.0.weight", out Tensor? proj0) && proj0.Shape.Rank == 2)
            {
                imageDim = (int)proj0.Shape[1];
            }
            if (weights.TryGetValue("image_proj.proj.2.weight", out Tensor? proj2) && proj2.Shape.Rank == 2)
            {
                numTokens = (int)(proj2.Shape[0] / crossDim);
            }
            // FaceID-Plus perceiver: proj_in [crossDim, clipDim] gives the CLIP-Vision hidden size (1280 ViT-H).
            if (weights.TryGetValue("image_proj.perceiver_resampler.proj_in.weight", out Tensor? projIn) && projIn.Shape.Rank == 2)
            {
                clipEmbeddingDim = (int)projIn.Shape[1];
            }
        }
        // The standard projection stores its Linear as image_proj.proj.weight [numTokens*crossDim, imageDim];
        // some converted checkpoints flatten it to image_proj.weight.
        else if ((weights.TryGetValue("image_proj.proj.weight", out Tensor? proj)
            || weights.TryGetValue("image_proj.weight", out proj)) && proj.Shape.Rank == 2)
        {
            imageDim = (int)proj.Shape[1];
        }
        return new IpAdapterConfig
        {
            BaseModel = baseModel,
            ClipImageModel = isFaceId ? isPlus ? "InsightFace ArcFace + ViT-H/14" : "InsightFace ArcFace" : "ViT-H/14",
            ImageEmbeddingDim = imageDim,
            NumImageTokens = numTokens,
            CrossAttentionDim = crossDim,
            IsPlus = isPlus,
            IsFaceId = isFaceId,
            IsFaceIdV2 = isFaceIdV2,
            ClipEmbeddingDim = clipEmbeddingDim,
        };
    }
}
