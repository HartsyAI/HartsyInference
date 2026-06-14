using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.SafeTensors;

namespace HartsyInference.Diffusion.Adapters;

/// <summary>Opens an IP-Adapter safetensors file, detects the variant (Standard/Plus/FaceID) and base model from key signatures and tensor shapes, and returns the parsed weight dictionary plus a derived config. The actual <see cref="IpAdapter.LoadWeights"/> step is the consumer's responsibility once that path is implemented.</summary>
public static class IpAdapterLoader
{
    /// <summary>Loads an IP-Adapter safetensors file with auto-detection.</summary>
    public static IpAdapterFile Load(string filePath)
    {
        SafeTensorsLoader loader = new();
        try
        {
            loader.Load(filePath);

            (IpAdapterBaseModel baseModel, bool isPlus, bool isFaceId) = Detect(filePath, loader.Descriptors);

            IpAdapterConfig config = BuildConfig(baseModel, isPlus, isFaceId, loader.Descriptors);
            Dictionary<string, Tensor> weights = loader.GetAllTensors();

            Logs.Info($"Loaded IP-Adapter '{Path.GetFileName(filePath)}' (base={baseModel}, plus={isPlus}, faceId={isFaceId}, tensors={weights.Count}).");

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

    private static (IpAdapterBaseModel baseModel, bool isPlus, bool isFaceId) Detect(string filePath, IReadOnlyDictionary<string, SafeTensorDescriptor> descriptors)
    {
        string lowered = Path.GetFileNameWithoutExtension(filePath).ToLowerInvariant();
        bool isFaceId = lowered.Contains("faceid") || lowered.Contains("face_id") || descriptors.Keys.Any(k => k.Contains("id_proj", StringComparison.Ordinal));
        bool isPlus = lowered.Contains("plus") || descriptors.ContainsKey("image_proj.norm.weight") || descriptors.ContainsKey("image_proj.proj_in.weight");

        IpAdapterBaseModel baseModel = DetectBaseModel(filePath, descriptors);
        return (baseModel, isPlus, isFaceId);
    }

    private static IpAdapterBaseModel DetectBaseModel(string filePath, IReadOnlyDictionary<string, SafeTensorDescriptor> descriptors)
    {
        string lowered = Path.GetFileNameWithoutExtension(filePath).ToLowerInvariant();
        if (lowered.Contains("sdxl") || lowered.Contains("xl_")) return IpAdapterBaseModel.Sdxl;
        if (lowered.Contains("flux")) return IpAdapterBaseModel.Flux;
        if (lowered.Contains("sd15") || lowered.Contains("sd-1") || lowered.Contains("sd_1")) return IpAdapterBaseModel.Sd15;

        SafeTensorDescriptor? probe = null;
        foreach (string key in descriptors.Keys)
        {
            if (key.EndsWith(".to_k_ip.weight", StringComparison.Ordinal) ||
                key.EndsWith(".to_v_ip.weight", StringComparison.Ordinal))
            {
                probe = descriptors[key];
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

        IEnumerable<string> sample = descriptors.Keys.Take(8);
        throw new HartsyInferenceException($"Could not detect IP-Adapter base model from key signatures. Sample keys: {string.Join(", ", sample)}");
    }

    private static IpAdapterConfig BuildConfig(IpAdapterBaseModel baseModel, bool isPlus, bool isFaceId, IReadOnlyDictionary<string, SafeTensorDescriptor> descriptors)
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
        if (descriptors.TryGetValue("image_proj.weight", out SafeTensorDescriptor? proj) && proj.Shape.Rank == 2)
        {
            imageDim = (int)proj.Shape[1];
        }
        return new IpAdapterConfig
        {
            BaseModel = baseModel,
            ClipImageModel = isFaceId ? "InsightFace ArcFace" : "ViT-H/14",
            ImageEmbeddingDim = imageDim,
            NumImageTokens = numTokens,
            CrossAttentionDim = crossDim,
            IsPlus = isPlus,
            IsFaceId = isFaceId,
        };
    }
}
