using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.SafeTensors;

namespace HartsyInference.Diffusion.Adapters;

/// <summary>Opens a ControlNet safetensors file, detects the base model architecture from key signatures and tensor shapes, and returns the parsed weight dictionary plus an auto-derived config. The actual <see cref="ControlNet.LoadWeights"/> step is the consumer's responsibility once that path is implemented; this loader handles file parsing and detection only.</summary>
public static class ControlNetLoader
{
    /// <summary>Loads a ControlNet safetensors file with auto-detection. Throws <see cref="HartsyInferenceException"/> if the base model cannot be inferred.</summary>
    public static ControlNetFile Load(string filePath, ControlNetMode? modeOverride = null)
    {
        SafeTensorsLoader loader = new();
        try
        {
            loader.Load(filePath);

            ControlNetBaseModel baseModel = DetectBaseModel(loader.Descriptors);
            ControlNetMode mode = modeOverride ?? DetectMode(filePath);
            ControlNetConfig config = baseModel switch
            {
                ControlNetBaseModel.Sd15 => ControlNetConfig.Sd15(mode),
                ControlNetBaseModel.Sdxl => ControlNetConfig.Sdxl(mode),
                ControlNetBaseModel.Flux => new ControlNetConfig
                {
                    BaseModel = ControlNetBaseModel.Flux,
                    Mode = mode,
                    BlockOutChannels = [3072, 3072, 3072],
                    CrossAttentionDim = 4096,
                },
                _ => throw new HartsyInferenceException($"Unsupported ControlNet base model {baseModel} for '{filePath}'."),
            };

            Dictionary<string, Tensor> weights = loader.GetAllTensors();

            Logs.Info($"Loaded ControlNet '{Path.GetFileName(filePath)}' (base={baseModel}, mode={mode}, tensors={weights.Count}).");

            ControlNetFile file = new()
            {
                FilePath = filePath,
                BaseModel = baseModel,
                Mode = mode,
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

    private static ControlNetBaseModel DetectBaseModel(IReadOnlyDictionary<string, SafeTensorDescriptor> descriptors)
    {
        if (descriptors.ContainsKey("controlnet_blocks.0.weight") || descriptors.Keys.Any(k => k.StartsWith("transformer_blocks.", StringComparison.Ordinal)))
            return ControlNetBaseModel.Flux;

        if (descriptors.TryGetValue("down_blocks.0.attentions.0.proj_in.weight", out SafeTensorDescriptor? projIn))
        {
            long inDim = projIn.Shape.Rank > 1 ? projIn.Shape[1] : 0;
            return inDim >= 1024 ? ControlNetBaseModel.Sdxl : ControlNetBaseModel.Sd15;
        }

        if (descriptors.TryGetValue("class_embedding.linear_1.weight", out _) ||
            descriptors.ContainsKey("add_embedding.linear_1.weight"))
            return ControlNetBaseModel.Sdxl;

        if (descriptors.Keys.Any(k => k.StartsWith("input_blocks.", StringComparison.Ordinal) || k.StartsWith("middle_block.", StringComparison.Ordinal)))
            return ControlNetBaseModel.Sd15;

        IEnumerable<string> sample = descriptors.Keys.Take(8);
        throw new HartsyInferenceException($"Could not detect ControlNet base model from key signatures. Sample keys: {string.Join(", ", sample)}");
    }

    private static ControlNetMode DetectMode(string filePath)
    {
        string lowered = Path.GetFileNameWithoutExtension(filePath).ToLowerInvariant();
        if (lowered.Contains("canny")) return ControlNetMode.Canny;
        if (lowered.Contains("depth") || lowered.Contains("zoedepth") || lowered.Contains("midas")) return ControlNetMode.Depth;
        if (lowered.Contains("openpose") || lowered.Contains("pose")) return ControlNetMode.OpenPose;
        if (lowered.Contains("scribble")) return ControlNetMode.Scribble;
        if (lowered.Contains("tile")) return ControlNetMode.Tile;
        if (lowered.Contains("normal")) return ControlNetMode.Normal;
        if (lowered.Contains("seg")) return ControlNetMode.Segmentation;
        if (lowered.Contains("inpaint")) return ControlNetMode.Inpaint;
        if (lowered.Contains("lineart") || lowered.Contains("line_art")) return ControlNetMode.LineArt;
        if (lowered.Contains("softedge") || lowered.Contains("hed") || lowered.Contains("pidi")) return ControlNetMode.SoftEdge;
        return ControlNetMode.Depth;
    }
}
