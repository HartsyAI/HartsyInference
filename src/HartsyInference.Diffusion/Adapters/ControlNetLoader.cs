using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Diffusion.Adapters;

/// <summary>Opens a ControlNet safetensors file, detects the base model architecture from key signatures and tensor shapes, and returns the parsed weight dictionary plus an auto-derived config. Accepts both diffusers-layout files and LDM-layout checkpoints (<c>control_model.*</c> keys — the original lllyasviel releases and fp16 repacks), converting the latter to diffusers keys via <see cref="ControlNetCheckpointConverter"/> so <see cref="ControlNet.LoadWeights"/> can consume <see cref="ControlNetFile.Weights"/> directly.</summary>
public static class ControlNetLoader
{
    /// <summary>Loads a ControlNet safetensors file with auto-detection. Throws <see cref="HartsyInferenceException"/> if the base model cannot be inferred.</summary>
    public static ControlNetFile Load(string filePath, ControlNetMode? modeOverride = null)
    {
        SafeTensorsLoader loader = new();
        try
        {
            loader.Load(filePath);

            bool isLdmLayout = ControlNetCheckpointConverter.IsLdmLayout(loader.Descriptors.Keys);
            ControlNetBaseModel baseModel = isLdmLayout ? DetectBaseModelLdm(loader.Descriptors)
                : DetectBaseModel(loader.Descriptors);
            ControlNetMode mode = modeOverride ?? DetectMode(filePath);
            FluxControlNetConfig? fluxConfig = baseModel == ControlNetBaseModel.Flux
                ? FluxControlNetConfig.FromDescriptors(loader.Descriptors) : null;
            QwenImageControlNetConfig? qwenConfig = baseModel == ControlNetBaseModel.QwenImage
                ? QwenImageControlNetConfig.FromDescriptors(loader.Descriptors) : null;
            int unionControlTypes = DetectUnionControlTypes(loader.Descriptors, baseModel, isLdmLayout);
            ControlNetConfig config = baseModel switch
            {
                ControlNetBaseModel.Sd15 => ControlNetConfig.Sd15(mode),
                ControlNetBaseModel.Sdxl => ControlNetConfig.Sdxl(mode) with { UnionControlTypeCount = unionControlTypes },
                ControlNetBaseModel.Flux => new ControlNetConfig
                {
                    BaseModel = ControlNetBaseModel.Flux,
                    Mode = mode,
                    BlockOutChannels = [fluxConfig!.HiddenSize, fluxConfig.HiddenSize, fluxConfig.HiddenSize],
                    CrossAttentionDim = fluxConfig.ContextInDim,
                },
                ControlNetBaseModel.QwenImage => new ControlNetConfig
                {
                    BaseModel = ControlNetBaseModel.QwenImage,
                    Mode = mode,
                    BlockOutChannels = [qwenConfig!.HiddenSize, qwenConfig.HiddenSize, qwenConfig.HiddenSize],
                    CrossAttentionDim = qwenConfig.ContextDim,
                },
                _ => throw new HartsyInferenceException($"Unsupported ControlNet base model {baseModel} for '{filePath}'."),
            };

            Dictionary<string, Tensor> weights = loader.GetAllTensors();
            if (isLdmLayout)
                weights = ControlNetCheckpointConverter.Convert(weights, isSdxl: baseModel == ControlNetBaseModel.Sdxl);

            string arch = fluxConfig is not null
                ? $", flux depth={fluxConfig.Depth}+{fluxConfig.DepthSingleBlocks}, union={(fluxConfig.IsUnion ? $"yes({fluxConfig.NumMode})" : "no")}"
                : qwenConfig is not null
                    ? $", qwen depth={qwenConfig.Depth}, hidden={qwenConfig.HiddenSize}"
                    : (unionControlTypes > 0 ? $", union={unionControlTypes} control types" : "");
            Logs.Info($"Loaded ControlNet '{Path.GetFileName(filePath)}' (base={baseModel}, mode={mode}, layout={(isLdmLayout ? "ldm→diffusers" : "diffusers")}, tensors={weights.Count}{arch}).");

            ControlNetFile file = new()
            {
                FilePath = filePath,
                BaseModel = baseModel,
                Mode = mode,
                Config = config,
                FluxConfig = fluxConfig,
                QwenConfig = qwenConfig,
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

    /// <summary>Detects xinsir-style union SDXL checkpoints by their extra keys: a <c>task_embedding</c> table plus the <c>control_add_embedding</c> MLP (diffusers <c>ControlNetUnionModel</c>). Returns the task-embedding row count (6 standard / 8 ProMax), or 0 for single-mode checkpoints. Union checkpoints only ship in diffusers layout.</summary>
    private static int DetectUnionControlTypes(
        IReadOnlyDictionary<string, SafeTensorDescriptor> descriptors, ControlNetBaseModel baseModel, bool isLdmLayout)
    {
        if (isLdmLayout || baseModel != ControlNetBaseModel.Sdxl)
            return 0;
        if (!descriptors.TryGetValue("task_embedding", out SafeTensorDescriptor? taskEmb)
            || !descriptors.ContainsKey("control_add_embedding.linear_1.weight"))
            return 0;
        if (taskEmb.Shape.Rank != 2)
        {
            throw new HartsyInferenceException(
                $"Union ControlNet task_embedding must be rank-2 [numTypes, channels]; got {taskEmb.Shape}.");
        }
        return (int)taskEmb.Shape[0];
    }

    /// <summary>Base-model detection for LDM-layout checkpoints (before key conversion): SDXL carries <c>label_emb</c> (ADM conditioning); otherwise the cross-attention context width in any <c>attn2.to_k.weight</c> discriminates (768 = SD1.5, 2048 = SDXL, 1024 = SD2.x which is unsupported).</summary>
    private static ControlNetBaseModel DetectBaseModelLdm(IReadOnlyDictionary<string, SafeTensorDescriptor> descriptors)
    {
        foreach (KeyValuePair<string, SafeTensorDescriptor> kvp in descriptors)
        {
            string ldmKey = ControlNetCheckpointConverter.StripPrefix(kvp.Key);
            if (ldmKey.StartsWith("label_emb.", StringComparison.Ordinal))
                return ControlNetBaseModel.Sdxl;
        }

        foreach (KeyValuePair<string, SafeTensorDescriptor> kvp in descriptors)
        {
            if (!kvp.Key.EndsWith("attn2.to_k.weight", StringComparison.Ordinal) || kvp.Value.Shape.Rank != 2)
                continue;
            long contextDim = kvp.Value.Shape[1];
            return contextDim switch
            {
                768 => ControlNetBaseModel.Sd15,
                2048 => ControlNetBaseModel.Sdxl,
                _ => throw new HartsyInferenceException(
                    $"Unsupported LDM ControlNet cross-attention context dim {contextDim} (key '{kvp.Key}'). " +
                    "Only SD1.5 (768) and SDXL (2048) ControlNets are supported."),
            };
        }

        // Minimal/truncated checkpoints without attention keys: SD1.5 is the only family shipped this way.
        return ControlNetBaseModel.Sd15;
    }

    private static ControlNetBaseModel DetectBaseModel(IReadOnlyDictionary<string, SafeTensorDescriptor> descriptors)
    {
        // Qwen before Flux: both DiT families carry controlnet_x_embedder/controlnet_blocks/transformer_blocks
        // keys, but only Qwen has the base transformer's img_in/txt_norm embed pair (Flux uses
        // x_embedder/context_embedder).
        if (descriptors.ContainsKey("img_in.weight") && descriptors.ContainsKey("txt_norm.weight")
            && descriptors.ContainsKey("controlnet_x_embedder.weight"))
            return ControlNetBaseModel.QwenImage;

        if (descriptors.ContainsKey("controlnet_x_embedder.weight")
            || descriptors.ContainsKey("controlnet_mode_embedder.weight")
            || descriptors.ContainsKey("controlnet_blocks.0.weight")
            || descriptors.Keys.Any(k => k.StartsWith("transformer_blocks.", StringComparison.Ordinal)))
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
