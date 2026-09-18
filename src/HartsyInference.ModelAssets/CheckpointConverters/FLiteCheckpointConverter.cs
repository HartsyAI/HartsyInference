using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Core.Memory;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Loads F-Lite from its diffusers folder layout (`Freepik/F-Lite`):
///
/// <code>
/// F-Lite/
///   dit_model/diffusion_pytorch_model.safetensors      (or sharded)
///   text_encoder/model.safetensors                     (T5-XXL, possibly sharded)
///   tokenizer/                                          (T5 SentencePiece)
///   vae/diffusion_pytorch_model.safetensors            (Flux Schnell VAE, 16-ch)
///   model_index.json
/// </code>
///
/// The F-Lite DiT keys are already in diffusers naming — no remap is needed for the public release. Optional weights (norm scales, qkv biases, qk_norm scales) are silently absent on the 10B since it ships with `train_bias_and_rms=false`.
///
/// VAE keys follow the Flux/SD3 AutoencoderKL diffusers convention; routes through <see cref="CheckpointConvertUtils.ConvertVaeKey"/>.</summary>
public sealed class FLiteCheckpointConverter
{
    /// <summary>Result of loading an F-Lite folder. Each dict keys are already in the format the corresponding component expects (no further remap needed).</summary>
    public sealed class ConvertedWeights
    {
        /// <summary>FLiteTransformer (DiT) weights.</summary>
        public required Dictionary<string, Tensor> Transformer { get; init; }

        /// <summary>T5-XXL text encoder weights.</summary>
        public required Dictionary<string, Tensor> T5 { get; init; }

        /// <summary>VAE weights (Flux Schnell VAE, 16-channel).</summary>
        public required Dictionary<string, Tensor> Vae { get; init; }
    }

    /// <summary>Loads all F-Lite components from a folder. Reads every checkpoint container in <c>{root}/dit_model</c>, <c>{root}/text_encoder</c>, <c>{root}/vae</c> and partitions them into the three component dicts.</summary>
    /// <remarks>Each component opens as one <see cref="Checkpoints.CheckpointSource"/>, so a GGUF or a quantized
    /// repack of any of the three loads, and its quantization companions are folded across the whole component
    /// rather than per shard — F-Lite never folded them at all before, so an fp8_scaled build ran at
    /// <c>1/scale</c>.</remarks>
    /// <returns>The three component dicts, plus the single handle that keeps their memory mapped.</returns>
    public static (ConvertedWeights weights, IDisposable sources) LoadFolder(string folderPath)
    {
        if (!Directory.Exists(folderPath))
            throw new HartsyInference.Core.Exceptions.HartsyInferenceException($"F-Lite folder does not exist: {folderPath}");

        List<IDisposable> sources = new List<IDisposable>(3);
        try
        {
            Dictionary<string, Tensor> transformer = LoadComponent(folderPath, "dit_model", sources);
            Dictionary<string, Tensor> textEncoder = LoadComponent(folderPath, "text_encoder", sources);
            // VAE in F-Lite folder is the Flux Schnell VAE saved in diffusers format — keys are
            // already canonical (encoder.down_blocks.X.resnets.Y..., decoder.up_blocks.X..., etc.)
            // No key remap is needed (and would mangle them — ConvertVaeKey targets LDM-format keys).
            Dictionary<string, Tensor> vae = LoadComponent(folderPath, "vae", sources);

            Logs.Info($"F-Lite loaded: dit={transformer.Count} keys, t5={textEncoder.Count} keys, vae={vae.Count} keys.");

            ConvertedWeights converted = new()
            {
                Transformer = transformer,
                T5 = textEncoder,
                Vae = vae,
            };
            return (converted, new CompositeDisposable([.. sources]));
        }
        catch
        {
            foreach (IDisposable source in sources) source.Dispose();
            throw;
        }
    }

    private static Dictionary<string, Tensor> LoadComponent(string root, string component, List<IDisposable> sources)
    {
        string componentDir = Path.Combine(root, component);
        if (!Directory.Exists(componentDir))
            throw new HartsyInference.Core.Exceptions.HartsyInferenceException($"F-Lite component directory missing: {componentDir}");

        string[] shards = CheckpointConvertUtils.DiscoverContainerFiles(componentDir);
        if (shards.Length == 0)
            throw new HartsyInference.Core.Exceptions.HartsyInferenceException($"No safetensors or GGUF checkpoint in {componentDir}");

        Checkpoints.CheckpointSource source = Checkpoints.CheckpointSource.OpenShards(shards);
        sources.Add(source);
        return new Dictionary<string, Tensor>(source.Weights);
    }
}
