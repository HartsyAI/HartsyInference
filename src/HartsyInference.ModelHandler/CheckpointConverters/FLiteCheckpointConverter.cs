using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.SafeTensors;

namespace HartsyInference.ModelHandler.CheckpointConverters;

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

    /// <summary>Holds the SafeTensorsLoader instances open while the returned tensors are in use. Dispose to release the mmap-backed memory.</summary>
    public sealed class LoaderHandle : IDisposable
    {
        private readonly List<SafeTensorsLoader> _loaders;

        internal LoaderHandle(List<SafeTensorsLoader> loaders) { _loaders = loaders; }

        public void Dispose()
        {
            foreach (SafeTensorsLoader loader in _loaders) loader.Dispose();
            _loaders.Clear();
        }
    }

    /// <summary>Loads all F-Lite components from a folder. Reads every safetensors file in <c>{root}/dit_model</c>, <c>{root}/text_encoder</c>, <c>{root}/vae</c> and partitions them into the three component dicts.</summary>
    public static (ConvertedWeights weights, LoaderHandle handle) LoadAndConvert(string folderPath)
    {
        if (!Directory.Exists(folderPath))
            throw new HartsyInference.Core.Exceptions.HartsyInferenceException($"F-Lite folder does not exist: {folderPath}");

        List<SafeTensorsLoader> loaders = new();

        try
        {
            Dictionary<string, Tensor> transformer = LoadComponent(folderPath, "dit_model", loaders);
            Dictionary<string, Tensor> textEncoder = LoadComponent(folderPath, "text_encoder", loaders);
            // VAE in F-Lite folder is the Flux Schnell VAE saved in diffusers format — keys are
            // already canonical (encoder.down_blocks.X.resnets.Y..., decoder.up_blocks.X..., etc.)
            // No key remap is needed (and would mangle them — ConvertVaeKey targets LDM-format keys).
            Dictionary<string, Tensor> vae = LoadComponent(folderPath, "vae", loaders);

            Logs.Info($"F-Lite loaded: dit={transformer.Count} keys, t5={textEncoder.Count} keys, vae={vae.Count} keys.");

            ConvertedWeights converted = new()
            {
                Transformer = transformer,
                T5 = textEncoder,
                Vae = vae,
            };
            return (converted, new LoaderHandle(loaders));
        }
        catch
        {
            foreach (SafeTensorsLoader loader in loaders) loader.Dispose();
            throw;
        }
    }

    private static Dictionary<string, Tensor> LoadComponent(string root, string component, List<SafeTensorsLoader> loaders)
    {
        string componentDir = Path.Combine(root, component);
        if (!Directory.Exists(componentDir))
            throw new HartsyInference.Core.Exceptions.HartsyInferenceException($"F-Lite component directory missing: {componentDir}");

        string[] shards = Directory.GetFiles(componentDir, "*.safetensors");
        if (shards.Length == 0)
            throw new HartsyInference.Core.Exceptions.HartsyInferenceException($"No safetensors shards in {componentDir}");

        Dictionary<string, Tensor> merged = new();
        foreach (string shard in shards)
        {
            SafeTensorsLoader loader = new();
            loader.Load(shard);
            loaders.Add(loader);
            foreach (KeyValuePair<string, Tensor> kvp in loader.GetAllTensors())
            {
                if (merged.ContainsKey(kvp.Key))
                    throw new HartsyInference.Core.Exceptions.HartsyInferenceException($"Duplicate key '{kvp.Key}' across shards in {componentDir}");
                merged[kvp.Key] = kvp.Value;
            }
        }
        return merged;
    }

}
