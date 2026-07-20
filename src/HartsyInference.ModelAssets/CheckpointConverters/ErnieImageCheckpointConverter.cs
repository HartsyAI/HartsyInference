using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Loads ERNIE-Image (Baidu, Apache-2.0) checkpoints from the canonical diffusers folder layout. There is **no single-file format** for ERNIE-Image — diffusers ships only the per-component folder layout (<c>transformer/</c>, <c>vae/</c>, <c>text_encoder/</c>, <c>scheduler/</c>, <c>tokenizer/</c>, optional <c>pe/</c> + <c>pe_tokenizer/</c>). Verified by grepping <c>diffusers/loaders/single_file_utils.py</c> — no <c>ernie</c> entries.
///
/// This converter loads each <c>.safetensors</c> shard under <c>{model_dir}/transformer/</c>, merges into one tensor dict, applies FP8 scale companion folding (if any community fp8 distribution ships <c>.scale_weight</c> tensors), and returns the result alongside loader handles for lifetime management. The diffusers folder layout already uses correct <c>x_embedder.proj.{w,b}</c>, <c>layers.{i}.*</c>, etc. names — no key remapping is needed.
///
/// VAE and text encoder are loaded separately by the pipeline (see <see cref="HartsyInference.Diffusion.Pipelines.ErnieImagePipeline"/>).</summary>
public sealed class ErnieImageCheckpointConverter
{
    /// <summary>Result of loading the ERNIE-Image transformer shards.</summary>
    public sealed class ConvertedWeights
    {
        /// <summary>Transformer weights in diffusers format. No remapping is applied — the upstream naming already matches our <c>ErnieImageTransformer.LoadWeights</c> expectations.</summary>
        public required Dictionary<string, Tensor> Transformer { get; init; }
    }

    /// <summary>Loads the transformer shards under <c>{folderPath}/transformer/</c> (or <paramref name="folderPath"/> directly if no <c>transformer</c> subfolder is found). Returns the merged dict + loader handles. Caller must dispose the loaders to free the mmap'd files.</summary>
    public static (ConvertedWeights Weights, IReadOnlyList<SafeTensorsLoader> Loaders) LoadAndConvert(string folderPath)
    {
        if (!Directory.Exists(folderPath))
            throw new DirectoryNotFoundException($"ERNIE-Image folder not found: {folderPath}");

        // Accept either the diffusers folder layout (`transformer/`) or the Comfy-Org repackage
        // (`diffusion_models/`, used by `Comfy-Org/ERNIE-Image`). Fall back to the root folder
        // if neither subfolder exists (single-file or flat layout).
        string transformerDir = Path.Combine(folderPath, "transformer");
        string diffusionModelsDir = Path.Combine(folderPath, "diffusion_models");
        string actualDir = Directory.Exists(transformerDir) ? transformerDir
            : Directory.Exists(diffusionModelsDir) ? diffusionModelsDir
            : folderPath;

        string[] shards = Directory.GetFiles(actualDir, "*.safetensors");
        if (shards.Length == 0)
            throw new FileNotFoundException($"No .safetensors shards found in: {actualDir}");
        Array.Sort(shards);

        Dictionary<string, Tensor> merged = new(2000);
        List<SafeTensorsLoader> loaders = new(shards.Length);
        try
        {
            foreach (string shard in shards)
            {
                SafeTensorsLoader loader = new();
                loader.Load(shard);
                loaders.Add(loader);
                foreach (KeyValuePair<string, Tensor> kvp in loader.GetAllTensors())
                {
                    if (kvp.Key.EndsWith(".scaled_fp8") || kvp.Key == "scaled_fp8") continue;
                    merged[kvp.Key] = kvp.Value;
                }
            }

            // Fold FP8-scaled per-tensor companions (`*.scale_weight`) into Tensor.Fp8ScaleFactor.
            // Harmless no-op for stock BF16/FP16 distributions; required for any ComfyUI fp8_scaled
            // community variant of ERNIE-Image. Mirrors the SD3/Flux converter pattern.
            Dictionary<string, Tensor> folded = CheckpointConvertUtils.ApplyFp8ScaledDequant(merged);

            ConvertedWeights converted = new() { Transformer = folded };
            return (converted, loaders);
        }
        catch
        {
            foreach (SafeTensorsLoader l in loaders) l.Dispose();
            throw;
        }
    }

    /// <summary>Loads VAE shards from <c>{folderPath}/vae/</c>. Returns the merged dict + loader handles. Caller must dispose loaders.</summary>
    public static (Dictionary<string, Tensor> Weights, IReadOnlyList<SafeTensorsLoader> Loaders) LoadVae(string folderPath)
    {
        string vaeDir = Path.Combine(folderPath, "vae");
        if (!Directory.Exists(vaeDir))
            throw new DirectoryNotFoundException($"VAE folder not found: {vaeDir}");

        string[] shards = Directory.GetFiles(vaeDir, "*.safetensors");
        if (shards.Length == 0)
            throw new FileNotFoundException($"No VAE .safetensors shards found in: {vaeDir}");
        Array.Sort(shards);

        Dictionary<string, Tensor> merged = new(400);
        List<SafeTensorsLoader> loaders = new(shards.Length);
        try
        {
            foreach (string shard in shards)
            {
                SafeTensorsLoader loader = new();
                loader.Load(shard);
                loaders.Add(loader);
                foreach (KeyValuePair<string, Tensor> kvp in loader.GetAllTensors())
                    merged[kvp.Key] = kvp.Value;
            }
            return (merged, loaders);
        }
        catch
        {
            foreach (SafeTensorsLoader l in loaders) l.Dispose();
            throw;
        }
    }

    /// <summary>Loads text-encoder shards from <c>{folderPath}/text_encoder/</c>. Returns the merged dict + loader handles. The exact tensor names depend on which encoder Baidu shipped; see <c>{folderPath}/text_encoder/config.json</c> to determine the architecture before constructing the encoder.</summary>
    public static (Dictionary<string, Tensor> Weights, IReadOnlyList<SafeTensorsLoader> Loaders) LoadTextEncoder(string folderPath)
    {
        // Accept diffusers `text_encoder/` or Comfy-Org `text_encoders/` (plural).
        string teDir1 = Path.Combine(folderPath, "text_encoder");
        string teDir2 = Path.Combine(folderPath, "text_encoders");
        string teDir = Directory.Exists(teDir1) ? teDir1 : teDir2;
        if (!Directory.Exists(teDir))
            throw new DirectoryNotFoundException($"text_encoder folder not found: {teDir1} or {teDir2}");

        string[] shards = Directory.GetFiles(teDir, "*.safetensors");
        if (shards.Length == 0)
            throw new FileNotFoundException($"No text_encoder .safetensors shards found in: {teDir}");
        Array.Sort(shards);

        Dictionary<string, Tensor> merged = new(800);
        List<SafeTensorsLoader> loaders = new(shards.Length);
        try
        {
            foreach (string shard in shards)
            {
                SafeTensorsLoader loader = new();
                loader.Load(shard);
                loaders.Add(loader);
                foreach (KeyValuePair<string, Tensor> kvp in loader.GetAllTensors())
                    merged[kvp.Key] = kvp.Value;
            }
            return (merged, loaders);
        }
        catch
        {
            foreach (SafeTensorsLoader l in loaders) l.Dispose();
            throw;
        }
    }
}
