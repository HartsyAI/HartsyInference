using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
using System.Text.RegularExpressions;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Engine.HuggingFace;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

using HartsyInference.Engine.Features;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>ERNIE-Image recipe (Baidu, ~8B, Apache-2.0): the checkpoint is the single-stream DiT, the Ministral-3-3B text encoder (<see cref="SideModels.Ministral_3_3B"/>) and the Flux.2 128-channel VAE (<see cref="SideModels.Flux2Vae"/>) resolve as side models. Lifted from the SwarmUI backend's <c>ErnieImageLoader</c>, including its sharded-transformer merge and its tokenizer fallback: there is no embedded ERNIE tokenizer, so the Mistral3 byte-level BPE <c>tokenizer.json</c> is fetched once from <c>baidu/ERNIE-Image</c> (the Comfy-Org repackage omits it) and read by <see cref="ErnieTokenizer"/>. Constructs and drives through <see cref="ErnieImageRecipePipeline"/>.</summary>
public sealed partial class ErnieImageRecipe : IArchitectureRecipe
{
    /// <summary>Repo hosting the ERNIE tokenizer.json (Mistral3 BPE, ~17 MB); the Comfy-Org repackage that ships the TE/VAE/DiT does not include it.</summary>
    private const string TokenizerRepo = "baidu/ERNIE-Image";

    /// <summary>Path of the tokenizer.json inside <see cref="TokenizerRepo"/>.</summary>
    private const string TokenizerRepoPath = "tokenizer/tokenizer.json";

    /// <inheritdoc/>
    public string Name => "ernie-image";


    /// <inheritdoc/>
    /// <remarks>ERNIE-Image shares the Flux.2 VAE; the encoder half is built alongside the decoder.
    /// <para><see cref="ImageFeatures.Lora"/> added 2026-08-20. <see cref="HartsyInference.Diffusion.Models.Denoisers.ErnieImageTransformer"/> names its blocks <c>layers.{i}</c>, a root the bare-root LoRA detector only started recognizing in the same change.</para></remarks>
    public ImageFeatures Supports => ImageFeatures.Img2Img | ImageFeatures.Inpaint | ImageFeatures.SeamlessTiling | ImageFeatures.VariationSeed | ImageFeatures.Refiner | ImageFeatures.Lora;
    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "ernie-image", StringComparison.OrdinalIgnoreCase);

    /// <summary>ERNIE-Image's official sampling settings: 50 steps at CFG 4.0, 1024x1024 (<c>GenerationDefaults.ErnieImage</c>).</summary>
    public static ImageDefaults FamilyDefaults { get; } = new ImageDefaults { Steps = 50, CfgScale = 4.0f, Width = 1024, Height = 1024 };

    /// <inheritdoc/>
    public ImageDefaults Defaults => FamilyDefaults;

    /// <inheritdoc/>
    public IRecipePipeline Construct(RecipeContext context)
    {
        // TODO(E-IMG-4): honor a user-picked VAE override from ImageRequest.Components (the SwarmUI loader read
        // T2IParamTypes.VAE); the text encoder was already pinned to the canonical Ministral-3-3B there.
        string tePath = ModelDownloader.EnsureSideModelAsync(SideModels.Ministral_3_3B, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
        string vaePath = ModelDownloader.EnsureSideModelAsync(SideModels.Flux2Vae, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();

        List<SafeTensorsLoader> loaders = new List<SafeTensorsLoader>();
        try
        {
            // ERNIE-Image ships the transformer as a sharded diffusers set; loading only the picked shard leaves half
            // the keys missing (final_norm.linear.weight lives in shard 2), so every sibling shard is merged in.
            Logs.Info($"[ErnieImageRecipe] Loading transformer: {Path.GetFileName(context.CheckpointPath)}.");
            Dictionary<string, Tensor> transformerWeights = LoadShardedComponent(context.CheckpointPath, applyFp8Dequant: true, loaders);

            Logs.Info($"[ErnieImageRecipe] Loading Ministral-3-3B text encoder: {Path.GetFileName(tePath)}.");
            Dictionary<string, Tensor> teWeights = ComponentLoader.Load(tePath, "ErnieImageRecipe", keyTransform: null, applyFp8Dequant: true, loaders);

            Logs.Info($"[ErnieImageRecipe] Loading Flux.2 VAE: {Path.GetFileName(vaePath)}.");
            Dictionary<string, Tensor> vaeWeights = ComponentLoader.Load(vaePath, "ErnieImageRecipe", keyTransform: null, applyFp8Dequant: false, loaders);

            ErnieImageConfig config = ErnieImageConfig.V1;

            ErnieImageTransformer transformer = new ErnieImageTransformer(config);
            // Merge any requested LoRAs BEFORE LoadWeights — device caches are identity-keyed, so merging
            // after would leave layers serving the pre-merge tensors (the Sd3Recipe ordering rule).
            MergedLoraStack? loraStack = LoraApplier.BuildAndApply(
                LoraResolver.Resolve(context.Loras), context.Backend, transformerWeights: transformerWeights);
            transformer.LoadWeights(transformerWeights);

            VaeDecoder vae = new VaeDecoder(VaeConfig.Flux2);
            vae.LoadWeights(vaeWeights);
            VaeEncoder? vaeEncoder = LoaderVaeUtils.TryBuildEncoder(VaeConfig.Flux2, vaeWeights, "ErnieImageRecipe");

            LlamaStyleEncoder llama = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Ministral3B);
            llama.LoadWeights(teWeights);
            ErnieImageLlamaTextEncoder textEncoder = new ErnieImageLlamaTextEncoder(llama)
                .WithHiddenSize(LlamaStyleEncoderConfig.Ministral3B.HiddenSize);

            ErnieTokenizer tokenizer = new ErnieTokenizer(EnsureTokenizerJson());

            // TODO(E-IMG-5): the diffusers ERNIE pipeline un-normalizes the latent with the Flux.2 VAE's
            // bn.running_mean/running_var before decode; ErnieImagePipeline takes those as optional ctor args.
            // Left null to match the engine's validated wiring (the stage/shape needs confirming first — the BN
            // element count must match the 128-channel PACKED latent, not the 32-channel one).
            ErnieImagePipeline pipeline = new ErnieImagePipeline(context.Backend, textEncoder, transformer, vae, config, vaeEncoder: vaeEncoder);
            Logs.Info("[ErnieImageRecipe] ERNIE-Image ready.");
            return new ErnieImageRecipePipeline(pipeline, tokenizer, textEncoder, llama, transformer, loaders, loraStack);
        }
        catch (Exception ex)
        {
            Logs.Error("[ErnieImageRecipe] Construction failed.", ex);
            foreach (SafeTensorsLoader loader in loaders)
            {
                loader.Dispose();
            }
            throw;
        }
    }

    /// <summary>Matches the diffusers <c>&lt;base&gt;-NNNNN-of-MMMMM.safetensors</c> shard filename convention.</summary>
    [GeneratedRegex(@"^(.*)-(\d+)-of-(\d+)\.safetensors$")]
    private static partial Regex ShardPattern();

    /// <summary>Loads a component that may be split across diffusers shards, merging every sibling shard when <paramref name="filePath"/> follows the shard naming convention; degrades to a single-file load otherwise.</summary>
    private static Dictionary<string, Tensor> LoadShardedComponent(string filePath, bool applyFp8Dequant, List<SafeTensorsLoader> loaders)
    {
        List<string> files;
        Match m = ShardPattern().Match(Path.GetFileName(filePath));
        if (m.Success)
        {
            string dir = Path.GetDirectoryName(filePath) ?? ".";
            string prefix = m.Groups[1].Value;
            int width = m.Groups[2].Value.Length;
            int total = int.Parse(m.Groups[3].Value);
            files = new List<string>(total);
            for (int i = 1; i <= total; i++)
            {
                files.Add(Path.Combine(dir, $"{prefix}-{i.ToString().PadLeft(width, '0')}-of-{total.ToString().PadLeft(width, '0')}.safetensors"));
            }
        }
        else
        {
            files = new List<string> { filePath };
        }

        Dictionary<string, Tensor> merged = new Dictionary<string, Tensor>();
        foreach (string file in files)
        {
            if (!File.Exists(file))
            {
                throw new FileNotFoundException($"ERNIE-Image transformer shard missing: {file}");
            }
            SafeTensorsLoader loader = new SafeTensorsLoader();
            loader.Load(file);
            loaders.Add(loader);
            foreach (KeyValuePair<string, Tensor> kv in loader.GetAllTensors())
            {
                if (kv.Key.EndsWith(".scaled_fp8", StringComparison.Ordinal) || kv.Key == "scaled_fp8")
                {
                    continue;
                }
                merged[kv.Key] = kv.Value;
            }
        }
        return applyFp8Dequant ? CheckpointConvertUtils.ApplyFp8ScaledDequant(merged) : merged;
    }

    /// <summary>Ensures the ERNIE tokenizer.json is present under the models root, downloading it from <see cref="TokenizerRepo"/> on first use.</summary>
    private static string EnsureTokenizerJson()
    {
        string path = Path.Combine(RepoPaths.ModelsRoot(), "text_encoders", "ERNIE", "tokenizer.json");
        if (!File.Exists(path))
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            Logs.Info($"[ErnieImageRecipe] Downloading ERNIE tokenizer.json from {TokenizerRepo}...");
            using HuggingFaceClient client = new HuggingFaceClient();
            client.DownloadFileAsync(TokenizerRepo, TokenizerRepoPath, path, progress: null, sha256: null, CancellationToken.None)
                .GetAwaiter().GetResult();
        }
        return path;
    }
}
