using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
using System.Text.Json;
using HartsyInference.Core.Logging;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.HuggingFace;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Lens recipe (Microsoft Lens, 3.8B dual-stream MMDiT): the checkpoint is the DiT (<c>lens_bf16</c>/<c>lens_mxfp8</c>/<c>lens_turbo_*</c>; MXFP8 quants are dequanted by the engine converter), while the GPT-OSS-20B text encoder (<see cref="SideModels.LensGptOss20b"/>) and the Flux.2 VAE (<see cref="SideModels.Flux2Vae"/>) resolve as side models. A "turbo" in the checkpoint filename selects <see cref="LensConfig.Turbo"/> (4 steps, no CFG); everything else uses <see cref="LensConfig.Default"/> (20 steps, CFG 5). Lifted from the SwarmUI backend's <c>LensLoader</c> — construction goes through <see cref="LensPipelineFactory"/> — and driven through <see cref="LensRecipePipeline"/>.</summary>
public sealed class LensRecipe : IArchitectureRecipe
{
    /// <summary>The microsoft/Lens repo is gated; the GPT-OSS o200k-harmony tokenizer Lens' encoder uses ships byte-identical with the ungated openai/gpt-oss-20b repo.</summary>
    private const string TokenizerRepo = "openai/gpt-oss-20b";

    /// <inheritdoc/>
    public string Name => "lens";


    /// <inheritdoc/>
    /// <remarks>Lens reuses the Flux.2 VAE, so its encoder rides that config's encode-parity gate. Img2img only:
    /// the loop integrates in packed token space and the shared mask-blend helpers have no variant for Lens's
    /// packing, so a masked path would need a blend that does not exist yet.
    /// <para><see cref="ImageFeatures.Lora"/> added 2026-08-20. <see cref="HartsyInference.Diffusion.Models.Denoisers.LensTransformer"/> names its blocks <c>transformer_blocks.{i}</c>, an already-recognized canonical diffusers root. The merge runs through <see cref="LensPipelineFactory.LoadFromComfyFiles"/>'s <c>onTransformerWeights</c> hook rather than beside a LoadWeights call — Lens is the only family whose transformer weights are loaded inside the Diffusion package.</para></remarks>
    public ImageFeatures Supports => ImageFeatures.Img2Img | ImageFeatures.SeamlessTiling | ImageFeatures.VariationSeed | ImageFeatures.Inpaint | ImageFeatures.Refiner | ImageFeatures.Lora;
    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "lens", StringComparison.OrdinalIgnoreCase);

    /// <summary>Lens's official sampling settings: 20 steps at CFG 5.0, 1024x1024 (<c>LensConfig.Default</c>); a Lens-Turbo checkpoint narrows this to 4 guidance-free steps via <see cref="LensRecipePipeline.VariantDefaults"/>.</summary>
    public static ImageDefaults FamilyDefaults { get; } = new ImageDefaults { Steps = 20, CfgScale = 5.0f, Width = 1024, Height = 1024 };

    /// <inheritdoc/>
    public ImageDefaults Defaults => FamilyDefaults;

    /// <summary>Lens-Turbo's official sampling settings: 4 distilled steps, guidance-free (<c>LensConfig.Turbo</c>).</summary>
    public static ImageDefaults TurboDefaults { get; } = new ImageDefaults { Steps = 4, CfgScale = 1.0f, Width = 1024, Height = 1024 };

    /// <inheritdoc/>
    public IRecipePipeline Construct(RecipeContext context)
    {
        // TODO(E-IMG-4): honor a user-picked VAE override from ImageRequest.Components (the SwarmUI loader read
        // input.Get(T2IParamTypes.VAE)); this always takes the canonical Flux.2 VAE.
        string textEncoderPath = ModelDownloader.EnsureSideModelAsync(SideModels.LensGptOss20b, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
        string vaePath = ModelDownloader.EnsureSideModelAsync(SideModels.Flux2Vae, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
        (string vocabPath, string mergesPath) = EnsureGptOssVocabMerges();

        LensConfig config = Path.GetFileName(context.CheckpointPath).Contains("turbo", StringComparison.OrdinalIgnoreCase)
            ? LensConfig.Turbo
            : LensConfig.Default;
        Logs.Info($"[LensRecipe] Loading Lens ({(config.DefaultCfgScale > 1 ? "standard" : "turbo")}): {Path.GetFileName(context.CheckpointPath)}.");

        // Merge any requested LoRAs BEFORE LoadWeights — device caches are identity-keyed, so merging after
        // would leave layers serving the pre-merge tensors (the Sd3Recipe ordering rule). Lens is the one family
        // whose transformer weights are loaded inside the Diffusion-package factory rather than here, so the
        // merge rides the factory's onTransformerWeights hook instead of sitting next to a LoadWeights call.
        MergedLoraStack? loraStack = null;
        LensPipelineBundle bundle = LensPipelineFactory.LoadFromComfyFiles(
            context.Backend, context.CheckpointPath, textEncoderPath, vaePath, config,
            onTransformerWeights: transformerWeights => loraStack = LoraApplier.BuildAndApply(
                LoraResolver.Resolve(context.Loras), context.Backend, transformerWeights: transformerWeights));
        try
        {
            GptOssTokenizer tokenizer = new GptOssTokenizer(vocabPath, mergesPath);
            Logs.Info("[LensRecipe] Lens ready.");
            return new LensRecipePipeline(bundle, config, tokenizer, loraStack);
        }
        catch (Exception ex)
        {
            Logs.Error("[LensRecipe] Tokenizer construction failed.", ex);
            bundle.Dispose();
            throw;
        }
    }

    /// <summary>Ensures the GPT-OSS <c>vocab.json</c> + <c>merges.txt</c> pair <see cref="GptOssTokenizer"/> consumes exists on disk, downloading and splitting the HuggingFace <c>tokenizer.json</c> once. Inlines the SwarmUI backend's <c>TokenizerExport.EnsureVocabMerges</c>.</summary>
    private static (string VocabPath, string MergesPath) EnsureGptOssVocabMerges()
    {
        string dir = Path.Combine(RepoPaths.ModelsRoot(), "text_encoders", "Lens");
        string vocabPath = Path.Combine(dir, "gpt_oss_vocab.json");
        string mergesPath = Path.Combine(dir, "gpt_oss_merges.txt");
        if (File.Exists(vocabPath) && File.Exists(mergesPath))
        {
            return (vocabPath, mergesPath);
        }
        Directory.CreateDirectory(dir);
        string rawPath = Path.Combine(dir, "gpt_oss_tokenizer_raw.json");
        Logs.Info("[LensRecipe] Downloading the GPT-OSS tokenizer vocab (one-time)...");
        using (HuggingFaceClient client = new HuggingFaceClient())
        {
            client.DownloadFileAsync(TokenizerRepo, "tokenizer.json", rawPath, progress: null, sha256: null, CancellationToken.None)
                .GetAwaiter().GetResult();
        }
        using (FileStream rawStream = File.OpenRead(rawPath))
        using (JsonDocument doc = JsonDocument.Parse(rawStream))
        {
            JsonElement model = doc.RootElement.GetProperty("model");
            File.WriteAllText(vocabPath, model.GetProperty("vocab").GetRawText());
            List<string> merges = new List<string>();
            foreach (JsonElement merge in model.GetProperty("merges").EnumerateArray())
            {
                merges.Add(merge.ValueKind == JsonValueKind.Array
                    ? $"{merge[0].GetString()} {merge[1].GetString()}"
                    : merge.GetString() ?? "");
            }
            File.WriteAllLines(mergesPath, merges);
        }
        File.Delete(rawPath);
        Logs.Info("[LensRecipe] GPT-OSS tokenizer vocab + merges exported.");
        return (vocabPath, mergesPath);
    }
}
