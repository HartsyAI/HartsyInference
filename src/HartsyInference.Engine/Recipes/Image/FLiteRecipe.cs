using System.IO;
using HartsyInference.Core.Logging;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>F-Lite recipe (Freepik / Fal.ai lightweight Flux-Schnell derivative). F-Lite ships only in the diffusers folder layout (<c>dit_model/</c> + <c>text_encoder/</c> + <c>vae/</c>) — there is no single-file release. The transformer, T5-XXL encoder, and Flux Schnell VAE all come from that folder; nothing is resolved as a side model. Lifted from the SwarmUI backend's <c>FLiteLoader</c>; constructs and drives through <see cref="FLiteRecipePipeline"/>.</summary>
public sealed class FLiteRecipe : IArchitectureRecipe
{
    /// <inheritdoc/>
    public string Name => "f-lite";

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "f-lite", StringComparison.OrdinalIgnoreCase);

    /// <summary>F-Lite's official sampling settings: 30 steps at CFG 6.0, 1024x1024 (<c>FLiteConfig.DefaultSteps</c>/<c>DefaultCfgScale</c>).</summary>
    public static ImageDefaults FamilyDefaults { get; } = new ImageDefaults { Steps = 30, CfgScale = 6.0f, Width = 1024, Height = 1024 };

    /// <inheritdoc/>
    public ImageDefaults Defaults => FamilyDefaults;

    /// <inheritdoc/>
    public IRecipePipeline Construct(RecipeContext context)
    {
        string folderPath = ResolveFolderRoot(context.CheckpointPath, "dit_model", "text_encoder", "vae");
        Logs.Info($"[FLiteRecipe] Loading F-Lite folder: {Path.GetFileName(folderPath)}.");

        (FLiteCheckpointConverter.ConvertedWeights converted, FLiteCheckpointConverter.LoaderHandle handle) = FLiteCheckpointConverter.LoadAndConvert(folderPath);
        Logs.Info($"[FLiteRecipe] Converted: {converted.Transformer.Count} dit / {converted.T5.Count} T5 / {converted.Vae.Count} VAE keys.");
        if (converted.Transformer.Count == 0 || converted.T5.Count == 0 || converted.Vae.Count == 0)
        {
            handle.Dispose();
            throw new InvalidOperationException($"F-Lite folder '{folderPath}' is missing dit_model / text_encoder / vae components.");
        }

        FLiteConfig config = FLiteConfig.V1;
        Logs.Info($"[FLiteRecipe] Building transformer (hidden={config.HiddenSize}, depth={config.Depth}).");
        FLiteTransformer transformer = new FLiteTransformer(config);
        transformer.LoadWeights(converted.Transformer);

        T5TextEncoder t5 = new T5TextEncoder(T5TextEncoderConfig.Xxl);
        t5.LoadWeights(converted.T5);

        VaeDecoder vae = new VaeDecoder(VaeConfig.Flux);
        vae.LoadWeights(converted.Vae);

        FLitePipeline pipeline = new FLitePipeline(context.Backend, t5, transformer, vae, config);
        Logs.Info("[FLiteRecipe] F-Lite ready.");
        return new FLiteRecipePipeline(pipeline, new T5Tokenizer(maxLength: 512), handle);
    }

    /// <summary>Resolves the F-Lite diffusers-folder root from the checkpoint path: if it is already a directory containing one of the expected subfolders it is used directly, otherwise the loader walks up to 3 parent levels from the picked file (mirrors the SwarmUI loader's <c>ResolveFolderRoot</c>).</summary>
    private static string ResolveFolderRoot(string checkpointPath, params string[] expectedSubfolders)
    {
        string? dir = Directory.Exists(checkpointPath) ? checkpointPath : Path.GetDirectoryName(checkpointPath);
        for (int depth = 0; depth < 4 && dir is not null; depth++)
        {
            foreach (string sub in expectedSubfolders)
            {
                if (Directory.Exists(Path.Combine(dir, sub)))
                {
                    return dir;
                }
            }
            dir = Path.GetDirectoryName(dir);
        }
        throw new InvalidOperationException($"Could not find any of the expected subfolders [{string.Join(", ", expectedSubfolders)}] by walking up from '{checkpointPath}'. F-Lite requires a diffusers-style folder layout.");
    }
}
