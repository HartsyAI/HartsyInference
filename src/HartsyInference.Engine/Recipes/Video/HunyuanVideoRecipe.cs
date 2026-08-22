using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.TextEncoders;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Video.Pipelines;

using HartsyInference.Engine.Features;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>HunyuanVideo (Tencent 13B T2V) recipe: the Comfy-Org repacked bf16 single-file DiT + a standalone HunyuanVideo 3D VAE, conditioned by LLaVA-Llama-3-8B (fp8, <see cref="SideModels.LlavaLlama3"/>, layer −3 through the diffusers prompt template) and CLIP-L (<see cref="SideModels.ClipL"/>) for the pooled vector. Already parity-verified per memory <c>hunyuanvideo-13b-e2e</c>; this wrapper lifts <c>HunyuanVideoGenerationTests</c>' proven construction into the recipe registry so it is reachable from the CLI.</summary>
public sealed class HunyuanVideoRecipe : IVideoRecipe
{
    // The exact diffusers prompt template + template-token crop count (pipeline_hunyuan_video.py:70-81).
    private const string PromptTemplate =
        "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: " +
        "1. The main content and theme of the video." +
        "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects." +
        "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects." +
        "4. background environment, light, style and atmosphere." +
        "5. camera angles, movements, and transitions used in the video:<|eot_id|>" +
        "<|start_header_id|>user<|end_header_id|>\n\n{0}<|eot_id|>";
    internal const int CropStart = 95;
    internal const int LlamaLayer = 30; // hidden_states[-3] for a 32-layer model (num_hidden_layers_to_skip=2)

    /// <inheritdoc/>
    public string Name => "hunyuan-video";

    /// <inheritdoc/>
    public bool Matches(string familyId) =>
        string.Equals(familyId, "hunyuan-video", StringComparison.OrdinalIgnoreCase)
        || string.Equals(familyId, "hunyuanvideo", StringComparison.OrdinalIgnoreCase);

    /// <summary>HunyuanVideo's official sampling settings: 20 steps at embedded-guidance 6.0 (no CFG here — no negative branch amplifies fp8 noise), 512x320, 25 frames @ 24fps — the geometry <c>HunyuanVideoGenerationTests</c> verified coherent (real 720p is the trained resolution but far too slow for a CLI turnaround; per MODEL_STATUS_VIDEO.md the 512x320/2.15s-per-step config is the proven production path).</summary>
    public VideoDefaults Defaults { get; } = new VideoDefaults { Steps = 20, CfgScale = 6.0f, Width = 512, Height = 320, Frames = 25, Fps = 24 };

    /// <inheritdoc/>
    /// <remarks><see cref="VideoFeatures.Lora"/> added 2026-08-20 — the first conditioning this family declares at all (it previously inherited <see cref="IVideoRecipe"/>'s <c>None</c>). Image-to-video remains unwired and is tracked separately: HunyuanVideo-I2V is a distinct upstream checkpoint with a different input-channel count, not a flag on this T2V one.</remarks>
    public VideoFeatures Supports => VideoFeatures.Lora;

    /// <inheritdoc/>
    public IVideoRecipePipeline Construct(RecipeContext context)
    {
        // TODO(E-IMG-4/5): image-to-video conditioning and a VideoRequest.Components LLaVA/CLIP/VAE override are
        // deferred — this is the text-to-video path only.
        List<SafeTensorsLoader> loaders = new List<SafeTensorsLoader>();
        try
        {
            Logs.Info($"[HunyuanVideoRecipe] Loading + converting DiT: {Path.GetFileName(context.CheckpointPath)}.");
            (Dictionary<string, Tensor> ditWeights, SafeTensorsLoader ditLoader) = HunyuanVideoCheckpointConverter.LoadAndConvert(context.CheckpointPath);
            loaders.Add(ditLoader);
            if (ditWeights.Count == 0)
            {
                throw new InvalidOperationException($"HunyuanVideo checkpoint '{context.CheckpointPath}' has no recognized DiT weights after conversion.");
            }
            // BF16 -> F16 (native cuBLAS F16 GEMM; the CacheWeightCasts=false transient path leaves BF16 weights
            // unapplied -> a blank/flat-gray render, memory bf16-transformer-needs-f16-cast).
            foreach (string k in ditWeights.Keys.ToList())
            {
                if (ditWeights[k].DType == DType.BF16)
                {
                    ditWeights[k] = ditWeights[k].CastTo(DType.F16);
                }
            }
            HunyuanVideoConfig config = HunyuanVideoConfig.T2V;
            HunyuanVideoDit dit = new HunyuanVideoDit(config);
            // Merge any requested LoRAs BEFORE LoadWeights — device caches are identity-keyed, so merging
            // after would leave layers serving the pre-merge tensors (the Sd3Recipe ordering rule).
            MergedLoraStack? loraStack = LoraApplier.BuildAndApply(
                LoraResolver.Resolve(context.Loras), context.Backend, transformerWeights: ditWeights);
            dit.LoadWeights(ditWeights);

            string vaePath = ModelDownloader.EnsureSideModelAsync(SideModels.HunyuanVideoVae3D, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
            HunyuanVideoVaeDecoder vae = new HunyuanVideoVaeDecoder();
            vae.LoadWeights(CastBf16ToF16(HunyuanVideoCheckpointConverter.ConvertVaeDecoder(LoadStandalone(loaders, vaePath))));

            // bf16-resident, block-streamed DiT — caching F16 casts would roughly double VRAM and OOM a 24 GB card.
            RecipeBackendFlags.DisableCacheWeightCasts(context, "HunyuanVideoRecipe");

            string llavaPath = ModelDownloader.EnsureSideModelAsync(SideModels.LlavaLlama3, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
            LlamaStyleEncoder llava = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Llama31_8B);
            llava.LoadWeights(TextEncoderQuantNormalizer.Normalize(LoadStandalone(loaders, llavaPath)));

            string clipPath = ModelDownloader.EnsureSideModelAsync(SideModels.ClipL, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
            ClipTextEncoder clipL = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipL);
            clipL.LoadWeights(LoadStandalone(loaders, clipPath), "text_model");

            HunyuanVideoPipeline pipeline = new HunyuanVideoPipeline(context.Backend, dit, vae, config);
            Logs.Info("[HunyuanVideoRecipe] HunyuanVideo ready (text-to-video).");
            return new HunyuanVideoRecipePipeline(context.Backend, pipeline, llava, clipL, new ClipTokenizer(), dit, loaders, loraStack);
        }
        catch (Exception ex)
        {
            Logs.Error("[HunyuanVideoRecipe] Construction failed.", ex);
            foreach (SafeTensorsLoader loader in loaders)
            {
                loader.Dispose();
            }
            throw;
        }
    }

    /// <summary>Builds the templated + BOS-prefixed Llama-3 token sequence the diffusers pipeline feeds LLaVA (add_special_tokens=True → BOS prepended), from the embedded Llama-3 byte-level BPE tokenizer.</summary>
    internal static int[] BuildTemplatedTokens(string prompt)
    {
        using Stream json = EmbeddedTokenizerResources.OpenLlama3TokenizerJson();
        HartsyInference.ModelAssets.Tokenizers.GgufTokenizer tok = HfTokenizerJson.LoadByteLevelBpe(json);
        string templated = string.Format(System.Globalization.CultureInfo.InvariantCulture, PromptTemplate, prompt);
        int[] ids = tok.Encode(templated, addSpecial: true);
        int[] withBos = new int[ids.Length + 1];
        withBos[0] = tok.BosId ?? 128000;
        Array.Copy(ids, 0, withBos, 1, ids.Length);
        return withBos;
    }

    private static Dictionary<string, Tensor> LoadStandalone(List<SafeTensorsLoader> loaders, string path)
    {
        SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(path);
        loaders.Add(loader);
        return CheckpointConvertUtils.ApplyFp8ScaledDequant(loader.GetAllTensors());
    }

    private static Dictionary<string, Tensor> CastBf16ToF16(Dictionary<string, Tensor> weights)
    {
        foreach (string key in weights.Keys.ToList())
        {
            if (weights[key].DType == DType.BF16)
            {
                weights[key] = weights[key].CastTo(DType.F16);
            }
        }
        return weights;
    }
}
