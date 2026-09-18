using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Memory;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Models.Vae.QwenImage;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

using HartsyInference.Engine.Features;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Anima recipe (Cosmos-Predict2-2B lineage): the single-file checkpoint carries BOTH the DiT trunk and the Anima-specific <c>llm_adapter</c> sub-transformer, while the Qwen-3 0.6B text encoder (<see cref="SideModels.Qwen3_0_6B"/>) and the Qwen-Image 3D-causal VAE (<see cref="SideModels.QwenImageVae"/>) resolve as side models. Lifted from the SwarmUI backend's <c>AnimaLoader</c>; constructs the components and drives generation through <see cref="AnimaRecipePipeline"/>.</summary>
public sealed class AnimaRecipe : IArchitectureRecipe
{
    /// <summary>Anima trains with T5 inputs tokenized to <c>max_length=512</c>; the pipeline right-pads the LlmAdapter output to exactly 512 and throws beyond it, so the T5 ids must be capped here.</summary>
    internal const int T5MaxTokens = 512;

    /// <inheritdoc/>
    public string Name => "anima";

    /// <inheritdoc/>
    /// <remarks>Anima shares the Qwen-Image VAE; its encoder half is built alongside the decoder and <see cref="AnimaPipeline"/> implements the latent-mask blend.</remarks>
    public ImageFeatures Supports => ImageFeatures.Img2Img | ImageFeatures.Inpaint | ImageFeatures.SeamlessTiling | ImageFeatures.VariationSeed | ImageFeatures.Refiner | ImageFeatures.Lora;

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "anima", StringComparison.OrdinalIgnoreCase);

    /// <summary>Anima's sampling settings: 20 steps, guidance-free (CFG 1.0), 1024x1024 — the Cosmos-Predict2 lineage samples without classifier-free guidance, and the pipeline resolves steps against <c>GenerationDefaults.Generic</c>.</summary>
    public static ImageDefaults FamilyDefaults { get; } = new ImageDefaults { Steps = 20, CfgScale = 1.0f, Width = 1024, Height = 1024 };

    /// <inheritdoc/>
    public ImageDefaults Defaults => FamilyDefaults;

    /// <inheritdoc/>
    public IRecipePipeline Construct(RecipeContext context)
    {
        // TODO(E-IMG-4): honor a user-picked Qwen/VAE override from ImageRequest.Components instead of always taking
        // the canonical SideModels entry (the SwarmUI loader read input.Get(T2IParamTypes.QwenModel/VAE) and
        // header-probed the pick for the Qwen-Image `conv2.weight` / `decoder.conv1.weight` signature keys).
        string qwenPath = ModelDownloader.EnsureSideModelAsync(SideModels.Qwen3_0_6B, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
        string vaePath = ModelDownloader.EnsureSideModelAsync(SideModels.QwenImageVae, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();

        List<IDisposable> loaders = new List<IDisposable>();
        IDisposable? checkpoint = null;
        try
        {
            // 1. Single-file Anima checkpoint → DiT trunk + LlmAdapter buckets, through the one container so a
            // GGUF or fp8_scaled repack of the same file reaches the converter identically.
            CheckpointSource source = CheckpointSource.Open(context.CheckpointPath);
            checkpoint = source;
            AnimaCheckpointConverter.ConvertedWeights converted = AnimaCheckpointConverter.Convert(source.Weights);
            if (converted.Transformer.Count == 0)
            {
                throw new InvalidOperationException("Anima checkpoint contains no DiT trunk weights (looked for net.x_embedder.* / net.blocks.*).");
            }
            if (converted.LlmAdapter.Count == 0)
            {
                throw new InvalidOperationException(
                    "Anima checkpoint contains no net.llm_adapter.* weights. This isn't a Comfy-format Anima single-file " +
                    "(expected ~571 transformer tensors + ~118 llm_adapter tensors).");
            }
            Logs.Info($"[AnimaRecipe] Parsed checkpoint: {converted.Transformer.Count} DiT tensors, {converted.LlmAdapter.Count} llm_adapter tensors, fp8_mix={converted.IsFp8Mix}.");
            // Any quant this run's devices have no packed-weight kernel for widens here rather than failing inside
            // the first GEMM. The adapter runs on the same devices as the trunk, so it is prepared the same way.
            QuantizedWeightPolicy.PreparedWeights prepared =
                QuantizedWeightPolicy.PrepareForBackends(converted.Transformer, context.TransformerBackends);
            checkpoint = new CompositeDisposable(source, prepared);
            QuantizedWeightPolicy.PreparedWeights preparedAdapter =
                QuantizedWeightPolicy.PrepareForBackends(converted.LlmAdapter, context.TransformerBackends);
            checkpoint = new CompositeDisposable(source, prepared, preparedAdapter);

            AnimaConfig animaConfig = AnimaConfig.AnimaPreview3;
            AnimaTransformer transformer = new AnimaTransformer(animaConfig);
            // Merge any requested LoRAs BEFORE LoadWeights — device caches are identity-keyed, so merging
            // after would leave layers serving the pre-merge tensors (the Sd3Recipe ordering rule).
            MergedLoraStack? loraStack = RecipeLoraMerge.Apply(
                context,
                new LoraMergeTargets { Transformer = converted.Transformer },
                "AnimaRecipe");
            transformer.LoadWeights(converted.Transformer);
            AnimaLlmAdapter llmAdapter = new AnimaLlmAdapter(animaConfig.LlmAdapter);
            llmAdapter.LoadWeights(converted.LlmAdapter);

            // 2. Qwen-3 0.6B base text encoder + its embedded tokenizer.
            SafeTensorsLoader qwenLoader = new SafeTensorsLoader();
            qwenLoader.Load(qwenPath);
            loaders.Add(qwenLoader);
            IReadOnlyDictionary<string, Tensor> qwenWeights = qwenLoader.GetAllTensors();
            if (qwenWeights.Count == 0)
            {
                throw new InvalidOperationException($"Qwen-3 0.6B model file '{qwenPath}' has no tensors.");
            }
            LlamaStyleEncoder qwen = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Qwen3_0_6B);
            qwen.LoadWeights(qwenWeights);
            Qwen3Tokenizer tokenizer = new Qwen3Tokenizer(maxLength: 256);

            // Anima's text stack is dual: the Qwen-3 hidden states are the LlmAdapter cross-attention K/V source, while a
            // T5-XXL SentencePiece tokenization of the SAME prompt is the adapter's main-stream lookup (embed[t5_ids]).
            // Only token ids are needed — no T5 encoder model is loaded.
            T5Tokenizer t5Tokenizer = new T5Tokenizer(maxLength: T5MaxTokens);

            // 3. Qwen-Image VAE (3D causal, WAN 2.1 family). Keys use upstream WAN naming verbatim — no normalization.
            SafeTensorsLoader vaeLoader = new SafeTensorsLoader();
            vaeLoader.Load(vaePath);
            loaders.Add(vaeLoader);
            IReadOnlyDictionary<string, Tensor> vaeWeights = vaeLoader.GetAllTensors();
            if (vaeWeights.Count == 0)
            {
                throw new InvalidOperationException($"VAE file '{vaePath}' has no tensors.");
            }
            QwenImageVaeDecoder vae = new QwenImageVaeDecoder(VaeConfig.QwenImage);
            vae.LoadWeights(vaeWeights);
            // Encoder half from the same staged dict; null on a decode-only VAE, which the pipeline refuses by name.
            QwenImageVaeEncoder? vaeEncoder = LoaderVaeUtils.TryBuildQwenEncoder(VaeConfig.QwenImage, vaeWeights, "AnimaRecipe");

            AnimaPipeline pipeline = new AnimaPipeline(context.Backend, transformer, llmAdapter, vae, vaeEncoder, animaConfig);
            Logs.Info("[AnimaRecipe] Anima ready (scheduler=FlowMatchEuler shift=3.0).");
            return new AnimaRecipePipeline(pipeline, qwen, tokenizer, t5Tokenizer, transformer, llmAdapter, context.Backend, checkpoint, loaders, loraStack);
        }
        catch (Exception ex)
        {
            Logs.Error("[AnimaRecipe] Construction failed.", ex);
            foreach (IDisposable loader in loaders)
            {
                loader.Dispose();
            }
            checkpoint?.Dispose();
            throw;
        }
    }
}
