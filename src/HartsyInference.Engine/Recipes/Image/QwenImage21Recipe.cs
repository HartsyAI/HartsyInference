using HartsyInference.Core.Logging;
using HartsyInference.Core.Memory;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Engine.Features;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Qwen-Image 2.1 recipe (Alibaba, ~7B single-stream DiT). A different architecture from
/// <see cref="QwenImageRecipe"/> rather than a revision of it — see <see cref="QwenImage21Config"/> — so it is its
/// own family id, <c>qwen-image-2.1</c>, and its own detector rule. The checkpoint is transformer-only in the Comfy
/// <c>diffusion_models</c> layout, so the Qwen3-VL-8B encoder
/// (<see cref="SideModels.Qwen3VL_8B_QwenImage21"/>) and the 64-channel RGBA VAE
/// (<see cref="SideModels.QwenImage21Vae"/>) resolve as side models.</summary>
public sealed class QwenImage21Recipe : IArchitectureRecipe
{
    /// <inheritdoc/>
    public string Name => "qwen-image-2.1";

    /// <summary>The system turn Qwen-Image 2.1 wraps every prompt in. Dropped from the conditioning afterwards, but
    /// present during encoding — the encoder is causal, so it shapes the rows that are kept.</summary>
    public const string SystemPrompt = "Comprehend and analyze the provided prompt.";

    /// <inheritdoc/>
    /// <remarks>Text-to-image only in this pass. Reference-image editing (the model's other half: VAE latents
    /// spliced into the sequence at vision-token slots, per ComfyUI's <c>TextEncodeQwenImage21</c>) needs the
    /// Wan 2.2 VAE <b>encoder</b> parameterized the same way the decoder now is, plus the interleaved
    /// text/reference sequence in <see cref="QwenImage21Transformer"/>. LoRA is likewise deferred: adapters address
    /// <c>img_mlp.gate_layer</c>/<c>proj</c>, the two halves of the fused <c>gate_up</c> this loads whole, so the
    /// merge needs a row-range split that also slices the int8 build's per-row scale.</remarks>
    public ImageFeatures Supports => ImageFeatures.SeamlessTiling | ImageFeatures.VariationSeed;

    /// <inheritdoc/>
    /// <remarks>Ledger evidence in <c>PromptWeightingModeLedgerTests</c>. <c>QwenImage21Tokenizer</c> subclasses
    /// <c>Qwen3VLTokenizer</c>, whose <c>tokenize_with_weights</c> passes <c>disable_weights=True</c>, so SwarmUI's
    /// probe puts the family on CondScale: encode at weight 1, then scale each token's cond row — here after the
    /// system-turn drop, which is what makes the right-alignment offset negative.</remarks>
    public Diffusion.Prompting.PromptWeightingMode PromptWeighting => Diffusion.Prompting.PromptWeightingMode.CondScale;

    /// <inheritdoc/>
    public bool Matches(string familyId) =>
        string.Equals(familyId, "qwen-image-2.1", StringComparison.OrdinalIgnoreCase)
        || string.Equals(familyId, "qwen-image-21", StringComparison.OrdinalIgnoreCase);

    /// <summary>The settings ComfyUI's shipped <c>image_qwen_image_2_1_t2i</c> template starts at: 25 steps, euler,
    /// CFG 1.0, 1024×1024. CFG 1 is the official path — the negative prompt is unused unless it is raised. The
    /// upstream pipeline uses 40–50 steps; 25 is the template's own default and what the benchmark measures.</summary>
    public static ImageDefaults FamilyDefaults { get; } = new ImageDefaults { Steps = 25, CfgScale = 1.0f, Width = 1024, Height = 1024 };

    /// <inheritdoc/>
    public ImageDefaults Defaults => FamilyDefaults;

    /// <inheritdoc/>
    /// <remarks>None declared. Component placement would need the encoder and VAE to actually run on
    /// <c>TextEncoderBackend</c>/<c>VaeBackend</c>; the pipeline currently runs both on the single request backend,
    /// and declaring a capability the pipeline does not consume is how a user gets a setting that silently does
    /// nothing. The encoder/DiT residency swap that makes a 24 GB card work is unconditional and needs no flag.</remarks>
    public MemoryCapabilities MemorySupports => MemoryCapabilities.None;

    public IRecipePipeline Construct(RecipeContext context)
    {
        List<SafeTensorsLoader> loaders = new List<SafeTensorsLoader>();
        IDisposable? checkpoint = null;
        try
        {
            CheckpointSource source = CheckpointSource.Open(context.CheckpointPath);
            checkpoint = source;
            if (!QwenImage21CheckpointConverter.MatchesByKeys([.. source.Weights.Keys]))
            {
                throw new InvalidOperationException(
                    $"'{Path.GetFileName(context.CheckpointPath)}' is not a Qwen-Image 2.1 checkpoint: it lacks one of "
                    + string.Join(", ", QwenImage21CheckpointConverter.SignatureKeys)
                    + ". Qwen-Image v1 shares the transformer_blocks naming but is a different architecture — use -m qwen-image for it.");
            }
            QwenImage21CheckpointConverter.ConvertedWeights converted = QwenImage21CheckpointConverter.Convert(source.Weights);
            Logs.Info($"[QwenImage21Recipe] Parsed checkpoint: {converted.Transformer.Count} transformer tensors.");

            QwenImage21Config config = ConfigFromWeights(converted.Transformer);
            QwenImage21Transformer transformer = new QwenImage21Transformer(config);
            QuantizedWeightPolicy.PreparedWeights prepared =
                QuantizedWeightPolicy.PrepareForBackends(converted.Transformer, context.TransformerBackends);
            checkpoint = new CompositeDisposable(source, prepared);
            transformer.LoadWeights(converted.Transformer);

            // ── Text encoder: Qwen3-VL-8B, last decoder layer with NO final norm. ──
            // ComfyUI sets layer_norm_hidden_state=False for this model, matching transformers 4.57's
            // hidden_states[-1]; 5.x norms that tap and Qwen's results are not tuned to it. The vision tower is
            // dropped here because text-to-image never reaches it.
            string encoderPath = ModelDownloader.EnsureSideModelAsync(
                SideModels.Qwen3VL_8B_QwenImage21, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
            SafeTensorsLoader encoderLoader = new SafeTensorsLoader();
            encoderLoader.Load(encoderPath);
            loaders.Add(encoderLoader);
            LlamaStyleEncoder textEncoder = new LlamaStyleEncoder(
                LlamaStyleEncoderConfig.Qwen3_VL_8B with { HasFinalNorm = false });
            textEncoder.LoadWeights(encoderLoader.GetAllTensors());

            // ── VAE: the Wan 2.2 decoder with this model's parameters. dec_dim 144 and the five-stage
            // [1,2,4,8,8] ladder are read back from the file rather than assumed, so a differently-sized repack
            // fails on a shape mismatch at load instead of halfway through a decode. ──
            string vaePath = ModelDownloader.EnsureSideModelAsync(
                SideModels.QwenImage21Vae, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
            SafeTensorsLoader vaeLoader = new SafeTensorsLoader();
            vaeLoader.Load(vaePath);
            loaders.Add(vaeLoader);
            Dictionary<string, Tensor> vaeWeights =
                VaePrecisionHelper.CastWeights(vaeLoader.GetAllTensors(), [DType.F16, DType.BF16], DType.F32);
            Wan22VaeDecoder vae = BuildVae(vaeWeights);
            vae.LoadWeights(vaeWeights);

            QwenImage21Pipeline pipeline = new QwenImage21Pipeline(context.Backend, textEncoder, transformer, vae, config);
            Qwen3Tokenizer tokenizer = new Qwen3Tokenizer(maxLength: 1024);
            Logs.Info($"[QwenImage21Recipe] Ready ({config.Depth} blocks, hidden {config.HiddenSize}; "
                + "Qwen3-VL-8B encoder, no final norm; flow-match Euler at shift 0.69).");
            return new QwenImage21RecipePipeline(pipeline, tokenizer, textEncoder, transformer, vae, loaders, checkpoint);
        }
        catch (Exception ex)
        {
            Logs.Error("[QwenImage21Recipe] Construction failed.", ex);
            foreach (SafeTensorsLoader loader in loaders)
            {
                loader.Dispose();
            }
            checkpoint?.Dispose();
            throw;
        }
    }

    /// <summary>Derives the config from the checkpoint rather than pinning the released dimensions, mirroring
    /// ComfyUI's <c>detect_unet_config</c>. The MLP ratio comes from the fused <c>gate_up</c>'s row count, which is
    /// <c>2 · ratio · hidden</c>.</summary>
    private static QwenImage21Config ConfigFromWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        int headDim = (int)weights["transformer_blocks.0.attn.norm_q.weight"].Shape[0];
        Tensor imgIn = weights["img_in.weight"];
        int hidden = (int)imgIn.Shape[0];
        int depth = 0;
        while (weights.ContainsKey($"transformer_blocks.{depth}.attn.to_q.weight"))
        {
            depth++;
        }
        if (!weights.TryGetValue("transformer_blocks.0.img_mlp.gate_up.weight", out Tensor? gateUp))
        {
            throw new NotSupportedException(
                "This Qwen-Image 2.1 checkpoint stores img_mlp as separate gate_layer/proj projections. Only the "
                + "fused gate_up layout the Comfy-Org release ships is implemented.");
        }
        return new QwenImage21Config
        {
            HiddenSize = hidden,
            NumHeads = hidden / headDim,
            HeadDim = headDim,
            Depth = depth,
            InChannels = (int)imgIn.Shape[1],
            OutChannels = (int)weights["proj_out.weight"].Shape[0],
            ContextDim = (int)weights["txt_in.text_norm.weight"].Shape[0],
            MlpRatio = (int)gateUp.Shape[0] / 2 / hidden,
        };
    }

    /// <summary>Builds the decoder from the VAE file's own shapes: encoder width, decoder width and channel count
    /// are all read back, the way ComfyUI's <c>sd.py</c> branch does.</summary>
    private static Wan22VaeDecoder BuildVae(IReadOnlyDictionary<string, Tensor> weights)
    {
        int decDim = (int)weights["decoder.head.0.gamma"].Shape[0];
        int zDim = (int)weights["conv2.weight"].Shape[0];
        return new Wan22VaeDecoder(
            dim: decDim,
            zDim: zDim,
            dimMult: [1, 2, 4, 8, 8],
            numResBlocks: 2,
            // temperal_downsample [false, true, true, true] reversed; stage 3 has no time_conv in the file.
            temperalUpsample: [true, true, true, false],
            patchSize: 1,
            temporalKernel: 1,
            latentMean: QwenImage21LatentNorm.Mean,
            latentStd: QwenImage21LatentNorm.Std);
    }
}
