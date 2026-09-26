using HartsyInference.ModelAssets.Checkpoints;
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
    /// <remarks>Ledger evidence in <c>PromptWeightingModeLedgerTests</c>: <c>supported_models.py:1038</c> →
    /// <c>hunyuan_video.HunyuanVideoTokenizer</c> = CLIP-L + <c>LLAMA3Tokenizer</c>, neither of which disables
    /// weights. Only the Llama arm is blended — CLIP-L contributes a pooled vector here — and the blend runs on the
    /// full sequence BEFORE the template crop, which is the order <c>encode_token_weights</c> uses.</remarks>
    public Diffusion.Prompting.PromptWeightingMode PromptWeighting =>
        Diffusion.Prompting.PromptWeightingMode.ComfyBlend;

    /// <inheritdoc/>
    /// <inheritdoc/>
    public MemoryCapabilities MemorySupports => MemoryCapabilities.BlockStreaming;

    public IVideoRecipePipeline Construct(RecipeContext context)
    {
        // TODO(E-IMG-4/5): image-to-video conditioning and a VideoRequest.Components LLaVA/CLIP/VAE override are
        // deferred — this is the text-to-video path only.
        // IDisposable rather than SafeTensorsLoader: the container owns the mapping whatever the format is.
        List<IDisposable> loaders = new List<IDisposable>();
        try
        {
            Logs.Info($"[HunyuanVideoRecipe] Loading + converting DiT: {Path.GetFileName(context.CheckpointPath)}.");
            // One container for either format, and it folds fp8/int8 companions before the converter sees them.
            CheckpointSource ditSource = CheckpointSource.Open(context.CheckpointPath);
            loaders.Add(ditSource);
            Dictionary<string, Tensor> ditWeights = HunyuanVideoCheckpointConverter.Convert(
                new Dictionary<string, Tensor>(ditSource.Weights, StringComparer.Ordinal));
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
            MergedLoraStack? loraStack = RecipeLoraMerge.Apply(
                context,
                new LoraMergeTargets { Transformer = ditWeights },
                "HunyuanVideoRecipe");
            dit.LoadWeights(ditWeights);

            string vaePath = ModelDownloader.EnsureSideModelAsync(SideModels.HunyuanVideoVae3D, onProgress: null, context.Cancel).GetAwaiter().GetResult();
            HunyuanVideoVaeDecoder vae = new HunyuanVideoVaeDecoder();
            vae.LoadWeights(VaePrecisionHelper.CastWeights(HunyuanVideoCheckpointConverter.ConvertVaeDecoder(LoadStandalone(loaders, vaePath)), [DType.BF16], DType.F16));

            // bf16-resident, block-streamed DiT — caching F16 casts would roughly double VRAM and OOM a 24 GB card.
            RecipeBackendFlags.DisableCacheWeightCasts(context, "HunyuanVideoRecipe");

            string llavaPath = ModelDownloader.EnsureSideModelAsync(SideModels.LlavaLlama3, onProgress: null, context.Cancel).GetAwaiter().GetResult();
            LlamaStyleEncoder llava = new LlamaStyleEncoder(LlamaStyleEncoderConfig.LlavaLlama3_8B);
            llava.LoadWeights(TextEncoderQuantNormalizer.Normalize(LoadStandalone(loaders, llavaPath)));

            string clipPath = ModelDownloader.EnsureSideModelAsync(SideModels.ClipL, onProgress: null, context.Cancel).GetAwaiter().GetResult();
            ClipTextEncoder clipL = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipL);
            clipL.LoadWeights(LoadStandalone(loaders, clipPath), "text_model");

            HunyuanVideoPipeline pipeline = new HunyuanVideoPipeline(context.Backend, dit, vae, config);
            Logs.Info("[HunyuanVideoRecipe] HunyuanVideo ready (text-to-video).");
            return new HunyuanVideoRecipePipeline(context.Backend, pipeline, llava, clipL, new ClipTokenizer(), dit, loaders, loraStack);
        }
        catch (Exception ex)
        {
            Logs.Error("[HunyuanVideoRecipe] Construction failed.", ex);
            foreach (IDisposable loader in loaders)
            {
                loader.Dispose();
            }
            throw;
        }
    }

    /// <summary>Builds the templated + BOS-prefixed Llama-3 token sequence the diffusers pipeline feeds LLaVA (add_special_tokens=True → BOS prepended), from the embedded Llama-3 byte-level BPE tokenizer.</summary>
    internal static int[] BuildTemplatedTokens(string prompt) => BuildWeightedTokens(prompt).Tokens;

    /// <summary>The templated Llama-3 ids plus one weight per row, template positions pinned to 1.</summary>
    /// <remarks>ComfyUI blends the FULL sequence and crops afterwards — <c>encode_token_weights</c>
    /// (<c>hunyuan_video.py:104-110</c>) calls <c>self.llama.encode_token_weights</c> first and only then computes
    /// <c>template_end</c> — so the weights returned here are indexed against the uncropped sequence and the caller
    /// must blend before <c>CropSequence</c>.</remarks>
    internal static Diffusion.Prompting.WeightedTokenSequence BuildWeightedTokens(string prompt)
    {
        using Stream json = EmbeddedTokenizerResources.OpenLlama3TokenizerJson();
        HartsyInference.ModelAssets.Tokenizers.GgufTokenizer tok = HfTokenizerJson.LoadByteLevelBpe(json);
        int placeholder = PromptTemplate.IndexOf("{0}", StringComparison.Ordinal);
        int[] prefix = [StartId(tok), .. tok.Encode(PromptTemplate[..placeholder], addSpecial: true)];
        int[] suffix = tok.Encode(PromptTemplate[(placeholder + 3)..], addSpecial: true);
        // The crop is a hard-coded count, so a tokenizer revision that moved the template's length would silently
        // crop into the prompt (or leave template rows in the conditioning) instead of failing. Checked rather than
        // trusted, because both outcomes render plausibly.
        if (prefix.Length != CropStart)
        {
            throw new InvalidOperationException(
                $"HunyuanVideo's prompt template tokenizes to {prefix.Length} ids but CropStart is {CropStart}.");
        }
        return Diffusion.Prompting.TemplatedPromptTokens.Build(
            Diffusion.Prompting.PromptTagFlattening.Flatten(prompt),
            t => Templated(tok, t), t => tok.Encode(t, addSpecial: true), prefix, suffix);
    }

    /// <summary>The whole-template encode, kept verbatim for the unweighted path so wiring weighting moves nothing.</summary>
    private static int[] Templated(HartsyInference.ModelAssets.Tokenizers.GgufTokenizer tok, string prompt)
    {
        string templated = string.Format(System.Globalization.CultureInfo.InvariantCulture, PromptTemplate, prompt);
        int[] ids = tok.Encode(templated, addSpecial: true);
        int[] withBos = new int[ids.Length + 1];
        withBos[0] = StartId(tok);
        Array.Copy(ids, 0, withBos, 1, ids.Length);
        return withBos;
    }

    /// <summary>ComfyUI's <c>LLAMAModel special_tokens={"start": 128000, "pad": 128258}</c> (<c>hunyuan_video.py:32</c>).
    /// <c>LLAMA3Tokenizer</c> sets <c>pad_with_end=False</c> with an explicit <c>pad_token=128258</c>, so the pad is
    /// NOT the Llama end-of-text id.</summary>
    private const int StartTokenId = 128000;
    private const int PadTokenId = 128258;

    private static int StartId(HartsyInference.ModelAssets.Tokenizers.GgufTokenizer tok) => tok.BosId ?? StartTokenId;

    /// <summary>The ComfyBlend baseline for a conditioning of <paramref name="length"/> rows: <c>gen_empty_tokens</c>
    /// emits start + end + padding and this model declares no end, so it is one start token then pad. Built per
    /// prompt because the tokenizer does not pad to a fixed window.</summary>
    internal static int[] EmptyBaseline(int length)
    {
        int[] empty = new int[length];
        empty[0] = StartTokenId;
        Array.Fill(empty, PadTokenId, 1, length - 1);
        return empty;
    }

    /// <summary>Opens a side component through the container. The explicit <c>ApplyFp8ScaledDequant</c> this used to
    /// carry is gone because the container folds companions on open — doing it twice would be harmless, but leaving
    /// it here would suggest the container does not.</summary>
    private static Dictionary<string, Tensor> LoadStandalone(List<IDisposable> loaders, string path)
    {
        CheckpointSource source = CheckpointSource.Open(path);
        loaders.Add(source);
        return new Dictionary<string, Tensor>(source.Weights, StringComparer.Ordinal);
    }
}
