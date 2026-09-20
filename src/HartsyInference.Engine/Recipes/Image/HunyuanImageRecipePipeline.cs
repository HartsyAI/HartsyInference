using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
using System.Globalization;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Diffusion.Prompting;

using HartsyInference.Engine.Features;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>A constructed HunyuanImage 2.1 pipeline driven against the native <see cref="ImageRequest"/>. <see cref="HunyuanImagePipeline"/> owns the Qwen2.5-VL-7B forward (including its own TE⇄DiT residency swap and embedding cache), so this only produces the padded chat-template token ids + attention masks and calls <see cref="HunyuanImagePipeline.GenerateFromTokens"/>. Mirrors the SwarmUI backend's <c>HunyuanImageLoader.Generate</c>. Wraps the constructed HunyuanImage pipeline plus its components, taking ownership of every disposable.</summary>
public sealed class HunyuanImageRecipePipeline(HunyuanImagePipeline pipeline, Qwen2Tokenizer tokenizer, LlamaStyleEncoder llama,
    HunyuanImageQwenTextEncoder qwenEncoder, HunyuanImageTransformer transformer, HunyuanImageVaeDecoder vae,
    List<SafeTensorsLoader> loaders, IDisposable? ggufHandle, MergedLoraStack? loraStack = null) : IRecipePipeline
{

    private readonly HunyuanImagePipeline _pipeline = pipeline;
    private readonly Qwen2Tokenizer _tokenizer = tokenizer;
    private readonly LlamaStyleEncoder _llama = llama;
    private readonly HunyuanImageQwenTextEncoder _qwenEncoder = qwenEncoder;
    private readonly HunyuanImageTransformer _transformer = transformer;
    private readonly HunyuanImageVaeDecoder _vae = vae;
    private readonly List<SafeTensorsLoader> _loaders = loaders;
    private readonly IDisposable? _ggufHandle = ggufHandle;
    private readonly MergedLoraStack? _loraStack = loraStack;

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string prompt = request.Prompt;
        string negative = request.NegativePrompt ?? "";
        int steps = request.Steps ?? HunyuanImageRecipe.FamilyDefaults.Steps;
        float cfg = request.CfgScale ?? HunyuanImageRecipe.FamilyDefaults.CfgScale;
        // 32× VAE + unit patches → the image dims must be a multiple of 32.
        (int reqWidth, int reqHeight) = RecipeRequestMapper.Size(request);
        int width = (reqWidth / 32) * 32;
        int height = (reqHeight / 32) * 32;

        // TODO(E-IMG-4/5): img2img/inpaint, LoRA, ControlNet, regional prompting and the ByT5 glyph branch are
        // deferred — text-to-image only.
        (int[] ids, int[] mask, float[]? weights) = TokenizePadded(_tokenizer, prompt);
        bool useCfg = cfg > 1.0f;
        // An empty negative encodes to exactly the 34-token template, which the encoder's prefix-drop rejects —
        // give it one real token.
        if (useCfg && string.IsNullOrWhiteSpace(negative))
        {
            negative = ".";
        }
        int[]? negIds = null;
        int[]? negMask = null;
        float[]? negWeights = null;
        if (useCfg)
        {
            (negIds, negMask, negWeights) = TokenizePadded(_tokenizer, negative);
        }

        // Resolved at the same width/height the inner request carries — HunyuanImage rejects sizes that are not a
        // multiple of 32 rather than snapping, so these are already the dimensions the pipeline validates against.
        using Img2ImgResolver.Img2ImgSpec? img2img = RecipeImg2ImgBinder.Resolve(request, width, height);
        TextToImageRequest inner = RecipeImg2ImgBinder.Apply(
            new TextToImageRequest
            {
                SeamlessTiling = request.SeamlessTiling,
                    VariationSeed = request.VariationSeed?.Seed ?? -1,
                    VariationSeedStrength = request.VariationSeed?.Strength ?? 0,
                Prompt = prompt,
                Width = width,
                Height = height,
                Steps = steps,
                CfgScale = cfg,
                Seed = RecipeRequestMapper.MapSeed(request.Seed),
                // Routed through the resolver rather than read raw, so an unavailable sampler is refused by name here —
                // before the checkpoint loads — instead of deep inside the pipeline, or silently dropped.
                Scheduler = SamplingParamResolver.ResolveSchedulerName(request),
            },
            img2img);

        Action<GenerationProgress> bridge = RecipeProgressAdapter.Create(progress, cancel, totalSteps: steps);

        (byte[] rgb, int outW, int outH, int usedSeed) = _pipeline.GenerateFromTokens(
            ids, mask, negIds, negMask, inner, onProgress: bridge,
            promptTokenWeights: weights, negativeTokenWeights: negWeights);

        return new ImageResult
        {
            Rgb = rgb,
            Width = outW,
            Height = outH,
            Seed = usedSeed,
            Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["arch"] = "hunyuan-image",
                ["size"] = $"{outW}x{outH}",
                ["seed"] = usedSeed.ToString(CultureInfo.InvariantCulture),
                ["steps"] = steps.ToString(CultureInfo.InvariantCulture),
                ["cfg"] = cfg.ToString(CultureInfo.InvariantCulture),
            },
        };
    }

    /// <summary>Chat-template encode padded to the fixed 1034-token window with a matching attention mask (diffusers <c>_get_qwen_prompt_embeds</c>), plus the per-token weights parsed from the emphasis grammar.</summary>
    /// <remarks>The weights are returned at the sequence's REAL length, not padded to 1034. The encoder trims to
    /// the mask's real length, encodes that, then slices <c>[34, realLen)</c> — so right-aligning <c>realLen</c>
    /// weights against <c>realLen − 34</c> conditioning rows gives offset −34 and the template weights fall off
    /// the front exactly as SwarmUI intends. Handing the padded 1034 array through instead would give offset
    /// <c>keep − 1034</c> and push every prompt weight off the front: a silent no-op, not an error.</remarks>
    internal static (int[] ids, int[] mask, float[]? weights) TokenizePadded(Qwen2Tokenizer tokenizer, string prompt)
    {
        WeightedTokenSequence sequence = TemplatedPromptTokens.Build(
            PromptTagFlattening.Flatten(prompt),
            t => tokenizer.EncodeChat(t, systemPrompt: HunyuanImageQwenTextEncoder.SystemPrompt, addGenerationPrompt: false),
            // Legacy EncodeRaw, deliberately, because that is what EncodeChat's own AppendBpe uses: byte-identity
            // is measured against OUR base path, not against the HF fast tokenizer. See the TODO below.
            tokenizer.EncodeRaw, TemplatePrefix(tokenizer), [Qwen2Tokenizer.ImEndId]);
        int[] raw = sequence.Tokens;
        int realLen = Math.Min(raw.Length, HunyuanImageQwenTextEncoder.PaddedLength);
        int[] ids = Qwen2Tokenizer.PadToLength(raw, HunyuanImageQwenTextEncoder.PaddedLength);
        int[] mask = new int[HunyuanImageQwenTextEncoder.PaddedLength];
        for (int i = 0; i < realLen; i++)
        {
            mask[i] = 1;
        }
        return (ids, mask, sequence.IsUniformlyUnweighted ? null : sequence.Weights[..realLen]);
    }

    /// <summary>The ids <see cref="Qwen2Tokenizer.EncodeChat"/> puts before the prompt, assembled the same way it
    /// does, and checked against what that method actually emits for an empty prompt rather than against a
    /// constant — a drift between the split and the whole-string encode is what would silently misplace every
    /// weight.</summary>
    /// <remarks>
    /// <para>TODO — PRE-EXISTING, out of scope for weighting, unverified against HF. This prefix is <b>33</b>
    /// ids, but <see cref="HunyuanImageQwenTextEncoder.TemplatePrefixTokens"/> is 34 and the encoder slices from
    /// there, so the prompt's FIRST token's hidden state is dropped from the conditioning on every generation.
    /// The cause is one step down: <c>EncodeRaw("\n")</c> returns ZERO ids on this tokenizer, so the newline
    /// between <c>&lt;|im_end|&gt;</c> and the next <c>&lt;|im_start|&gt;</c> vanishes, where HF emits id 198.
    /// diffusers' <c>prompt_template_encode_start_idx</c> of 34 is right for HF's tokenization and one too many
    /// for ours. Fixing it means fixing the tokenizer, which moves every existing HunyuanImage generation.</para>
    /// <para>Weighting is unaffected by that discrepancy: the weights are indexed by sequence POSITION and
    /// right-aligned against a cond of <c>realLen − 34</c> rows, so weight <c>i</c> lands on sequence position
    /// <c>i</c> whatever the template's true length is. The dropped token simply loses its weight with it.</para>
    /// <para>Weighted spans also inherit the leading-space drop documented on
    /// <see cref="Qwen2Tokenizer.EncodeRaw"/>. The base encode has the same bug, so the two agree and the
    /// unweighted path is byte-identical; <see cref="Qwen2Tokenizer.EncodeRawByteLevel"/> is the fix for both.</para>
    /// </remarks>
    private static int[] TemplatePrefix(Qwen2Tokenizer tokenizer)
    {
        List<int> prefix = new List<int>(48) { Qwen2Tokenizer.ImStartId };
        prefix.AddRange(tokenizer.EncodeRaw("system\n" + HunyuanImageQwenTextEncoder.SystemPrompt));
        prefix.Add(Qwen2Tokenizer.ImEndId);
        prefix.AddRange(tokenizer.EncodeRaw("\n"));
        prefix.Add(Qwen2Tokenizer.ImStartId);
        prefix.AddRange(tokenizer.EncodeRaw("user\n"));
        int wholeTemplate = tokenizer.EncodeChat(
            "", systemPrompt: HunyuanImageQwenTextEncoder.SystemPrompt, addGenerationPrompt: false).Length;
        if (prefix.Count + 1 != wholeTemplate)
        {
            throw new InvalidOperationException(
                $"HunyuanImage's split chat template is {prefix.Count} prefix + 1 suffix ids but EncodeChat "
                + $"emits {wholeTemplate} for an empty prompt — the two must agree or every weight is misplaced.");
        }
        return [.. prefix];
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _pipeline.Dispose();
        _qwenEncoder.Dispose();
        _llama.Dispose();
        _transformer.Dispose();
        _vae.Dispose();
        foreach (SafeTensorsLoader loader in _loaders)
        {
            loader.Dispose();
        }
        _ggufHandle?.Dispose();
        // Last: the stack owns the merged weight tensors the transformer was serving.
        _loraStack?.Dispose();
    }
}
