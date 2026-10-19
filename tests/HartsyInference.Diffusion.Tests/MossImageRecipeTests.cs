using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Variants;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>MOSS-Image's engine wiring outside the weights: the Qwen3.5 thinking template and its emphasis split, the
/// layout JSON a regional prompt becomes (OpenMOSS <c>compile_prompt.py</c>'s format and box rule), the <c>slg.*</c>
/// request extras, the diffusers folder a checkpoint path resolves to, the SFT build named by its folder, and the family
/// routing.</summary>
public sealed class MossImageRecipeTests : IDisposable
{
    private const string Prefix = MossImagePromptTokenizer.TemplatePrefix;
    private const string Suffix = MossImagePromptTokenizer.TemplateSuffix;

    private readonly string _tempDir = Directory.CreateTempSubdirectory("moss_image_recipe_").FullName;

    [Fact]
    public void Tokenize_WrapsThePromptInTheThinkingTemplate()
    {
        CharTokenizer chars = new();
        WeightedTokenSequence sequence = new MossImagePromptTokenizer(chars).Tokenize("a cat");
        Assert.Equal(chars.Encode(Prefix + "a cat" + Suffix, addSpecial: true), sequence.Tokens);
        Assert.All(sequence.Weights, w => Assert.Equal(1f, w));
        Assert.Equal(CharTokenizer.ImStart, sequence.Tokens[0]);
        Assert.Equal(CharTokenizer.Think, sequence.Tokens[^2]);
    }

    [Fact]
    public void Tokenize_WeightsOnlyTheEmphasizedSpan()
    {
        CharTokenizer chars = new();
        WeightedTokenSequence sequence = new MossImagePromptTokenizer(chars).Tokenize("a (cat:1.5)");
        Assert.Equal(chars.Encode(Prefix + "a cat" + Suffix, addSpecial: true), sequence.Tokens);
        int start = chars.Encode(Prefix + "a ", addSpecial: true).Length;
        for (int i = 0; i < sequence.Tokens.Length; i++)
            Assert.Equal(i >= start && i < start + 3 ? 1.5f : 1f, sequence.Weights[i]);
    }

    [Fact]
    public void Tokenize_NormalizesToNfc_AndTruncatesToTheBudget()
    {
        CharTokenizer chars = new();
        Assert.Contains((int)'é', new MossImagePromptTokenizer(chars).Tokenize("café").Tokens);
        Assert.Equal(10, new MossImagePromptTokenizer(chars, maxTokens: 10).Tokenize("a long prompt").Tokens.Length);
    }

    [Fact]
    public void Tokenize_EditPrompt_PutsThePictureBlocksFirstInTheUserTurn()
    {
        CharTokenizer chars = new();
        VisionPlaceholderPrompt placeholders = new(chars);
        string blocks = placeholders.Render([2, 3]);
        Assert.Equal("Picture 1: <|vision_start|><|image_pad|><|image_pad|><|vision_end|>"
            + "Picture 2: <|vision_start|><|image_pad|><|image_pad|><|image_pad|><|vision_end|>", blocks);
        WeightedTokenSequence sequence = new MossImagePromptTokenizer(chars).Tokenize("make it blue", blocks, 5, placeholders.ImagePadId);
        Assert.Equal(chars.Encode(Prefix + blocks + "make it blue" + Suffix, addSpecial: true), sequence.Tokens);
        Assert.Equal(5, placeholders.CountPlaceholders(sequence.Tokens));
        WeightedTokenSequence weighted = new MossImagePromptTokenizer(chars).Tokenize("make it (blue:1.3)", blocks, 5, placeholders.ImagePadId);
        Assert.Equal(sequence.Tokens, weighted.Tokens);
    }

    [Fact]
    public void Tokenize_EditPrompt_RefusesTruncationAndMiscountedPlaceholders()
    {
        CharTokenizer chars = new();
        VisionPlaceholderPrompt placeholders = new(chars);
        string blocks = placeholders.Render([4]);
        Assert.Throws<ArgumentException>(() => new MossImagePromptTokenizer(chars, maxTokens: 20).Tokenize("x", blocks, 4, placeholders.ImagePadId));
        Assert.Throws<InvalidDataException>(() => new MossImagePromptTokenizer(chars).Tokenize("x", blocks, 3, placeholders.ImagePadId));
        Assert.Throws<InvalidDataException>(() => new VisionPlaceholderPrompt(new CharTokenizer(withVision: false)));
    }

    [Fact]
    public void ReferencePolicy_DefaultsToTheOstrisBudgets_AndReadsTheExtras()
    {
        ReferenceImagePolicy defaults = MossImageRecipePipeline.ReferencePolicy(null);
        Assert.Equal(3, defaults.MaxReferences);
        Assert.Equal(1024 * 1024, defaults.LatentTargetArea);
        Assert.Equal(384 * 384, defaults.VisionTargetArea);
        Assert.False(defaults.LatentUpscale);
        Assert.False(defaults.VisionUpscale);
        Assert.Equal(ReferenceOverflow.Refuse, defaults.Overflow);
        ReferenceImagePolicy custom = MossImageRecipePipeline.ReferencePolicy(new Dictionary<string, object>
        {
            [RequestExtras.ReferenceMaxPixels] = 0,
            [RequestExtras.ReferenceVisionPixels] = "65536",
        });
        Assert.Equal(0, custom.LatentTargetArea);
        Assert.Equal(65536, custom.VisionTargetArea);
        Assert.False(RequestExtras.Bool(new Dictionary<string, object> { [RequestExtras.ReferenceKvCache] = "off" }, RequestExtras.ReferenceKvCache, true));
        Assert.True(RequestExtras.Bool(null, RequestExtras.ReferenceKvCache, true));
    }

    [Fact]
    public void SceneLayout_MatchesOpenMossCompiledFormat()
    {
        string layout = LayoutPromptBuilder.BuildSceneLayout("a poster <region:0.25,0.5,0.5,0.25> (a red \"cat\":1.2)", 1024, 1024);
        Assert.Equal("{\"summary\":{\"caption\":\"a poster\"},\"items\":[{\"label\":\"object\",\"caption\":\"a red \\\"cat\\\"\","
            + "\"bbox\":[0,-256,256,256]}],\"bbox_coord_space\":\"abs_centralized\",\"bbox_format\":\"yxyx\"}", layout);
        Assert.Equal("plain prompt", LayoutPromptBuilder.BuildSceneLayout("plain prompt", 1024, 1024));
        Assert.StartsWith("{\"summary\":{\"caption\":\"a dog; a cat\"}",
            LayoutPromptBuilder.BuildSceneLayout("<region:0,0,0.5,1> a dog <region:0.5,0,0.5,1> a cat", 512, 512), StringComparison.Ordinal);
    }

    /// <summary>OpenMOSS's own worked example (references/layout.md): the cup at <c>[176, 224, 800, 864]</c> (xyxy) on a
    /// 1536×1024 canvas compiles to <c>[-288, -592, 352, 32]</c>.</summary>
    [Fact]
    public void CenteredPixelBox_ReproducesTheReferenceExample_AndIgnoresFloatNoise()
    {
        Assert.Equal((-288, -592, 352, 32),
            LayoutPromptBuilder.CenteredPixelBox(176f / 1536, 224f / 1024, 624f / 1536, 640f / 1024, 1536, 1024));
        Assert.Equal((-200, -200, 200, 200), LayoutPromptBuilder.CenteredPixelBox(0.3f, 0.3f, 0.4f, 0.4f, 1000, 1000));
        Assert.Equal((-512, -512, 512, 512), LayoutPromptBuilder.CenteredPixelBox(-0.5f, -0.5f, 2f, 2f, 1024, 1024));
    }

    [Fact]
    public void IntList_ReadsStringsNumbersAndSequences()
    {
        Dictionary<string, object> extra = new()
        {
            ["a"] = "22, 24 26",
            ["b"] = "none",
            ["c"] = 7,
            ["d"] = new[] { 1, 2 },
            ["e"] = "22,x",
            ["f"] = System.Text.Json.JsonDocument.Parse("[3, 4]").RootElement,
        };
        Assert.Equal([22, 24, 26], RequestExtras.IntList(extra, "a"));
        Assert.Empty(RequestExtras.IntList(extra, "b")!);
        Assert.Equal([7], RequestExtras.IntList(extra, "c"));
        Assert.Equal([1, 2], RequestExtras.IntList(extra, "d"));
        Assert.Null(RequestExtras.IntList(extra, "e"));
        Assert.Equal([3, 4], RequestExtras.IntList(extra, "f"));
        Assert.Null(RequestExtras.IntList(extra, "absent"));
    }

    [Fact]
    public void Guidance_DefaultsToTheReference_AndReadsEveryExtra()
    {
        Assert.Equal(MossImageGuidance.Reference.UncondSkipLayers, MossImageRecipePipeline.GuidanceFrom(null).UncondSkipLayers);
        Assert.False(MossImageRecipePipeline.GuidanceFrom(null).HasSlgBranch);
        MossImageGuidance custom = MossImageRecipePipeline.GuidanceFrom(new Dictionary<string, object>
        {
            [RequestExtras.SlgUncondSkipLayers] = "none",
            [RequestExtras.SlgLayers] = "22,24",
            [RequestExtras.SlgScale] = 3.0,
            [RequestExtras.SlgStart] = 0.01,
            [RequestExtras.SlgStop] = 0.9,
        });
        Assert.Empty(custom.UncondSkipLayers);
        Assert.Equal([22, 24], custom.SlgLayers);
        Assert.True(custom.HasSlgBranch);
        Assert.Equal(0.01f, custom.Start);
        Assert.Equal(0.9f, custom.Stop);
    }

    [Fact]
    public void PipelineRoot_IsFoundFromTheFolderAndAnyFileInIt_ButNotBesideAComfyExport()
    {
        string root = Path.Combine(_tempDir, "MOSS-Image-Base");
        string transformer = Path.Combine(root, "transformer");
        Directory.CreateDirectory(transformer);
        string shard = Path.Combine(transformer, "diffusion_pytorch_model-00001-of-00002.safetensors");
        File.WriteAllBytes(shard, []);
        Assert.Equal(root, MossImageRecipe.PipelineRoot(root));
        Assert.Equal(root, MossImageRecipe.PipelineRoot(transformer));
        Assert.Equal(root, MossImageRecipe.PipelineRoot(shard));
        Assert.Equal(transformer, MossImageRecipe.TransformerPath(root, root));
        Assert.Equal(shard, MossImageRecipe.TransformerPath(shard, root));

        string comfy = Path.Combine(_tempDir, "models", "diffusion_models", "moss-image-dit.safetensors");
        Directory.CreateDirectory(Path.GetDirectoryName(comfy)!);
        Directory.CreateDirectory(Path.Combine(_tempDir, "models", "transformer"));
        File.WriteAllBytes(comfy, []);
        Assert.Null(MossImageRecipe.PipelineRoot(comfy));
    }

    [Fact]
    public void Variant_SftIsNamedByTheFileOrFolder_AndBaseIsTheDefault()
    {
        Assert.Equal("sft", Classify(["moss-image-sft-dit"]).Variant.Id);
        Assert.Equal("sft", Classify(["diffusion_pytorch_model-00001-of-00004", "transformer", "MOSS-Image-SFT"]).Variant.Id);
        Assert.Equal("base", Classify(["diffusion_pytorch_model-00001-of-00004", "transformer", "MOSS-Image-Base"]).Variant.Id);
        Assert.Equal("sft", ModelVariantResolver.Classify(MossImageVariants.Catalog, CheckpointProbe.Empty, ["sft"]).Variant.Id);
    }

    [Fact]
    public void Family_RoutesToTheRecipe_WithTheReferenceDefaultsAndFullSamplerSeam()
    {
        IArchitectureRecipe? recipe = RecipeRegistry.Resolve("moss-image");
        Assert.IsType<MossImageRecipe>(recipe);
        Assert.Same(recipe, RecipeRegistry.Resolve("moss_image"));
        Assert.Equal(50, recipe!.Defaults.Steps);
        Assert.Equal(3f, recipe.Defaults.CfgScale);
        Assert.Equal(PromptWeightingMode.CondScale, recipe.PromptWeighting);
        Assert.True(recipe.Supports.HasFlag(ImageFeatures.RefEdit));
        Assert.Equal(new ImageInputLimits(4, false, 3), recipe.InputLimits);
        Assert.NotEmpty(SamplingCapabilities.ForImage("moss-image").Samplers);
    }

    private static ResolvedModelVariant Classify(IReadOnlyList<string> fileNames) =>
        ModelVariantResolver.Classify(MossImageVariants.Catalog, CheckpointProbe.Empty with { FileNames = fileNames }, []);

    public void Dispose()
    {
        try
        {
            Directory.Delete(_tempDir, recursive: true);
        }
        catch (IOException)
        {
            // best-effort cleanup
        }
    }

    /// <summary>One id per UTF-16 unit, with Qwen3.5's three template specials matched first as the HF tokenizer
    /// matches its added tokens: enough to pin where the template and the weights land.</summary>
    private sealed class CharTokenizer(bool withVision = true) : ILlmTokenizer
    {
        public const int ImStart = 100_000, ImEnd = 100_001, Think = 100_002, VisionStart = 100_003, ImagePad = 100_004, VisionEnd = 100_005;

        private readonly (string Literal, int Id)[] Specials = withVision
            ? [("<|im_start|>", ImStart), ("<|im_end|>", ImEnd), ("<think>", Think), ("<|vision_start|>", VisionStart),
                ("<|image_pad|>", ImagePad), ("<|vision_end|>", VisionEnd)]
            : [("<|im_start|>", ImStart), ("<|im_end|>", ImEnd), ("<think>", Think)];

        public int? BosId => null;

        public int? EosId => ImEnd;

        public IReadOnlyList<int> StopIds => [ImEnd];

        public string? BosToken => null;

        public string? EosToken => "<|im_end|>";

        public int[] Encode(string text, bool addSpecial)
        {
            List<int> ids = [];
            for (int i = 0; i < text.Length;)
            {
                (string Literal, int Id) special = addSpecial
                    ? Specials.FirstOrDefault(s => string.CompareOrdinal(text, i, s.Literal, 0, s.Literal.Length) == 0)
                    : default;
                if (special.Literal is not null)
                {
                    ids.Add(special.Id);
                    i += special.Literal.Length;
                }
                else
                {
                    ids.Add(text[i++]);
                }
            }
            return [.. ids];
        }

        public int[] EncodeOrdinary(string text) => Encode(text, addSpecial: false);

        public string Decode(IReadOnlyList<int> ids) => throw new NotSupportedException();

        public int? SpecialId(string token) => Specials.FirstOrDefault(s => s.Literal == token) is { Literal: not null } s ? s.Id : null;
    }
}
