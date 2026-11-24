using System.Text;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Variants;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>How the FLUX.2 recipe tells Open Vision Banana from the distilled Klein builds it is shaped like, what it declares
/// for it, and that its task prompts reach the text encoder untouched.</summary>
public sealed class OpenVisionBananaRecipeTests
{
    // The FunAILab/Open-Vision-Banana README prompts (PROMPTS.md @ d457d3f), verbatim.
    private const string DepthPrompt =
        "Estimate a dense relative inverse-depth map for the input image. Produce exactly one output value "
        + "for every input pixel at the same pixel location, with closer points brighter and farther points "
        + "darker. Encode the result as neutral grayscale (R=G=B) using one consistent mapping over the "
        + "complete image. Preserve object silhouettes, thin structures, occlusion boundaries, smooth surfaces, "
        + "and small depth changes. Do not alter the scene or add texture, shading, color, text, labels, "
        + "legends, borders, or objects; return only the registered depth map.";

    private const string NormalsPrompt =
        "Create a dense camera-space unit surface-normal map registered to the input. Use +x image-right, "
        + "+y image-up, +z toward the camera and encode (x,y,z) as R=(1-x)/2, G=(1+y)/2, B=(1+z)/2. "
        + "Preserve planes, curved surfaces, thin structures, and sharp orientation boundaries; "
        + "return only the RGB normal map.";

    [Theory]
    [InlineData(3072, "open-vision-banana-4b")]
    [InlineData(3072, "ovb-4b")]
    [InlineData(4096, "open-vision-banana-9b")]
    [InlineData(4096, "ovb-9b")]
    public void TheCatalogIdSelectsTheFineTuneOverItsBasesStructure(int hidden, string hint)
    {
        ResolvedModelVariant variant = ModelVariantResolver.Classify(Flux2Variants.Catalog, Probe(hidden), [hint]);

        Assert.True(Flux2Variants.IsOpenVisionBanana(variant));
        Assert.Equal(hidden == 3072 ? "open-vision-banana-4b" : "open-vision-banana-9b", variant.Id);
    }

    [Theory]
    [InlineData(3072, "klein-4b")]
    [InlineData(4096, "klein-9b")]
    public void WithoutAHintTheWeightsAreTheirKleinBase(int hidden, string expected)
    {
        Assert.Equal(expected, ModelVariantResolver.Classify(Flux2Variants.Catalog, Probe(hidden), []).Id);
    }

    [Fact]
    public void ConfigsFollowTheKleinBasesTheyWereTunedFrom()
    {
        Flux2Config small = Flux2Variants.ConfigFor(Resolved(Flux2Variants.OpenVisionBanana4B));
        Flux2Config big = Flux2Variants.ConfigFor(Resolved(Flux2Variants.OpenVisionBanana9B));

        Assert.Equal((3072, 5, 20, 7680), (small.HiddenSize, small.Depth, small.DepthSingleBlocks, small.ContextInDim));
        Assert.Equal((4096, 8, 24, 12288), (big.HiddenSize, big.Depth, big.DepthSingleBlocks, big.ContextInDim));
        Assert.False(small.GuidanceEmbed);
        Assert.False(big.GuidanceEmbed);
    }

    [Fact]
    public void ItDeclaresTheBaseSamplingTheReleasePublishesAndReferenceEditing()
    {
        Flux2Recipe recipe = new();
        ResolvedModelVariant ovb = Resolved(Flux2Variants.OpenVisionBanana4B);

        Assert.Equal((28, 3.5f), (recipe.DefaultsFor(ovb).Steps, recipe.DefaultsFor(ovb).CfgScale));
        Assert.Equal(ImageFeatures.RefEdit | ImageFeatures.Lora, recipe.SupportsFor(ovb));
        // The distilled Klein it is shaped like keeps its own contract.
        Assert.Equal(8, recipe.DefaultsFor(Resolved(Flux2Variants.Klein4B)).Steps);
        Assert.Equal(recipe.Supports, recipe.SupportsFor(Resolved(Flux2Variants.Klein4B)));
    }

    [Theory]
    [InlineData("open-vision-banana-4b")]
    [InlineData("open-vision-banana-9b")]
    public void TheFluxTwoRecipeServesTheCatalogIdsAndTheCatalogNamesTheReleaseFiles(string id)
    {
        Assert.True(new Flux2Recipe().Matches(id));
        Assert.Same(RecipeRegistry.Resolve("flux2"), RecipeRegistry.Resolve(id));

        CatalogEntry entry = ModelCatalog.Find(id) ?? throw new InvalidOperationException($"{id} is not in the catalog.");
        Assert.All(entry.Assets.Where(a => a.Role == "transformer"), asset =>
        {
            Assert.Equal("FunAILab/Open-Vision-Banana", asset.Repo);
            Assert.Equal(64, asset.Sha256?.Length);
        });
        Assert.Contains(entry.Assets, a => a.Role == "config");
    }

    [Theory]
    [InlineData(647, 489, 640, 480)]    // floored, not rounded: diffusers' (w // 16) * 16
    [InlineData(640, 480, 640, 480)]
    [InlineData(1024, 1024, 1024, 1024)]
    [InlineData(4000, 3000, 1168, 880)] // 4000x3000 -> 1182x886 at one megapixel, then floored
    public void TheReferenceSizeFollowsDiffusersFlooring(int w, int h, int expectedW, int expectedH)
    {
        (int rw, int rh) = Flux2EditRecipePipeline.ReferenceSize(w, h);

        Assert.Equal((expectedW, expectedH), (rw, rh));
        Assert.True((long)rw * rh <= 1024 * 1024);
    }

    /// <summary>The task prompts go to the encoder as typed. The shared tokenizer parses emphasis, tags and escapes, so the
    /// edit pipeline bypasses it; this pins that the verbatim ids are exactly the chat template over the raw string, that the
    /// prompts fit the encoder's 512 tokens, and that even a 27-class legend does.</summary>
    [Theory]
    [InlineData(DepthPrompt)]
    [InlineData(NormalsPrompt)]
    [InlineData("[TASK: SEMANTIC_SEGMENTATION] Generate a semantic segmentation visualization of the input image, using this color mapping: {\"road\": [8,4,249]}")]
    public void TaskPromptsReachTheEncoderVerbatim(string prompt)
    {
        using Qwen3Tokenizer tokenizer = new(maxLength: 512);
        Flux2EditRecipePipeline recipe = new(null!, tokenizer, "open-vision-banana-4b", null!, []);

        Diffusion.Pipelines.IrisZPrompt ids = recipe.Tokenize(prompt);
        (int[] expected, int realLength) = tokenizer.EncodeChatWithLength(prompt);

        Assert.Equal(expected, ids.Tokens);
        Assert.Equal(realLength < expected.Length ? realLength : 0, ids.ValidLength);
        Assert.Equal(512, ids.Tokens.Length);
        Assert.True(realLength < 512, $"{realLength} real tokens must stay under the 512 the model reads");
    }

    /// <summary>diffusers tokenizes with <c>truncation=True, max_length=512</c>, which drops the END of the chat template. A
    /// long legend (27 classes with three-digit colours) reaches that cap in the reference as well, so the ids here must be the
    /// first 512 of the untruncated encoding, with no padding after them.</summary>
    [Fact]
    public void ALongLegendIsTruncatedLikeTheReferenceDoes()
    {
        string[] classes = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain",
            "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle", "bench", "bird", "cat", "dog", "horse",
            "sheep", "cow", "umbrella"];
        StringBuilder legend = new();
        for (int i = 0; i < classes.Length; i++)
            legend.Append(i == 0 ? "" : ", ").Append("'").Append(classes[i]).Append("': [").Append(i * 9 % 256).Append(", ").Append(255 - i * 7).Append(", ").Append(i * 31 % 256).Append(']');
        string prompt = "[TASK: SEMANTIC_SEGMENTATION] Generate a semantic segmentation visualization of the input image, using this color mapping for the listed classes: { "
            + legend + " }. Paint pixels belonging to a listed class with exactly its RGB color. Pixels that do not belong to one of these listed classes must be black [0, 0, 0] "
            + "and must not be forced into a listed class. Output only the segmentation visualization image.";
        using Qwen3Tokenizer capped = new(maxLength: 512);
        using Qwen3Tokenizer uncapped = new(maxLength: 4096);

        (int[] ids, int realLength) = capped.EncodeChatWithLength(prompt);
        (int[] full, int fullLength) = uncapped.EncodeChatWithLength(prompt);

        Assert.True(fullLength > 512, $"{fullLength} tokens: this legend must overflow for the test to mean anything");
        Assert.Equal(512, realLength);
        Assert.Equal(full[..512], ids);
    }

    [Theory]
    [InlineData(3072, "OVB-4B-zeroshot", "open-vision-banana-4b", null)]
    [InlineData(3072, "OVB-4B-zeroshot", "open-vision-banana-4b", "flux.2-klein-4b")]
    [InlineData(4096, "OVB-9B-zeroshot", "open-vision-banana-9b", "flux.2-klein-9b")]
    [InlineData(3072, "Open-Vision-Banana-ovb-4b-checkpoint", "open-vision-banana-4b", null)]
    [InlineData(3072, "flux-2-klein-4b", "klein-4b", null)]
    [InlineData(4096, "flux-2-klein-9b", "klein-9b", "flux.2-klein-9b")]
    public void ALocalFolderNamedForTheFineTuneSelectsItEvenWhenTheHostOnlyKnowsTheBaseClass(int hidden, string folder, string expected, string? hint)
    {
        ResolvedModelVariant variant = ModelVariantResolver.Classify(Flux2Variants.Catalog, Probe(hidden, folder), hint is null ? [] : [hint]);

        Assert.Equal(expected, variant.Id);
        if (expected.StartsWith("open-vision-banana", StringComparison.Ordinal))
            Assert.Equal(ModelVariantSource.Filename, variant.Source);
    }

    [Fact]
    public void AnExplicitHintForTheFineTuneStillWinsOverAFolderNamedForTheBase()
    {
        ResolvedModelVariant variant = ModelVariantResolver.Classify(Flux2Variants.Catalog, Probe(3072, "flux-2-klein-4b"), ["open-vision-banana-4b"]);

        Assert.Equal("open-vision-banana-4b", variant.Id);
        Assert.Equal(ModelVariantSource.CallerHint, variant.Source);
    }

    [Fact]
    public void ItAppliesNoPromptWeightingAndDoesNotAdvertiseIt()
    {
        Flux2Recipe recipe = new();
        ResolvedModelVariant ovb = Resolved(Flux2Variants.OpenVisionBanana4B);
        ResolvedModelVariant klein = Resolved(Flux2Variants.Klein4B);

        Assert.Equal(PromptWeightingMode.None, recipe.PromptWeightingFor(ovb));
        Assert.Equal(PromptWeightingMode.CondScale, recipe.PromptWeightingFor(klein));
        Assert.Equal(PromptWeightingMode.CondScale, recipe.PromptWeightingFor(null));
        Assert.DoesNotContain(ImageFeatures.PromptWeighting, Enumerate(HartsyInference.Engine.Catalog.RecipeResolution.ImageFeaturesFor(recipe, ovb)));
        Assert.Contains(ImageFeatures.PromptWeighting, Enumerate(HartsyInference.Engine.Catalog.RecipeResolution.ImageFeaturesFor(recipe, klein)));
    }

    /// <summary>What <c>ImagesService</c> does to a prompt before the pipeline sees it, under the mode OVB declares: the README task
    /// prompts come through unchanged, parentheses, brackets, braces and all.</summary>
    [Theory]
    [InlineData(DepthPrompt)]
    [InlineData(NormalsPrompt)]
    [InlineData("[TASK: SEMANTIC_SEGMENTATION] Generate a semantic segmentation visualization of the input image, using this color mapping: {\"road\": [8,4,249], \"sky\": [2,204,254]} Color each pixel of a listed class with its color and leave everything else black.")]
    [InlineData("[TASK: REFERRING_SEGMENTATION] Segment the region that best matches this description: \"table in the room\". Paint the matching region red [220, 30, 30] and leave everything else black [0, 0, 0].")]
    public void TheServiceLeavesTaskPromptsUntouched(string prompt)
    {
        PromptWeightingMode mode = new Flux2Recipe().PromptWeightingFor(Resolved(Flux2Variants.OpenVisionBanana4B));

        Assert.Equal(prompt, PromptFeatureFlattening.Prepare(prompt, mode, preserveScheduling: false));
    }

    [Theory]
    [InlineData("F32", true, "F16")]
    [InlineData("BF16", true, "F16")]
    [InlineData("F16", true, "F16")]
    [InlineData("F32", false, "F32")]
    [InlineData("BF16", false, "F16")]
    public void FloatTensorsNarrowOnLoadAndOnlyOvbNarrowsFp32(string stored, bool ovb, string expected)
    {
        ResolvedModelVariant variant = Resolved(ovb ? Flux2Variants.OpenVisionBanana9B : Flux2Variants.Klein9B);

        Assert.Equal(DTypeNamed(expected), Flux2Recipe.LoadedDType(variant, DTypeNamed(stored)));
    }

    [Theory]
    [InlineData(647, 489, 640, 480, 640, 483, 0, -1)]   // diffusers: 489*640//647 = 483, pasted one row up
    [InlineData(489, 647, 480, 640, 483, 640, -1, 0)]
    [InlineData(1000, 500, 992, 496, 992, 496, 0, 0)]
    [InlineData(640, 480, 640, 480, 640, 480, 0, 0)]
    public void TheCoverGeometryIsDiffusersResizeAndCrop(int srcW, int srcH, int w, int h, int scaledW, int scaledH, int offsetX, int offsetY)
    {
        Assert.Equal((scaledW, scaledH, offsetX, offsetY), Flux2EditRecipePipeline.CoverGeometry(srcW, srcH, w, h));
    }

    [Fact]
    public void ThePhotoBecomesAOneByThreeByHeightByWidthTensorAtTheFlooredSize()
    {
        byte[] rgb = new byte[647 * 489 * 3];
        for (int i = 0; i < rgb.Length; i++) rgb[i] = (byte)(i % 251);

        using Tensor tensor = Flux2EditRecipePipeline.ToReferenceTensor(new HartsyInference.Engine.Requests.ImageData { Rgb = rgb, Width = 647, Height = 489 });

        Assert.Equal(new TensorShape(1, 3, 480, 640), tensor.Shape);
        ReadOnlySpan<float> values = tensor.AsReadOnlySpan<float>();
        Assert.All(values.ToArray(), v => Assert.InRange(v, -1f, 1f));
    }

    private static DType DTypeNamed(string name) =>
        (DType)(typeof(DType).GetProperty(name)?.GetValue(null) ?? typeof(DType).GetField(name)!.GetValue(null)!);

    private static IEnumerable<ImageFeatures> Enumerate(ImageFeatures features) =>
        Enum.GetValues<ImageFeatures>().Where(f => f != ImageFeatures.None && (features & f) == f);

    private static CheckpointProbe Probe(int hidden, string folder)
    {
        Dictionary<string, TensorShape> shapes = new(StringComparer.Ordinal)
        {
            ["x_embedder.weight"] = new TensorShape(hidden, 128),
        };
        return new CheckpointProbe(new HashSet<string>(shapes.Keys, StringComparer.Ordinal), shapes,
            new Dictionary<string, string>(), [folder, "diffusion_pytorch_model-00001-of-00002"]);
    }

    private static ResolvedModelVariant Resolved(ModelVariant variant) =>
        new("flux2", variant, ModelVariantSource.Structure, "test");

    private static CheckpointProbe Probe(int hidden)
    {
        Dictionary<string, TensorShape> shapes = new(StringComparer.Ordinal)
        {
            ["x_embedder.weight"] = new TensorShape(hidden, 128),
        };
        return new CheckpointProbe(new HashSet<string>(shapes.Keys, StringComparer.Ordinal), shapes,
            new Dictionary<string, string>(), []);
    }
}
