using HartsyInference.Engine.Recipes;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Pins which families declare <see cref="ImageFeatures.Img2Img"/>, because that single bit is what decides
/// whether SwarmUI routes an init-image request to this backend or refuses it. A family that gains the bit without its
/// recipe pipeline actually binding the init image would silently generate text-to-image — the exact failure mode
/// Mage-Flow shipped with — so this list is meant to be edited deliberately, one family at a time, as each is wired
/// and verified.</summary>
public sealed class ImageFeatureDeclarationTests
{
    /// <summary>Families whose recipe pipeline binds <c>request.Img2Img</c> through <c>RecipeImg2ImgBinder</c>.</summary>
    private static readonly string[] ExpectedImg2Img =
    [
        "sd15", "sdxl",                                                 // Phase 0 (pre-existing)
        "sd3", "zimage", "qwen-image", "zeta-chroma", "chroma-radiance", // Phase 1
        // Phase 2, grouped by the VAE whose encode-parity gate covers them. These are engine family ids
        // (IArchitectureRecipe.Name), not SwarmUI compat-class ids — "flux2", not "flux-2".
        "krea2", "anima",                                       // QwenImageVaeEncoder
        "chroma", "flux1", "lumina2", "kandinsky5", "f-lite",   // VaeConfig.Flux
        "flux2", "ernie-image", "ideogram4",                    // VaeConfig.Flux2
        "auraflow",                                             // VaeConfig.AuraFlow (= Sdxl)
        "lance-image",                                          // Wan22VaeEncoder — img2img only
        // Phase 4 — built from nothing; these three had no img2img at any layer
        "hidream",                                              // VaeConfig.Flux, full masked path
        "hunyuan-image", "lens",                                // token-space loops (masked path added in 72b725ae)
        "sdxl-refiner",                                         // fb306df2 — drivable standalone, img2img by nature
    ];

    /// <summary>Reference-image editing: the init image becomes in-context reference latents rather than a noised
    /// start, so <c>Creativity</c> has nothing to select. Deliberately a separate bit — Mage-Flow used to declare
    /// <see cref="ImageFeatures.Img2Img"/> and would silently accept a creativity value it cannot honour.</summary>
    private static readonly string[] ExpectedRefEdit =
    [
        "mage-flow", "omnigen2", "boogu",
        "qwen-image",   // the only family offering both modes; Img2Img.Mode selects
    ];

    /// <summary>Inpaint additionally requires a masked path in the diffusion pipeline. Mage-Flow has none, and
    /// <c>sdxl-refiner</c> is a second-pass model with no mask blend of its own.</summary>
    private static readonly string[] ExpectedInpaint =
    [
        "sd15", "sdxl",
        "sd3", "zimage", "qwen-image", "zeta-chroma", "chroma-radiance",
        "krea2", "anima",
        "chroma", "flux1", "lumina2", "kandinsky5", "f-lite",
        "flux2", "ernie-image", "ideogram4",
        "auraflow",
        "hidream",
        // 72b725ae — per-token blend (MaskBlendUtilities.BlendTokensInPlace) covering all three token-space
        // loops. Lance's old "not supported" throw went with it: the /16 mask granularity it cited is a
        // coarseness, not a blocker.
        "hunyuan-image", "lens", "lance-image",
    ];

    /// <summary>Every family whose recipe calls <c>LoraApplier.BuildAndApply</c> before its transformer's
    /// <c>LoadWeights</c>. As of 2026-08-20 that is every image family except <c>sdxl-refiner</c>.
    /// <para>Declaring the bit without wiring the merge is the failure this pin exists for, and it is silent in the
    /// worst way: the request passes the feature gate, the LoRA is never merged, and the user gets a normal image
    /// they read as "the LoRA is too weak". That is the exact regression commit <c>fc975b71</c> hardened
    /// <c>LoraApplier</c> against for the zero-key-match case; this covers the no-call-at-all case, which no runtime
    /// check can catch because nothing ever asks.</para>
    /// <para><c>sdxl-refiner</c> is excluded deliberately, not pending — see the reasoning on
    /// <c>SdxlRefinerRecipe.Supports</c>. Its UNet layout means an SDXL LoRA names nothing in it.</para></summary>
    private static readonly string[] ExpectedLora =
    [
        "sd15", "sdxl", "flux1", "sd3",                                     // Phase 0 (pre-existing)
        "qwen-image", "anima", "chroma", "lumina2", "hidream",              // fc975b71
        "flux2", "zimage", "omnigen2", "zeta-chroma", "chroma-radiance",    // fc975b71
        // 2026-08-20 sweep. The five marked (*) needed the bare-root LoRA format detector widened past its old
        // block-root allow-list before their files could even be recognized at load.
        "krea2", "ideogram4",                                               // (*) ideogram4: layers.N
        "auraflow",                                                         // (*) joint_transformer_blocks.N
        "f-lite", "ernie-image",                                            // (*) ernie-image: layers.N
        "kandinsky5",                                                       // (*) text_/visual_transformer_blocks.N
        "lance-image",                                                      // (*) layers.N
        "boogu",                                                            // (*) double_stream_layers.N
        "hunyuan-image", "lens", "mage-flow",
    ];

    private readonly ITestOutputHelper _output;

    public ImageFeatureDeclarationTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void LoraIsDeclaredByExactlyTheFamiliesThatMergeIt()
    {
        string[] actual = DeclaringFamilies(ImageFeatures.Lora);
        _output.WriteLine($"lora: {string.Join(", ", actual)}");
        Assert.Equal([.. ExpectedLora.Order(StringComparer.Ordinal)], actual);
    }

    [Fact]
    public void Img2ImgIsDeclaredByExactlyTheWiredFamilies()
    {
        string[] actual = DeclaringFamilies(ImageFeatures.Img2Img);
        _output.WriteLine($"img2img: {string.Join(", ", actual)}");
        Assert.Equal([.. ExpectedImg2Img.Order(StringComparer.Ordinal)], actual);
    }

    [Fact]
    public void RefEditIsDeclaredByExactlyTheEditModels()
    {
        string[] actual = DeclaringFamilies(ImageFeatures.RefEdit);
        _output.WriteLine($"refedit: {string.Join(", ", actual)}");
        Assert.Equal([.. ExpectedRefEdit.Order(StringComparer.Ordinal)], actual);
    }

    /// <summary>An edit model must not also claim strength-based img2img unless it genuinely implements both, because
    /// the two obey different contracts and <c>Img2ImgMode.Auto</c> resolves ambiguity in favour of the classic path.</summary>
    [Fact]
    public void QwenImageIsTheOnlyFamilyDeclaringBothInitImageModes()
    {
        string[] both = [.. DeclaringFamilies(ImageFeatures.RefEdit)
            .Where(f => (Supports(f) & ImageFeatures.Img2Img) != ImageFeatures.None)];
        Assert.Equal(["qwen-image"], both);
    }

    [Fact]
    public void InpaintIsDeclaredByExactlyTheFamiliesWithAMaskedPath()
    {
        string[] actual = DeclaringFamilies(ImageFeatures.Inpaint);
        _output.WriteLine($"inpaint: {string.Join(", ", actual)}");
        Assert.Equal([.. ExpectedInpaint.Order(StringComparer.Ordinal)], actual);
    }

    /// <summary>Inpaint without img2img is incoherent — a mask re-paints an existing image, so the init image is
    /// mandatory. The composition plan enforces this per request; this catches a recipe that declares the pair wrong.</summary>
    [Fact]
    public void NoFamilyDeclaresInpaintWithoutImg2Img()
    {
        foreach (string family in DeclaringFamilies(ImageFeatures.Inpaint))
        {
            ImageFeatures supports = RecipeRegistry.Resolve(family)!.Supports;
            Assert.True((supports & ImageFeatures.Img2Img) != 0,
                $"{family} declares Inpaint without Img2Img; an inpaint mask requires an init image.");
        }
    }

    /// <summary>Test doubles registered by other fixtures share this prefix. <see cref="RecipeRegistry"/> is a static
    /// list with no removal, and xunit runs the whole assembly in one process, so a fake registered by
    /// <see cref="RecipeImg2ImgBinderTests"/> is visible here depending on test order — this filter keeps the pin about
    /// production families only, rather than making it order-dependent.</summary>
    private const string TestDoublePrefix = "test-";

    private static string[] DeclaringFamilies(ImageFeatures feature) =>
        [.. RecipeRegistry.RegisteredNames
            .Distinct(StringComparer.Ordinal)
            .Where(name => !name.StartsWith(TestDoublePrefix, StringComparison.Ordinal))
            .Where(name => (Supports(name) & feature) != ImageFeatures.None)
            .Order(StringComparer.Ordinal)];

    private static ImageFeatures Supports(string family) =>
        RecipeRegistry.Resolve(family)?.Supports ?? ImageFeatures.None;
}
