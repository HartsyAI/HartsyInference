using HartsyInference.Engine.Recipes;
using HartsyInference.ModelAssets.Metadata;
using Xunit;

namespace HartsyInference.Engine.Tests;

/// <summary>The registry validator refuses a recipe whose prompt block leaves the vocabulary or names the wrong modality, so a bad
/// declaration fails when the registry loads rather than reaching a client.</summary>
public sealed class RecipePromptValidatorTests
{
    private static readonly PromptProfile _valid = new()
    {
        Kind = "image", Encoders = ["clip"], Format = "tags", Negative = "separate", Supports = ["weights"],
        Guide = "docs/prompting/sdxl.md",
    };

    [Fact]
    public void AValidProfileIsAccepted()
    {
        Assert.Empty(RecipeRegistryValidator.Problems(new StubImage(_valid)));
        Assert.Empty(RecipeRegistryValidator.Problems(new StubVideo(_valid with { Kind = "video" })));
        Assert.Empty(RecipeRegistryValidator.Problems(new StubImage(null)));
    }

    [Fact]
    public void AValueOutsideTheVocabularyIsReportedWithTheRecipeName()
    {
        IReadOnlyList<string> problems = RecipeRegistryValidator.Problems(new StubImage(_valid with { Format = "yaml" }));

        Assert.Contains(problems, problem => problem.StartsWith("stub: prompt format 'yaml'", StringComparison.Ordinal));
    }

    [Fact]
    public void BoxesWithoutBoxSupportAreReported()
    {
        IReadOnlyList<string> problems = RecipeRegistryValidator.Problems(new StubImage(_valid with { Boxes = "yxyx-1000" }));

        Assert.Contains(problems, problem => problem.Contains("boxes is set but supports does not list 'boxes'", StringComparison.Ordinal));
    }

    [Fact]
    public void AKindThatDisagreesWithTheRecipesModalityIsReportedOnBothRegistries()
    {
        IReadOnlyList<string> image = RecipeRegistryValidator.Problems(new StubImage(_valid with { Kind = "video" }));
        IReadOnlyList<string> video = RecipeRegistryValidator.Problems(new StubVideo(_valid));

        Assert.Contains("stub: prompt kind 'video' is not 'image'.", image);
        Assert.Contains("stub: prompt kind 'image' is not 'video'.", video);
    }

    [Theory]
    [InlineData("sdxl.md")]
    [InlineData("docs/prompting/SDXL.md")]
    [InlineData("docs/prompting/x.txt")]
    [InlineData("../docs/prompting/x.md")]
    [InlineData(" ")]
    public void AGuideThatIsNotADocsPromptingPathIsReported(string guide)
    {
        Assert.Contains(RecipeRegistryValidator.Problems(new StubImage(_valid with { Guide = guide })),
            problem => problem.Contains("is not 'docs/prompting/<name>.md'", StringComparison.Ordinal));
    }

    [Fact]
    public void AnEmptyEncoderListIsReportedBecauseItWouldWriteAnEmptyHeaderValue()
    {
        Assert.Contains(RecipeRegistryValidator.Problems(new StubImage(_valid with { Encoders = [] })),
            problem => problem.Contains("encoders is empty", StringComparison.Ordinal));
    }

    [Fact]
    public void RegisteringAnInvalidImageOrVideoRecipeIsRefused()
    {
        Assert.Throws<ArgumentException>(() => RecipeRegistry.Register(new StubImage(_valid with { Negative = "maybe" })));
        Assert.Throws<ArgumentException>(() => VideoRecipeRegistry.Register(new StubVideo(_valid with { Kind = "video", Negative = "maybe" })));
    }

    private sealed class StubImage(PromptProfile? prompt) : IArchitectureRecipe
    {
        public string Name => "stub";

        public bool Matches(string familyId) => familyId == "stub";

        public IRecipePipeline Construct(RecipeContext context) => throw new NotSupportedException();

        public ImageDefaults Defaults { get; } = new() { Steps = 20, CfgScale = 5f, Width = 512, Height = 512 };

        public PromptProfile? Prompt { get; } = prompt;
    }

    private sealed class StubVideo(PromptProfile? prompt) : IVideoRecipe
    {
        public string Name => "stub";

        public bool Matches(string familyId) => familyId == "stub";

        public IVideoRecipePipeline Construct(RecipeContext context) => throw new NotSupportedException();

        public VideoDefaults Defaults { get; } = new() { Steps = 20, CfgScale = 5f, Width = 512, Height = 512, Frames = 81, Fps = 16 };

        public VideoConstraints Constraints { get; } = new() { Frames = FrameGrid.StepPlusOne(4, FrameNormalization.SnapNearest) };

        public PromptProfile? Prompt { get; } = prompt;
    }
}
