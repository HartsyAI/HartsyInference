using HartsyInference.Core.Backends;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ThreeD.Pipelines.Stages;

namespace HartsyInference.Engine.Tests.Services;

public sealed class MeshPreparationTests
{
    /// <summary>A matter that calls the middle 4 x 4 block of an 8 x 8 image foreground.</summary>
    private sealed class BlockMatter(bool available = true) : IForegroundMatter
    {
        public bool IsAvailable => available;

        public int Calls { get; private set; }

        public float[] Alpha(IBackend backend, ReadOnlySpan<byte> rgb, int width, int height)
        {
            Calls++;
            float[] alpha = new float[width * height];
            for (int y = 2; y < 6; y++)
            {
                for (int x = 2; x < 6; x++)
                    alpha[y * width + x] = 1f;
            }
            return alpha;
        }
    }

    private static ImageData Plain() => new() { Rgb = Enumerable.Repeat((byte)255, 8 * 8 * 3).ToArray(), Width = 8, Height = 8 };

    private static ImageData WithAlpha() => Plain() with { Alpha = [.. Enumerable.Range(0, 64).Select(i => i % 8 is >= 2 and < 6 && i / 8 is >= 2 and < 6 ? (byte)255 : (byte)0)] };

    private static PreparedMeshImage Run(string id, ImageData image, IForegroundMatter matter)
        => MeshPreparationService.Prepare(id, image, matter, null!, CancellationToken.None);

    [Fact]
    public void Hunyuan3DReceivesA512ImageRecentredOnWhiteAndKeepsTheCutout()
    {
        PreparedMeshImage result = Run("hunyuan3d", WithAlpha(), new BlockMatter());

        Assert.Equal((512, 512), (result.Image.Width, result.Image.Height));
        Assert.False(result.Matted);
        Assert.Equal(3, result.Image.Rgb.Length / (512 * 512));
        Assert.NotNull(result.Cutout);
    }

    [Fact]
    public void AnImageWithoutAlphaIsMattedFirstAndTheMatteIsReturnedAsTheCutout()
    {
        BlockMatter matter = new();

        PreparedMeshImage result = Run("triposr", Plain(), matter);

        Assert.Equal(1, matter.Calls);
        Assert.True(result.Matted);
        Assert.True(result.Cutout!.HasAlpha);
        Assert.Equal(255, result.Cutout.Alpha![3 * 8 + 3]);
        Assert.Equal(0, result.Cutout.Alpha[0]);
        Assert.Equal(result.Image.Width, result.Image.Height);
    }

    [Fact]
    public void AnRgbaImageWhoseAlphaIsEntirelyOpaqueIsMatted()
    {
        BlockMatter matter = new();
        ImageData opaque = Plain() with { Alpha = Enumerable.Repeat((byte)255, 64).ToArray() };

        PreparedMeshImage result = Run("hunyuan3d", opaque, matter);

        Assert.Equal(1, matter.Calls);
        Assert.True(result.Matted);
        Assert.False(MeshPreparationService.HasUsableAlpha(opaque));
    }

    [Fact]
    public void AnAlphaPlaneWithATransparentBorderIsUsedAsIs()
    {
        BlockMatter matter = new();

        PreparedMeshImage result = Run("hunyuan3d", WithAlpha(), matter);

        Assert.Equal(0, matter.Calls);
        Assert.False(result.Matted);
        Assert.True(MeshPreparationService.HasUsableAlpha(WithAlpha()));
    }

    [Fact]
    public void AnEmptyAlphaPlaneIsAnEmptyCutoutNotAMissingOne()
    {
        BlockMatter matter = new();
        ImageData empty = Plain() with { Alpha = new byte[64] };

        Assert.True(MeshPreparationService.HasUsableAlpha(empty));
        Assert.Throws<ArgumentException>(() => Run("hunyuan3d", empty, matter));
        Assert.Equal(0, matter.Calls);
    }

    [Fact]
    public void AnImageWithoutAnyAlphaPlaneHasNoUsableAlpha()
        => Assert.False(MeshPreparationService.HasUsableAlpha(Plain()));

    [Fact]
    public void AutoNeedsTheMatteForAnOpaqueRgbaImageToo()
    {
        MeshRequest request = new() { Image = Plain() with { Alpha = Enumerable.Repeat((byte)255, 64).ToArray() }, Preprocess = MeshPreprocess.Auto };

        Assert.Throws<NotSupportedException>(() => MeshRequestValidator.Validate("triposr", request, matterAvailable: false));
        MeshRequestValidator.Validate("triposr", request, matterAvailable: true);
    }

    [Fact]
    public void AnImageWithAnAlphaPlaneIsNotMattedAgain()
    {
        BlockMatter matter = new();

        Run("trellis", WithAlpha(), matter);

        Assert.Equal(0, matter.Calls);
    }

    [Fact]
    public void TrellisReceivesA518ImageWithoutAlpha()
    {
        PreparedMeshImage result = Run("trellis", WithAlpha(), new BlockMatter());

        Assert.Equal((518, 518), (result.Image.Width, result.Image.Height));
        Assert.False(result.Image.HasAlpha);
    }

    [Fact]
    public void PaintKeepsTheCutoutAsAlphaForItsOwnCompositing()
    {
        PreparedMeshImage result = Run("hunyuan3d-paint21", Plain(), new BlockMatter());

        Assert.True(result.Image.HasAlpha);
        Assert.True(result.Matted);
    }

    [Fact]
    public void WithoutTheMatteAnImageWithoutAlphaIsRefused()
        => Assert.Throws<NotSupportedException>(() => Run("hunyuan3d", Plain(), new BlockMatter(available: false)));

    [Theory]
    [InlineData("trellis2")]
    [InlineData("pixal3d")]
    [InlineData("hairdeck")]
    [InlineData("sam3d-objects")]
    public void ModelsThatPrepareTheirOwnInputGetTheImageBackUnchanged(string id)
    {
        ImageData image = Plain();
        BlockMatter matter = new();

        PreparedMeshImage result = Run(id, image, matter);

        Assert.Same(image, result.Image);
        Assert.Equal(0, matter.Calls);
        Assert.Null(result.Cutout);
        Assert.Single(result.Notes);
    }

    [Fact]
    public void ModelsThatTakeAMeshSayTheyTakeNoImage()
    {
        PreparedMeshImage result = Run("unirig", Plain(), new BlockMatter());

        Assert.Contains("takes no image", result.Notes[0]);
    }

    [Fact]
    public void AutoNeedsAlphaOrTheMatteForTheModelsThatPrepare()
    {
        MeshRequest request = new() { Image = Plain(), Preprocess = MeshPreprocess.Auto };

        NotSupportedException error = Assert.Throws<NotSupportedException>(() => MeshRequestValidator.Validate("hunyuan3d", request, matterAvailable: false));

        Assert.Equal(nameof(MeshRequest.Preprocess), MeshFieldTag.FieldOf(error));
        MeshRequestValidator.Validate("hunyuan3d", request, matterAvailable: true);
        MeshRequestValidator.Validate("hunyuan3d", request with { Image = WithAlpha() }, matterAvailable: false);
        MeshRequestValidator.Validate("hunyuan3d", request with { Preprocess = MeshPreprocess.None }, matterAvailable: false);
    }

    [Fact]
    public async Task PlanWarnsThatAnAutoPreprocessIsIgnoredByModelsThatPrepareTheirOwnInput()
    {
        using InferenceEngine engine = new("cpu");
        MeshRequest request = new() { Image = WithAlpha(), Preprocess = MeshPreprocess.Auto };

        MeshPlan plan = await engine.MeshPlanning.PlanAsync(
            new HartsyInference.Engine.Dispatch.ModelSpec
            {
                Requested = "trellis2",
                Modality = HartsyInference.Engine.Modality.Mesh,
                Catalog = HartsyInference.Engine.Registry.ModelCatalog.Find("trellis2"),
                LocalPath = Path.GetTempPath(),
            },
            request);

        Assert.Contains(plan.Issues, i => i.Code == "mesh.field.ignored" && i.Field == nameof(MeshRequest.Preprocess));
        Assert.True(plan.IsValid);
    }

    [Fact]
    public void ThePreparationIsDescribedForTheModelsThatUseItAndIgnoredForTheOthers()
    {
        using InferenceEngine engine = new("cpu");

        MeshFieldSpec used = engine.MeshPlanning.Describe("hunyuan3d").Fields[nameof(MeshRequest.Preprocess)];
        MeshFieldSpec ignored = engine.MeshPlanning.Describe("trellis2").Fields[nameof(MeshRequest.Preprocess)];

        Assert.Equal(MeshFieldAccess.Used, used.Access);
        Assert.Equal(["None", "Auto"], used.Values);
        Assert.Equal("None", used.Default);
        Assert.Equal(MeshFieldAccess.Ignored, ignored.Access);
    }
}
