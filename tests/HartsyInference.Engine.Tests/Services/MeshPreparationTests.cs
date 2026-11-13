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

    /// <summary>A matter that calls the middle half of any image foreground.</summary>
    private sealed class CentreMatter(bool available = true) : IForegroundMatter
    {
        public bool IsAvailable => available;

        public int Calls { get; private set; }

        public float[] Alpha(IBackend backend, ReadOnlySpan<byte> rgb, int width, int height)
        {
            Calls++;
            float[] alpha = new float[width * height];
            for (int y = height / 4; y < height * 3 / 4; y++)
            {
                for (int x = width / 4; x < width * 3 / 4; x++)
                    alpha[y * width + x] = 1f;
            }
            return alpha;
        }
    }

    private static ImageData Photo(int width, int height) => new()
    {
        Rgb = [.. Enumerable.Range(0, width * height * 3).Select(i => (byte)(i * 7 % 251))],
        Width = width,
        Height = height,
    };

    private static PreparedMeshImage RunOwnCrop(string id, ImageData image, IForegroundMatter matter)
    {
        using InferenceEngine engine = new("cpu");
        return MeshPreparationService.Prepare(id, image, matter, engine.Backend, CancellationToken.None);
    }

    [Theory]
    [InlineData("trellis2", 1.0)]
    [InlineData("pixal3d", 1.1)]
    public void TheOwnCropOfTheModelIsTheImageAndTheMatteTheCutout(string id, double cropScale)
    {
        ImageData photo = Photo(64, 64);
        CentreMatter matter = new();

        PreparedMeshImage result = RunOwnCrop(id, photo, matter);

        using InferenceEngine engine = new("cpu");
        Trellis2PreparedImage pipelineCrop = Trellis2ImagePreprocessor.Prepare(
            engine.Backend, new() { Rgb = photo.Rgb, Width = 64, Height = 64 }, new CentreMatter(), cropScale);
        Assert.Equal(pipelineCrop.Rgb, result.Image.Rgb);
        Assert.Equal((pipelineCrop.Width, pipelineCrop.Height), (result.Image.Width, result.Image.Height));
        Assert.False(result.Image.HasAlpha);
        Assert.True(result.Matted);
        Assert.Equal(1, matter.Calls);
        Assert.Equal(photo.Rgb, result.Cutout!.Rgb);
        Assert.Equal(255, result.Cutout.Alpha![32 * 64 + 32]);
        Assert.Equal(0, result.Cutout.Alpha[0]);
        Assert.Contains("BiRefNet matted it", result.Notes[0]);
    }

    [Fact]
    public void Pixal3DWidensTheCropAndTrellis2DoesNot()
    {
        ImageData photo = Photo(64, 64);

        PreparedMeshImage trellis = RunOwnCrop("trellis2", photo, new CentreMatter());
        PreparedMeshImage pixal = RunOwnCrop("pixal3d", photo, new CentreMatter());

        Assert.True(pixal.Image.Width > trellis.Image.Width);
    }

    [Fact]
    public void AnImageWithAlphaIsCroppedWithItsOwnAlphaAndNotMatted()
    {
        ImageData photo = Photo(64, 64);
        ImageData cutout = photo with { Alpha = [.. Enumerable.Range(0, 64 * 64).Select(i => i % 64 is >= 16 and < 48 && i / 64 is >= 16 and < 48 ? (byte)255 : (byte)0)] };
        CentreMatter matter = new();

        PreparedMeshImage result = RunOwnCrop("trellis2", cutout, matter);

        Assert.Equal(0, matter.Calls);
        Assert.False(result.Matted);
        Assert.Equal(cutout.Alpha, result.Cutout!.Alpha);
        Assert.DoesNotContain(result.Notes, n => n.Contains("BiRefNet"));
    }

    [Fact]
    public void ALargeImageIsShrunkToThePipelinesLongEdgeAndTheCutoutSaysSo()
    {
        ImageData photo = Photo(2048, 1024);
        CentreMatter matter = new();

        PreparedMeshImage result = RunOwnCrop("trellis2", photo, matter);

        Assert.Equal((1024, 512), (result.Cutout!.Width, result.Cutout.Height));
        Assert.Equal(1024 * 512, result.Cutout.Alpha!.Length);
        Assert.Contains(result.Notes, n => n.Contains("1024 x 512"));
    }

    [Fact]
    public void ALargeImageWithAlphaKeepsItsShrunkAlphaAsTheCutoutAndIsNotMatted()
    {
        ImageData photo = Photo(2048, 1024) with { Alpha = [.. Enumerable.Range(0, 2048 * 1024).Select(i => i % 2048 is >= 512 and < 1536 && i / 2048 is >= 256 and < 768 ? (byte)255 : (byte)0)] };
        CentreMatter matter = new();

        PreparedMeshImage result = RunOwnCrop("trellis2", photo, matter);

        Assert.Equal(0, matter.Calls);
        Assert.False(result.Matted);
        Assert.Equal((1024, 512), (result.Cutout!.Width, result.Cutout.Height));
        Assert.Equal(255, result.Cutout.Alpha![256 * 1024 + 512]);
        Assert.Equal(0, result.Cutout.Alpha[0]);
    }

    [Theory]
    [InlineData("trellis2")]
    [InlineData("pixal3d")]
    public void ACutoutSentBackIsNotMattedAgainAndGivesTheSameCrop(string id)
    {
        CentreMatter matter = new();
        PreparedMeshImage first = RunOwnCrop(id, Photo(64, 64), matter);

        PreparedMeshImage second = RunOwnCrop(id, first.Cutout!, matter);

        Assert.Equal(1, matter.Calls);
        Assert.False(second.Matted);
        Assert.Equal(first.Image.Rgb, second.Image.Rgb);
        Assert.Equal((first.Image.Width, first.Image.Height), (second.Image.Width, second.Image.Height));
    }

    [Theory]
    [InlineData("trellis2")]
    [InlineData("pixal3d")]
    public void WithoutTheMatteAndWithoutAlphaTheOwnCropIsRefused(string id)
        => Assert.Throws<NotSupportedException>(() => RunOwnCrop(id, Photo(64, 64), new CentreMatter(available: false)));

    [Theory]
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
